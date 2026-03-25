"""
PyTorch training backend.

This backend keeps the project data pipeline and CLI style close to the legacy
NumPy trainer, but runs the model, gradients, optimizer, and checkpointing in
PyTorch. The original NumPy trainer remains available under `--backend numpy`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

import config
from backends.torch.model import TorchCNN
from data.loaders import load_image
from data.preprocessing import preprocess_image, resize
from utils.safety import install_dataset_write_guard, tree_signature

install_dataset_write_guard()

try:
    import torch
    from torch import nn
except Exception as exc:
    raise SystemExit(f"PyTorch backend is unavailable in this interpreter: {exc}") from exc


def cosine_lr(base_lr: float, step: int, total_steps: int, min_lr_ratio: float = 0.1) -> float:
    if total_steps <= 1:
        return base_lr
    progress = min(max(step / max(1, total_steps - 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine))


def random_resized_crop_batch(
    x: np.ndarray,
    rng: np.random.Generator,
    scale: tuple[float, float] = (0.6, 1.0),
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
) -> np.ndarray:
    """Apply the same crop policy as the legacy trainer before converting to torch."""
    batch_size, height, width, _ = x.shape
    out = np.empty_like(x)
    for index in range(batch_size):
        area = height * width
        applied = False
        for _ in range(10):
            target_area = float(area) * float(rng.uniform(scale[0], scale[1]))
            log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
            aspect = float(np.exp(rng.uniform(*log_ratio)))
            crop_h = int(round(np.sqrt(target_area * aspect)))
            crop_w = int(round(np.sqrt(target_area / aspect)))
            if 1 <= crop_h <= height and 1 <= crop_w <= width:
                y0 = int(rng.integers(0, max(1, height - crop_h + 1)))
                x0 = int(rng.integers(0, max(1, width - crop_w + 1)))
                crop = x[index, y0 : y0 + crop_h, x0 : x0 + crop_w, :]
                crop_u8 = np.clip(crop * 255.0, 0, 255).astype(np.uint8)
                out[index] = np.asarray(resize(crop_u8, (height, width)), dtype=np.float32) / 255.0
                applied = True
                break
        if not applied:
            out[index] = x[index]
    return out


def augment_batch(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mirror the current NumPy augmentation policy for backend parity."""
    out = x.copy()
    batch_size, height, width, _ = out.shape
    if rng.random() < 0.9:
        out = random_resized_crop_batch(out, rng)
    flip_mask = rng.random(batch_size) < 0.5
    out[flip_mask] = out[flip_mask, :, ::-1, :]

    brightness = 1.0 + rng.uniform(-0.12, 0.12, size=(batch_size, 1, 1, 1))
    contrast = 1.0 + rng.uniform(-0.12, 0.12, size=(batch_size, 1, 1, 1))
    mean = np.mean(out, axis=(1, 2), keepdims=True)
    out = (out - mean) * contrast + mean
    out = out * brightness

    pad = 4
    padded = np.pad(out, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    for index in range(batch_size):
        y0 = int(rng.integers(0, 2 * pad + 1))
        x0 = int(rng.integers(0, 2 * pad + 1))
        out[index] = padded[index, y0 : y0 + height, x0 : x0 + width, :]

    for index in range(batch_size):
        if rng.random() < 0.3:
            cut = int(rng.integers(max(2, height // 8), max(4, height // 4)))
            cy = int(rng.integers(0, height))
            cx = int(rng.integers(0, width))
            y1 = max(0, cy - cut // 2)
            y2 = min(height, cy + cut // 2)
            x1 = max(0, cx - cut // 2)
            x2 = min(width, cx + cut // 2)
            out[index, y1:y2, x1:x2, :] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def list_labeled_paths(
    data_dir: Path,
    fallback_num_classes: int,
    allow_unlabeled_root: bool = False,
) -> tuple[list[Path], np.ndarray, list[str], bool]:
    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if class_dirs:
        paths: list[Path] = []
        labels: list[int] = []
        class_names: list[str] = []
        class_id = 0
        for directory in class_dirs:
            files = [entry for entry in sorted(directory.iterdir()) if entry.suffix.lower() in exts]
            if not files:
                continue
            paths.extend(files)
            labels.extend([class_id] * len(files))
            class_names.append(directory.name)
            class_id += 1
        if paths:
            return paths, np.asarray(labels, dtype=np.int64), class_names, False
    if not allow_unlabeled_root:
        raise SystemExit(
            "No class subdirectories found. Expected `data_dir/class_name/*.jpg` layout. "
            "Use --allow-unlabeled-root to synthesize labels if needed."
        )
    files = [entry for entry in sorted(data_dir.iterdir()) if entry.suffix.lower() in exts]
    if not files:
        raise SystemExit(f"No images found in {data_dir}")
    labels = np.arange(len(files), dtype=np.int64) % fallback_num_classes
    class_names = [str(index) for index in range(fallback_num_classes)]
    return files, labels, class_names, True


def class_distribution(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.bincount(labels, minlength=num_classes).astype(np.int64)


def make_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = class_distribution(labels, num_classes).astype(np.float64)
    weights = np.zeros((num_classes,), dtype=np.float64)
    non_zero = counts > 0
    if not np.any(non_zero):
        return np.ones((num_classes,), dtype=np.float64)
    weights[non_zero] = labels.size / (np.sum(non_zero) * counts[non_zero])
    weights[non_zero] /= np.mean(weights[non_zero])
    return weights


def build_balanced_epoch_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    counts = np.bincount(labels).astype(np.float64)
    class_prob = np.zeros_like(counts)
    non_zero = counts > 0
    class_prob[non_zero] = 1.0 / counts[non_zero]
    sample_prob = class_prob[labels]
    sample_prob /= np.sum(sample_prob)
    return rng.choice(labels.size, size=labels.size, replace=True, p=sample_prob)


def stable_partition_index(path: Path, num_parts: int) -> int:
    if num_parts <= 1:
        return 0
    digest = hashlib.md5(str(path.resolve()).lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_parts


def choose_partition(args) -> int:
    if not args.auto_next_partition:
        return max(0, min(args.partition, max(0, args.num_partitions - 1)))
    state_path = Path(args.partition_state)
    state = {"num_partitions": args.num_partitions, "next": 0}
    try:
        if state_path.is_file():
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if int(loaded.get("num_partitions", -1)) == args.num_partitions:
                state = loaded
    except Exception:
        pass
    part = int(state.get("next", 0)) % max(1, args.num_partitions)
    state["num_partitions"] = args.num_partitions
    state["next"] = (part + 1) % max(1, args.num_partitions)
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass
    return part


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device=cuda requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr_value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr_value


def _to_tensor_batch(x_batch: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x_batch)).to(device=device, dtype=torch.float32)


def _load_batch(paths: list[Path], input_size: tuple[int, int]) -> np.ndarray:
    batch = np.empty((len(paths), input_size[0], input_size[1], 3), dtype=np.float32)
    for index, image_path in enumerate(paths):
        image = load_image(image_path)
        batch[index] = preprocess_image(
            image,
            target_size=input_size,
            normalize_to=config.NORMALIZE_TO,
            input_value_range=config.INPUT_VALUE_RANGE,
        ).astype(np.float32)
    return batch


def evaluate_streaming(
    model: TorchCNN,
    paths: list[Path],
    labels: np.ndarray,
    num_classes: int,
    batch_size: int,
    input_size: tuple[int, int],
    device: torch.device,
) -> tuple[float, float]:
    if len(paths) == 0:
        return float("nan"), float("nan")
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            end = min(start + batch_size, len(paths))
            x_batch = _load_batch(paths[start:end], input_size)
            y_batch = torch.as_tensor(labels[start:end], device=device, dtype=torch.long)
            logits = model(_to_tensor_batch(x_batch, device))
            total_loss += float(criterion(logits, y_batch).item())
            total_correct += int((torch.argmax(logits, dim=1) == y_batch).sum().item())
            num_batches += 1
    return total_loss / max(1, num_batches), total_correct / max(1, len(paths))


def evaluate_preloaded(
    model: TorchCNN,
    x_data: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            x_batch = _to_tensor_batch(x_data[start:end], device)
            y_batch = torch.as_tensor(labels[start:end], device=device, dtype=torch.long)
            logits = model(x_batch)
            total_loss += float(criterion(logits, y_batch).item())
            total_correct += int((torch.argmax(logits, dim=1) == y_batch).sum().item())
            num_batches += 1
    return total_loss / max(1, num_batches), total_correct / max(1, len(labels))


def load_weights_forgiving(model: TorchCNN, checkpoint_path: str | Path, skip_prefixes: tuple[str, ...] = ("fc2.",)) -> tuple[list[str], list[str]]:
    checkpoint = Path(checkpoint_path)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict):
        raise SystemExit(f"Invalid checkpoint format: {checkpoint}")
    current = model.state_dict()
    loaded: list[str] = []
    skipped: list[str] = []
    filtered: dict[str, torch.Tensor] = {}
    for key, target in current.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped.append(key)
            continue
        source = state.get(key)
        if source is None or tuple(source.shape) != tuple(target.shape):
            skipped.append(key)
            continue
        filtered[key] = source
        loaded.append(key)
    model.load_state_dict(filtered, strict=False)
    return loaded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Train image classifier with the PyTorch backend")
    parser.add_argument("--help-md", action="store_true", help="Print Markdown guide and exit")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory with train images")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw", help="Optimizer type")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (for SGD)")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability in classifier head")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--min-lr-ratio", type=float, default=0.2, help="Minimum LR ratio for cosine schedule")
    parser.add_argument("--lr-schedule", choices=["cosine", "step", "constant"], default="cosine", help="LR schedule type")
    parser.add_argument("--step-size", type=int, default=10, help="StepLR epoch interval")
    parser.add_argument("--gamma", type=float, default=0.5, help="StepLR decay factor")
    parser.add_argument("--early-stop-metric", choices=["val_loss", "val_acc"], default="val_loss", help="Metric for early stopping")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Global gradient clip norm (0 disables)")
    parser.add_argument("--class-weighting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balance-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience in epochs")
    parser.add_argument("--min-delta", type=float, default=1e-3, help="Minimum improvement for the monitored metric")
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_torch_model.pt"))
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-unlabeled-root", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="If true, stream batches from disk; if false, preload all images into memory.")
    parser.add_argument("--init-from", type=str, default=None, help="Path to checkpoint .pt/.pth to initialize weights")
    parser.add_argument("--num-partitions", type=int, default=1, help="Split training set into P disjoint shards")
    parser.add_argument("--partition", type=int, default=0, help="Train only on shard id [0..P-1]")
    parser.add_argument("--auto-next-partition", action=argparse.BooleanOptionalAction, default=False, help="Rotate partition id across runs and save state to --partition-state")
    parser.add_argument("--partition-state", type=str, default=str(config.CHECKPOINT_DIR / "partition_state.json"), help="State file for --auto-next-partition")
    parser.add_argument(
        "--class-count",
        type=int,
        default=None,
        help="Total number of classes. Defaults to the detected dataset class count.",
    )
    parser.add_argument("--enforce-readonly-dataset", action=argparse.BooleanOptionalAction, default=True, help="Verify Dataset unchanged before/after run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Training device")
    args = parser.parse_args()

    if args.help_md:
        md_path = Path(__file__).resolve().parents[2] / "Image_Identify_CNN.md"
        try:
            print(md_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"Help file not found or unreadable: {exc}") from exc
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)
    data_dir = Path(args.data_dir) if args.data_dir else Path(config.DATA_DIR)
    data_dir = data_dir.resolve()
    pre_signature = tree_signature(data_dir) if args.enforce_readonly_dataset else None
    rng = np.random.default_rng(args.seed)

    fallback_num_classes = (
        int(args.class_count)
        if args.class_count is not None
        else max(1, config.get_num_classes(data_dir, require_images=True))
    )
    paths, labels, class_names, synthetic_labels = list_labeled_paths(
        data_dir,
        fallback_num_classes=fallback_num_classes,
        allow_unlabeled_root=args.allow_unlabeled_root,
    )
    detected_classes = int(labels.max() + 1) if labels.size else 0
    if args.class_count is not None and args.class_count < detected_classes:
        raise SystemExit(f"--class-count={args.class_count} < detected={detected_classes}")
    num_classes = int(args.class_count) if args.class_count is not None else detected_classes
    resolved_class_names = (
        class_names
        if synthetic_labels
        else config.get_class_names(data_dir, class_count=num_classes, require_images=True)
    )
    print(f"Using num_classes={num_classes} (detected={detected_classes}) on device={device}")
    print(f"Using classes={resolved_class_names}")

    indices = rng.permutation(len(paths))
    num_val = int(max(0.0, min(0.9, args.val_split)) * len(paths))
    val_idx = indices[:num_val]
    train_idx = indices[num_val:]
    train_paths_all = [paths[index] for index in train_idx]
    val_paths = [paths[index] for index in val_idx]
    y_train_all = labels[train_idx]
    y_val = labels[val_idx] if num_val > 0 else None

    num_partitions = max(1, int(args.num_partitions))
    partition = choose_partition(args)
    if partition < 0 or partition >= num_partitions:
        raise SystemExit(f"--partition must be in [0, {num_partitions - 1}]")
    if num_partitions > 1:
        keep_mask = [stable_partition_index(path, num_partitions) == partition for path in train_paths_all]
        train_paths = [path for path, keep in zip(train_paths_all, keep_mask) if keep]
        y_train = y_train_all[np.asarray(keep_mask, dtype=bool)]
        if len(train_paths) == 0:
            raise SystemExit("Selected partition contains no samples; adjust --num-partitions/--partition.")
        print(f"Partitioning: {num_partitions} shards, using shard {partition} -> {len(train_paths)} samples")
    else:
        train_paths = train_paths_all
        y_train = y_train_all

    train_counts = class_distribution(y_train, num_classes)
    print("Train samples:", y_train.size, " class_counts=", train_counts.tolist())
    if y_val is not None:
        val_counts = class_distribution(y_val, num_classes)
        print("Val samples:", y_val.size, " class_counts=", val_counts.tolist())

    class_weights = make_class_weights(y_train, num_classes) if args.class_weighting else None
    class_weights_tensor = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=float(args.label_smoothing))
    input_size = config.INPUT_SIZE

    model = TorchCNN(input_size=input_size, num_classes=num_classes, seed=args.seed, dropout_p=args.dropout)
    model.to(device)
    if args.init_from:
        try:
            model.load_weights(args.init_from, map_location=device)
            print(f"Initialized weights from: {args.init_from}")
        except Exception as exc:
            print(f"[warn] Strict load failed: {exc}")
            loaded, skipped = load_weights_forgiving(model, args.init_from, skip_prefixes=("fc2.",))
            print(f"Loaded {len(loaded)} keys from checkpoint; skipped {len(skipped)} (e.g., classifier head)")
            model.to(device)

    if args.optimizer == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    batch_size = max(1, int(args.batch_size))
    x_train = None
    x_val = None
    if not args.streaming:
        print("Preloading images into memory (float32)...")
        x_train = _load_batch(train_paths, input_size)
        if y_val is not None and len(val_paths) > 0:
            x_val = _load_batch(val_paths, input_size)

    num_batches = (y_train.size + batch_size - 1) // batch_size
    total_steps = max(1, args.epochs * num_batches)
    global_step = 0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_saved = False
    bad_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        permutation = build_balanced_epoch_indices(y_train, rng) if args.balance_sampling else rng.permutation(y_train.size)
        epoch_loss = 0.0
        epoch_correct = 0

        for batch_index in range(num_batches):
            start = batch_index * batch_size
            end = min(start + batch_size, y_train.size)
            idx_batch = permutation[start:end]
            y_batch_idx = y_train[idx_batch]

            if args.streaming:
                x_batch = _load_batch([train_paths[index] for index in idx_batch], input_size)
            else:
                x_batch = x_train[idx_batch]
            if args.augment:
                x_batch = augment_batch(x_batch, rng)

            current_lr = (
                args.lr
                if args.lr_schedule == "constant"
                else cosine_lr(args.lr, global_step, total_steps, min_lr_ratio=args.min_lr_ratio)
                if args.lr_schedule == "cosine"
                else float(args.lr * (args.gamma ** max(0, epoch // max(1, args.step_size))))
            )
            _set_optimizer_lr(optimizer, current_lr)

            optimizer.zero_grad(set_to_none=True)
            x_tensor = _to_tensor_batch(x_batch, device)
            y_tensor = torch.as_tensor(y_batch_idx, dtype=torch.long, device=device)
            logits = model(x_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_correct += int((torch.argmax(logits, dim=1) == y_tensor).sum().item())
            global_step += 1

        avg_loss = epoch_loss / max(1, num_batches)
        train_acc = epoch_correct / max(1, y_train.size)

        if y_val is not None and len(val_paths) > 0:
            if args.streaming or x_val is None:
                val_loss, val_acc = evaluate_streaming(model, val_paths, y_val, num_classes, batch_size, input_size, device)
            else:
                val_loss, val_acc = evaluate_preloaded(model, x_val, y_val, batch_size, device)
            improved = (val_loss < (best_val_loss - args.min_delta)) if args.early_stop_metric == "val_loss" else (val_acc > (best_val_acc + args.min_delta))
            if improved:
                if args.early_stop_metric == "val_loss":
                    best_val_loss = val_loss
                else:
                    best_val_acc = val_acc
                model.save_weights(args.checkpoint)
                best_saved = True
                bad_epochs = 0
            else:
                bad_epochs += 1
            gap = train_acc - val_acc
            marker = " *" if improved else ""
            print(f"Epoch {epoch + 1}/{args.epochs}  lr={current_lr:.6f}  train_loss={avg_loss:.4f}  train_acc={train_acc:.3f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  gap={gap:.3f}{marker}")
            if args.early_stop and bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}: metric {args.early_stop_metric} did not improve by {args.min_delta} for {args.patience} epochs.")
                break
        else:
            print(f"Epoch {epoch + 1}/{args.epochs}  lr={current_lr:.6f}  train_loss={avg_loss:.4f}  train_acc={train_acc:.3f}")

    if y_val is None or len(val_paths) == 0:
        model.save_weights(args.checkpoint)
        best_saved = True
    if best_saved:
        print(f"Saved checkpoint: {args.checkpoint}")

    if args.enforce_readonly_dataset and pre_signature is not None:
        post_signature = tree_signature(data_dir)
        if post_signature != pre_signature:
            raise SystemExit("Dataset changed during run; aborting (read-only enforcement).")

    print("Training finished.")


if __name__ == "__main__":
    main()









