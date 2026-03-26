"""Training script
- Streaming by default (load only current batch); optional preload via --no-streaming
- Resume from checkpoint with forgiving load (skips mismatched classifier head)
- Optional dataset partitioning across runs
- Dataset read-only guard + pre/post signature check
- Markdown help via -help / --help-md
- Dynamic class count: auto-detected from the dataset unless --class-count overrides it
"""

from __future__ import annotations

import argparse
import json
import hashlib
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

import config
from utils.safety import install_dataset_write_guard, tree_signature
from utils.training import (
    augment_batch,
    build_epoch_phase_map,
    compute_phase_learning_rate,
    parse_bool_flag,
    validate_phase_learning_rates,
)
from data.loaders import load_image
from data.preprocessing import preprocess_image
from backends.numpy.model import CNN
from nn.losses import cross_entropy_loss, cross_entropy_loss_backward

# Robust optimizers import with local AdamW fallback
try:
    from nn.optimizers import AdamW as AdamWImpl, SGD as SGDImpl
except Exception:
    try:
        from nn.optimizers import Adam as AdamWImpl, SGD as SGDImpl
    except Exception:
        AdamWImpl = None
        from nn.optimizers import SGD as SGDImpl

if AdamWImpl is None:
    class AdamWImpl:  # Local AdamW fallback (NumPy only)
        def __init__(self, parameters: List[Tuple[np.ndarray, np.ndarray]], lr: float = 1e-3,
                     betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-4):
            beta1, beta2 = betas
            if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
                raise ValueError("betas must satisfy 0 <= beta < 1")
            if eps <= 0.0:
                raise ValueError("eps must be > 0")
            if weight_decay < 0.0:
                raise ValueError("weight_decay must be >= 0")
            self.parameters = parameters
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            self.weight_decay = weight_decay
            self._m = [np.zeros_like(p) for p, _ in self.parameters]
            self._v = [np.zeros_like(p) for p, _ in self.parameters]
            self._t = 0
        def step(self) -> None:
            self._t += 1
            bc1 = 1.0 - self.beta1 ** self._t
            bc2 = 1.0 - self.beta2 ** self._t
            for i, (param, grad) in enumerate(self.parameters):
                if grad is None:
                    continue
                if self.weight_decay > 0.0:
                    param *= 1.0 - self.lr * self.weight_decay
                m = self._m[i]
                v = self._v[i]
                m *= self.beta1
                m += (1.0 - self.beta1) * grad
                v *= self.beta2
                v += (1.0 - self.beta2) * (grad * grad)
                m_hat = m / bc1
                v_hat = v / bc2
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        def zero_grad(self, parameters: List[Tuple[np.ndarray, np.ndarray]]) -> None:
            pass

# One-time guards and Markdown help (-help java-style)
install_dataset_write_guard()
if any(arg == "-help" for arg in sys.argv[1:]):
    md_path = Path(__file__).resolve().parents[2] / "Image_Identify_CNN.md"
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception as e:
        text = f"Help file not found or unreadable: {e}"
    try:
        print(text)
    except UnicodeEncodeError:
        enc = (sys.stdout.encoding or "utf-8")
        sys.stdout.write(text.encode(enc, errors="replace").decode(enc, errors="replace"))
    raise SystemExit(0)


def one_hot(labels: np.ndarray, num_classes: int, label_smoothing: float = 0.0) -> np.ndarray:
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError("label_smoothing must satisfy 0 <= value < 1")
    labels = labels.ravel()
    N = labels.size
    off_value = label_smoothing / num_classes
    on_value = 1.0 - label_smoothing + off_value
    out = np.full((N, num_classes), off_value, dtype=np.float64)
    out[np.arange(N), labels] = on_value
    return out


def clip_gradients(parameters: List[Tuple[np.ndarray, np.ndarray]], max_norm: float) -> float:
    if max_norm <= 0.0:
        return 0.0
    total_sq = 0.0
    for _, g in parameters:
        if g is not None:
            total_sq += float(np.sum(g * g))
    total = float(np.sqrt(total_sq))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for _, g in parameters:
            if g is not None:
                g *= scale
    return total


def list_labeled_paths(
    data_dir: Path,
    fallback_num_classes: int,
    allow_unlabeled_root: bool = False,
) -> tuple[list[Path], np.ndarray, list[str], bool]:
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if class_dirs:
        paths: list[Path] = []
        labels: list[int] = []
        class_names: list[str] = []
        class_id = 0
        for d in class_dirs:
            files = [f for f in sorted(d.iterdir()) if f.suffix.lower() in exts]
            if not files:
                continue
            paths.extend(files)
            labels.extend([class_id] * len(files))
            class_names.append(d.name)
            class_id += 1
        if paths:
            return paths, np.asarray(labels, dtype=np.int64), class_names, False
    if not allow_unlabeled_root:
        raise SystemExit(
            "No class subdirectories found. Expected `data_dir/class_name/*.jpg` layout. "
            "Use --allow-unlabeled-root to synthesize labels if needed."
        )
    files = [f for f in sorted(data_dir.iterdir()) if f.suffix.lower() in exts]
    if not files:
        raise SystemExit(f"No images found in {data_dir}")
    paths = files
    labels = np.arange(len(paths), dtype=np.int64) % fallback_num_classes
    class_names = [str(i) for i in range(fallback_num_classes)]
    return paths, labels, class_names, True


def class_distribution(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.bincount(labels, minlength=num_classes).astype(np.int64)


def make_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = class_distribution(labels, num_classes).astype(np.float64)
    weights = np.zeros((num_classes,), dtype=np.float64)
    nz = counts > 0
    if not np.any(nz):
        return np.ones((num_classes,), dtype=np.float64)
    weights[nz] = labels.size / (np.sum(nz) * counts[nz])
    weights[nz] /= np.mean(weights[nz])
    return weights


def build_balanced_epoch_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    counts = np.bincount(y).astype(np.float64)
    class_prob = np.zeros_like(counts)
    nz = counts > 0
    class_prob[nz] = 1.0 / counts[nz]
    sample_prob = class_prob[y]
    sample_prob /= np.sum(sample_prob)
    return rng.choice(y.size, size=y.size, replace=True, p=sample_prob)


def evaluate_streaming(model: CNN, paths: list[Path], labels: np.ndarray, num_classes: int, batch_size: int, input_size: tuple[int, int]) -> tuple[float, float]:
    n = len(paths)
    if n == 0:
        return float("nan"), float("nan")
    total_loss = 0.0
    total_correct = 0
    n_batches = 0
    model.eval()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_paths = paths[start:end]
        y_idx = labels[start:end]
        x = np.empty((len(batch_paths), input_size[0], input_size[1], 3), dtype=np.float64)
        for j, pth in enumerate(batch_paths):
            img = load_image(pth)
            x[j] = preprocess_image(img, target_size=input_size, normalize_to=config.NORMALIZE_TO, input_value_range=config.INPUT_VALUE_RANGE)
        y = one_hot(y_idx, num_classes, label_smoothing=0.0)
        logits = model.forward(x)
        total_loss += cross_entropy_loss(logits, y)
        preds = np.argmax(logits, axis=1)
        total_correct += int(np.sum(preds == y_idx))
        n_batches += 1
    return total_loss / max(1, n_batches), total_correct / max(1, n)


def evaluate_preloaded(model: CNN, X: np.ndarray, labels: np.ndarray, num_classes: int, batch_size: int) -> tuple[float, float]:
    n = labels.size
    total_loss = 0.0
    total_correct = 0
    n_batches = 0
    model.eval()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = X[start:end]
        y_idx = labels[start:end]
        y = one_hot(y_idx, num_classes, label_smoothing=0.0)
        logits = model.forward(x)
        total_loss += cross_entropy_loss(logits, y)
        preds = np.argmax(logits, axis=1)
        total_correct += int(np.sum(preds == y_idx))
        n_batches += 1
    return total_loss / max(1, n_batches), total_correct / max(1, n)


def stable_partition_index(p: Path, num_parts: int) -> int:
    if num_parts <= 1:
        return 0
    h = hashlib.md5(str(p.resolve()).lower().encode("utf-8")).hexdigest()
    return int(h[:8], 16) % num_parts


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


def load_weights_forgiving(model: CNN, npz_path: str | Path, skip_prefixes: tuple[str, ...] = ("fc2.",)) -> tuple[list[str], list[str]]:
    npz_path = str(npz_path)
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        raise SystemExit(f"Failed to open checkpoint {npz_path}: {e}")
    try:
        state = model.state_dict()
        loaded: list[str] = []
        skipped: list[str] = []
        files = set(data.files)
        for k, tgt in state.items():
            if any(k.startswith(p) for p in skip_prefixes):
                skipped.append(k)
                continue
            if k not in files:
                skipped.append(k)
                continue
            src = np.asarray(data[k], dtype=np.float64)
            if src.shape != tgt.shape:
                skipped.append(k)
                continue
            tgt[...] = src
            loaded.append(k)
        return loaded, skipped
    finally:
        try:
            data.close()
        except Exception:
            pass


def _build_optimizer(args: argparse.Namespace, parameters: List[Tuple[np.ndarray, np.ndarray]], lr_value: float):
    """Recreate the NumPy optimizer when the active parameter set changes."""
    if args.optimizer == "adamw":
        return AdamWImpl(parameters=parameters, lr=lr_value, weight_decay=args.weight_decay)
    return SGDImpl(parameters=parameters, lr=lr_value, momentum=args.momentum, weight_decay=args.weight_decay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NumPy-only CNN image classifier (streaming or preload)")
    parser.add_argument("--help-md", action="store_true", help="Print Markdown guide and exit")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory with train images")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-3], help="One learning rate per phase")
    parser.add_argument("--phase-count", type=int, default=1, help="Number of contiguous training phases")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Warmup epochs at the start of each phase")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw", help="Optimizer type")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (for SGD)")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability in classifier head")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--min-lr-ratio", type=float, default=0.2, help="Minimum LR ratio for cosine schedule")
    parser.add_argument("--lr-schedule", choices=["cosine", "step", "constant"], default="cosine", help="LR schedule type")
    parser.add_argument("--step-size", type=int, default=10, help="StepLR epoch interval inside each phase")
    parser.add_argument("--gamma", type=float, default=0.5, help="StepLR decay factor")
    parser.add_argument("--early-stop-metric", choices=["val_loss", "val_acc"], default="val_loss", help="Metric for early stopping")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Global gradient clip norm (0 disables)")
    parser.add_argument("--class-weighting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balance-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience in epochs")
    parser.add_argument("--min-delta", type=float, default=1e-3, help="Minimum improvement for monitored metrics")
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_model.npz"))
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-unlabeled-root", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True,
                        help="If true, stream batches from disk; if false, preload all images into memory.")
    parser.add_argument("--init-from", type=str, default=None, help="Path to checkpoint .npz to initialize weights")
    parser.add_argument("--num-partitions", type=int, default=1, help="Split training set into P disjoint shards")
    parser.add_argument("--partition", type=int, default=0, help="Train only on shard id [0..P-1]")
    parser.add_argument("--auto-next-partition", action=argparse.BooleanOptionalAction, default=False,
                        help="Rotate partition id across runs and save state to --partition-state")
    parser.add_argument("--partition-state", type=str, default=str(config.CHECKPOINT_DIR / "partition_state.json"),
                        help="State file for --auto-next-partition")
    parser.add_argument("--class-count", type=int, default=None, help="Total number of classes. Defaults to the detected dataset class count.")
    parser.add_argument("--enforce-readonly-dataset", action=argparse.BooleanOptionalAction, default=True,
                        help="Verify Dataset unchanged before/after run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--freeze-bn-affine",
        type=parse_bool_flag,
        nargs="?",
        const=True,
        default=True,
        help="When false, keep BN affine parameters trainable during backbone freeze",
    )
    args = parser.parse_args()

    if args.help_md:
        md_path = Path(__file__).resolve().parents[2] / "Image_Identify_CNN.md"
        try:
            text = md_path.read_text(encoding="utf-8")
        except Exception as e:
            text = f"Help file not found or unreadable: {e}"
        try:
            print(text)
        except UnicodeEncodeError:
            enc = (sys.stdout.encoding or "utf-8")
            sys.stdout.write(text.encode(enc, errors="replace").decode(enc, errors="replace"))
        return

    if args.phase_count > args.epochs:
        raise SystemExit("--phase-count cannot be greater than --epochs")
    try:
        lr_values = validate_phase_learning_rates(args.lr, args.phase_count)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    data_dir = Path(args.data_dir) if args.data_dir else Path(config.DATA_DIR)
    data_dir = data_dir.resolve()

    pre_sig = tree_signature(data_dir) if args.enforce_readonly_dataset else None
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
    print(f"Using num_classes={num_classes} (detected={detected_classes})")
    print(f"Using classes={resolved_class_names}")

    n_samples = len(paths)
    idx = rng.permutation(n_samples)
    n_val = int(max(0, min(0.9, args.val_split)) * n_samples)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train_paths_all = [paths[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    y_train_all = labels[train_idx]
    y_val = labels[val_idx] if n_val > 0 else None

    P = max(1, int(args.num_partitions))
    part = choose_partition(args)
    if part < 0 or part >= P:
        raise SystemExit(f"--partition must be in [0, {P-1}]")
    if P > 1:
        keep = [stable_partition_index(p, P) == part for p in train_paths_all]
        train_paths = [p for p, k in zip(train_paths_all, keep) if k]
        y_train = y_train_all[np.array(keep, dtype=bool)]
        if len(train_paths) == 0:
            raise SystemExit("Selected partition contains no samples; adjust --num-partitions/--partition.")
        print(f"Partitioning: {P} shards, using shard {part} -> {len(train_paths)} samples")
    else:
        train_paths = train_paths_all
        y_train = y_train_all

    train_counts = class_distribution(y_train, num_classes)
    print("Train samples:", y_train.size, " class_counts=", train_counts.tolist())
    if y_val is not None:
        val_counts = class_distribution(y_val, num_classes)
        print("Val samples:", y_val.size, " class_counts=", val_counts.tolist())

    class_weights = make_class_weights(y_train, num_classes) if args.class_weighting else None
    input_size = config.INPUT_SIZE

    model = CNN(input_size=input_size, num_classes=num_classes, seed=args.seed, dropout_p=args.dropout)
    model.set_backbone_frozen(False, freeze_bn_affine=args.freeze_bn_affine)
    if args.init_from:
        try:
            model.load_weights(args.init_from)
            print(f"Initialized weights from: {args.init_from}")
        except (ValueError, KeyError) as e:
            print(f"[warn] Strict load failed: {e}")
            loaded, skipped = load_weights_forgiving(model, args.init_from, skip_prefixes=("fc2.",))
            print(f"Loaded {len(loaded)} keys from checkpoint; skipped {len(skipped)} (e.g., classifier head)")

    batch_size = int(max(1, args.batch_size))
    X_train = None
    X_val = None
    if not args.streaming:
        def load_and_preprocess(paths_list: list[Path]) -> np.ndarray:
            Xp = np.empty((len(paths_list), input_size[0], input_size[1], 3), dtype=np.float64)
            for i, pth in enumerate(paths_list):
                img = load_image(pth)
                Xp[i] = preprocess_image(img, target_size=input_size, normalize_to=config.NORMALIZE_TO, input_value_range=config.INPUT_VALUE_RANGE)
            return Xp
        print("Preloading images into memory (float64)…")
        X_train = load_and_preprocess(train_paths)
        if y_val is not None and len(val_paths) > 0:
            X_val = load_and_preprocess(val_paths)

    n_batches = (y_train.size + batch_size - 1) // batch_size
    phase_map = build_epoch_phase_map(args.epochs, args.phase_count)
    current_phase_index = -1
    phase_best_val_acc = -float("inf")
    phase_plateau_epochs = 0
    freeze_patience = 5
    backbone_frozen = False

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_saved = False
    bad_epochs = 0
    current_lr = lr_values[0]
    optimizer = _build_optimizer(args, model.get_trainable_parameters(), lr_value=current_lr)

    for epoch in range(args.epochs):
        phase_config = phase_map[epoch]
        if phase_config.phase_index != current_phase_index:
            current_phase_index = phase_config.phase_index
            phase_best_val_acc = -float("inf")
            phase_plateau_epochs = 0
            backbone_frozen = False
            model.train()
            model.set_backbone_frozen(False, freeze_bn_affine=args.freeze_bn_affine)
            optimizer = _build_optimizer(args, model.get_trainable_parameters(), lr_value=lr_values[current_phase_index])
            print(
                f"Phase {current_phase_index + 1}/{args.phase_count}  "
                f"base_lr={lr_values[current_phase_index]:.6f}  "
                f"epochs={phase_config.epochs_in_phase}  "
                "cosine_restarts_per_phase=yes"
            )

        model.train()
        model.set_backbone_frozen(backbone_frozen, freeze_bn_affine=args.freeze_bn_affine)
        perm = build_balanced_epoch_indices(y_train, rng) if args.balance_sampling else rng.permutation(y_train.size)
        epoch_loss = 0.0
        epoch_correct = 0

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, y_train.size)
            idx_b = perm[start:end]
            y_batch_idx = y_train[idx_b]

            if args.streaming:
                batch_paths = [train_paths[i] for i in idx_b]
                x_batch = np.empty((len(batch_paths), input_size[0], input_size[1], 3), dtype=np.float64)
                for j, pth in enumerate(batch_paths):
                    img = load_image(pth)
                    x_batch[j] = preprocess_image(img, target_size=input_size, normalize_to=config.NORMALIZE_TO, input_value_range=config.INPUT_VALUE_RANGE)
            else:
                x_batch = X_train[idx_b]

            if args.augment:
                x_batch = augment_batch(x_batch, rng).astype(np.float64, copy=False)
            y_batch = one_hot(y_batch_idx, num_classes, label_smoothing=args.label_smoothing)

            current_lr = compute_phase_learning_rate(
                base_lr=lr_values[current_phase_index],
                schedule=args.lr_schedule,
                min_lr_ratio=args.min_lr_ratio,
                gamma=args.gamma,
                step_size=args.step_size,
                warmup_epochs=args.warmup_epochs,
                epoch_index_in_phase=phase_config.epoch_index_in_phase,
                epochs_in_phase=phase_config.epochs_in_phase,
                batch_index=b,
                num_batches=n_batches,
            )
            optimizer.lr = current_lr

            logits = model.forward(x_batch)
            loss = cross_entropy_loss(logits, y_batch, class_weights=class_weights)
            epoch_loss += loss
            preds = np.argmax(logits, axis=1)
            epoch_correct += int(np.sum(preds == y_batch_idx))

            dlogits = cross_entropy_loss_backward(logits, y_batch, class_weights=class_weights)
            model.backward(dlogits)
            params = model.get_trainable_parameters()
            if args.grad_clip > 0.0:
                clip_gradients(params, args.grad_clip)
            optimizer.parameters = params
            optimizer.step()

        avg_loss = epoch_loss / max(1, n_batches)
        train_acc = epoch_correct / max(1, y_train.size)

        if y_val is not None and len(val_paths) > 0:
            model.eval()
            if args.streaming or X_val is None:
                val_loss, val_acc = evaluate_streaming(model, val_paths, y_val, num_classes=num_classes, batch_size=batch_size, input_size=input_size)
            else:
                val_loss, val_acc = evaluate_preloaded(model, X_val, y_val, num_classes=num_classes, batch_size=batch_size)

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

            if val_acc > (phase_best_val_acc + args.min_delta):
                phase_best_val_acc = val_acc
                phase_plateau_epochs = 0
            else:
                phase_plateau_epochs += 1
                if (not backbone_frozen) and phase_plateau_epochs >= freeze_patience:
                    backbone_frozen = True
                    model.train()
                    model.set_backbone_frozen(True, freeze_bn_affine=args.freeze_bn_affine)
                    optimizer = _build_optimizer(args, model.get_trainable_parameters(), lr_value=current_lr)
                    print(
                        f"Backbone frozen at epoch {epoch + 1}: val_acc plateaued for {freeze_patience} epochs; "
                        "training the head until the next phase."
                    )

            star = " *" if improved else ""
            gap = train_acc - val_acc
            mode_text = "head-only" if backbone_frozen else "full"
            print(
                f"Epoch {epoch + 1}/{args.epochs}  phase={current_phase_index + 1}/{args.phase_count}  "
                f"mode={mode_text}  lr={optimizer.lr:.6f}  train_loss={avg_loss:.4f}  train_acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  gap={gap:.3f}{star}"
            )
            if args.early_stop and bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}: metric {args.early_stop_metric} did not improve by {args.min_delta} for {args.patience} epochs.")
                break
        else:
            mode_text = "head-only" if backbone_frozen else "full"
            print(
                f"Epoch {epoch + 1}/{args.epochs}  phase={current_phase_index + 1}/{args.phase_count}  "
                f"mode={mode_text}  lr={optimizer.lr:.6f}  train_loss={avg_loss:.4f}  train_acc={train_acc:.3f}"
            )

    if y_val is None or len(val_paths) == 0:
        model.save_weights(args.checkpoint)
        best_saved = True
    if best_saved:
        print(f"Saved checkpoint: {args.checkpoint}")

    if args.enforce_readonly_dataset and pre_sig is not None:
        post_sig = tree_signature(data_dir)
        if post_sig != pre_sig:
            raise SystemExit("Dataset changed during run; aborting (read-only enforcement).")

    print("Training finished.")

if __name__ == "__main__":
    main()
