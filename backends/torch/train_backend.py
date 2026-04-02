"""PyTorch training backend.

This backend keeps the project data pipeline and CLI style close to the legacy
NumPy trainer, but runs the model, gradients, optimizer, and checkpointing in
PyTorch. The original NumPy trainer remains available under `--backend numpy`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import config
from backends.torch.model import TorchCNN, load_checkpoint_state
from data.loaders import load_image
from data.preprocessing import preprocess_image
from utils.safety import install_dataset_write_guard, tree_signature
from utils.training import (
    ModelEMA,
    adjust_phase_lr_offset_after_unfreeze,
    apply_batch_mix,
    augment_batch,
    build_epoch_phase_map,
    compute_effective_learning_rate,
    compute_phase_learning_rate,
    parse_bool_flag,
    validate_augmentation_args,
    validate_cutmix_ratio,
    validate_ema_decay,
    validate_focal_gamma,
    validate_freeze_cycle_args,
    validate_mixup_alpha,
    validate_mixup_probability,
    validate_model_width_scale,
    validate_phase_learning_rates,
)

install_dataset_write_guard()

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception as exc:
    raise SystemExit(f"PyTorch backend is unavailable in this interpreter: {exc}") from exc


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


def stable_partition_index(path: Path, num_parts: int) -> int:
    """Hash a path into a stable partition id so repeated runs shard consistently."""
    if num_parts <= 1:
        return 0
    digest = hashlib.md5(str(path.resolve()).lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_parts


def choose_partition(args) -> int:
    """Resolve the active data partition, optionally rotating across runs."""
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
    """Resolve `cpu`, `cuda`, or `auto` into a concrete torch device."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device=cuda requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr_value: float) -> None:
    """Update every optimizer parameter group to the current effective LR."""
    for group in optimizer.param_groups:
        group["lr"] = lr_value


def _to_tensor_batch(x_batch: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a contiguous NHWC NumPy batch into a float32 torch tensor."""
    return torch.from_numpy(np.ascontiguousarray(x_batch)).to(device=device, dtype=torch.float32)


def _load_batch(paths: list[Path], input_size: tuple[int, int]) -> np.ndarray:
    """Load and preprocess one batch of images from disk."""
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


def _make_target_distribution(
    labels: np.ndarray,
    num_classes: int,
    label_smoothing: float,
) -> np.ndarray:
    """Build one-hot or label-smoothed targets before tensor conversion."""
    label_smoothing = float(label_smoothing)
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError("label_smoothing must satisfy 0 <= value < 1")
    labels = np.asarray(labels, dtype=np.int64).ravel()
    targets = np.full((labels.size, num_classes), label_smoothing / num_classes, dtype=np.float32)
    targets[np.arange(labels.size), labels] = 1.0 - label_smoothing + (label_smoothing / num_classes)
    return targets


def _soft_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute weighted soft-label cross entropy for hard, smoothed, or mixed labels."""
    log_probs = F.log_softmax(logits, dim=1)
    per_sample_nll = -(targets * log_probs).sum(dim=1)
    if class_weights is None:
        return per_sample_nll.mean()
    sample_weights = (targets * class_weights.view(1, -1)).sum(dim=1)
    return (sample_weights * per_sample_nll).sum() / sample_weights.sum().clamp_min(1e-12)


def _focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    alpha_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute generalized focal loss for standard or label-smoothed targets."""
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)
    pt = (targets * probs).sum(dim=1)
    ce = -(targets * log_probs).sum(dim=1)
    focal_factor = torch.pow((1.0 - pt).clamp(min=0.0, max=1.0), float(gamma))
    if alpha_weights is None:
        sample_alpha = torch.ones_like(pt)
    else:
        sample_alpha = (targets * alpha_weights.view(1, -1)).sum(dim=1)
    return (sample_alpha * focal_factor * ce).sum() / sample_alpha.sum().clamp_min(1e-12)


def _compute_batch_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    use_focal_loss: bool,
    focal_gamma: float,
    ce_class_weights: torch.Tensor | None,
    focal_alpha_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Select the correct loss path for the current run-level policy."""
    if use_focal_loss:
        return _focal_loss(logits, targets, gamma=focal_gamma, alpha_weights=focal_alpha_weights)
    return _soft_cross_entropy_loss(logits, targets, class_weights=ce_class_weights)


def evaluate_streaming(
    model: TorchCNN,
    paths: list[Path],
    labels: np.ndarray,
    num_classes: int,
    batch_size: int,
    input_size: tuple[int, int],
    device: torch.device,
    *,
    use_focal_loss: bool,
    focal_gamma: float,
    ce_class_weights: torch.Tensor | None,
    focal_alpha_weights: torch.Tensor | None,
) -> tuple[float, float]:
    """Evaluate a model while streaming validation images from disk."""
    if len(paths) == 0:
        return float("nan"), float("nan")
    total_loss = 0.0
    total_correct = 0
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            end = min(start + batch_size, len(paths))
            x_batch = _load_batch(paths[start:end], input_size)
            y_batch = labels[start:end]
            targets = _make_target_distribution(y_batch, num_classes, label_smoothing=0.0)
            y_tensor = torch.as_tensor(targets, device=device, dtype=torch.float32)
            logits = model(_to_tensor_batch(x_batch, device))
            loss = _compute_batch_loss(
                logits,
                y_tensor,
                use_focal_loss=use_focal_loss,
                focal_gamma=focal_gamma,
                ce_class_weights=ce_class_weights,
                focal_alpha_weights=focal_alpha_weights,
            )
            total_loss += float(loss.item())
            total_correct += int((torch.argmax(logits, dim=1) == torch.as_tensor(y_batch, device=device, dtype=torch.long)).sum().item())
            num_batches += 1
    return total_loss / max(1, num_batches), total_correct / max(1, len(paths))


def evaluate_preloaded(
    model: TorchCNN,
    x_data: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    batch_size: int,
    device: torch.device,
    *,
    use_focal_loss: bool,
    focal_gamma: float,
    ce_class_weights: torch.Tensor | None,
    focal_alpha_weights: torch.Tensor | None,
) -> tuple[float, float]:
    """Evaluate a model against a preloaded validation tensor."""
    total_loss = 0.0
    total_correct = 0
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            x_batch = _to_tensor_batch(x_data[start:end], device)
            y_batch = labels[start:end]
            targets = _make_target_distribution(y_batch, num_classes, label_smoothing=0.0)
            y_tensor = torch.as_tensor(targets, device=device, dtype=torch.float32)
            logits = model(x_batch)
            loss = _compute_batch_loss(
                logits,
                y_tensor,
                use_focal_loss=use_focal_loss,
                focal_gamma=focal_gamma,
                ce_class_weights=ce_class_weights,
                focal_alpha_weights=focal_alpha_weights,
            )
            total_loss += float(loss.item())
            total_correct += int((torch.argmax(logits, dim=1) == torch.as_tensor(y_batch, device=device, dtype=torch.long)).sum().item())
            num_batches += 1
    return total_loss / max(1, num_batches), total_correct / max(1, len(labels))


def load_weights_forgiving(model: TorchCNN, checkpoint_path: str | Path, skip_prefixes: tuple[str, ...] = ("fc2.",)) -> tuple[list[str], list[str]]:
    checkpoint = Path(checkpoint_path)
    state, _ = load_checkpoint_state(checkpoint, map_location="cpu")
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


def _apply_backbone_freeze_state(model: TorchCNN, backbone_frozen: bool, freeze_bn_affine: bool) -> None:
    """Apply the temporary freeze policy after each `model.train()` call.

    `model.train()` recursively flips every module back to training mode, so the
    trainer reapplies the backbone freeze state each epoch to keep BN running
    statistics fixed when the backbone is frozen.
    """
    for parameter in model.iter_head_parameters():
        parameter.requires_grad = True

    for parameter in model.iter_backbone_parameters():
        parameter.requires_grad = not backbone_frozen

    for bn_layer in model.backbone_batchnorm_layers():
        if backbone_frozen:
            bn_layer.eval()
            if bn_layer.weight is not None:
                bn_layer.weight.requires_grad = not freeze_bn_affine
            if bn_layer.bias is not None:
                bn_layer.bias.requires_grad = not freeze_bn_affine
        else:
            bn_layer.train()
            if bn_layer.weight is not None:
                bn_layer.weight.requires_grad = True
            if bn_layer.bias is not None:
                bn_layer.bias.requires_grad = True


def _build_optimizer(
    args: argparse.Namespace,
    model: TorchCNN,
    lr_value: float,
) -> torch.optim.Optimizer:
    """Recreate the optimizer intentionally when freeze state changes.

    Rebuilding the optimizer makes the transition explicit and avoids carrying
    stale momentum state across a different parameter set.
    """
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=lr_value, weight_decay=args.weight_decay)
    return torch.optim.SGD(parameters, lr=lr_value, momentum=args.momentum, weight_decay=args.weight_decay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train image classifier with the PyTorch backend")
    parser.add_argument("--help-md", action="store_true", help="Print Markdown guide and exit")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory with train images")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-3], help="One learning rate per phase")
    parser.add_argument("--phase-count", type=int, default=1, help="Number of contiguous training phases")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Warmup epochs at the start of each phase")
    parser.add_argument("--freeze-patience", type=int, default=8, help="Epochs without val_acc improvement before freezing the backbone")
    parser.add_argument("--freeze-epoch-num", type=int, default=10, help="How many epochs each temporary freeze window lasts")
    parser.add_argument("--after-unfreeze-lr-change", type=float, default=1e-4, help="Additive LR decrement applied after unfreeze when allowed")
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
    parser.add_argument("--focal-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--focal-gamma", type=float, default=1.5, help="Focal-loss gamma parameter")
    parser.add_argument("--focal-alpha", choices=["auto", "none"], default="auto", help="Focal alpha weighting policy")
    parser.add_argument("--mixup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixup-alpha", type=float, default=0.2, help="Shared Beta(alpha, alpha) parameter for MixUp and CutMix")
    parser.add_argument("--mixup-prob", type=float, default=0.4, help="Per-batch probability of enabling MixUp/CutMix routing")
    parser.add_argument("--cutmix-ratio", type=float, default=0.5, help="Conditional probability of using CutMix once batch mixing is enabled")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999, help="Base EMA decay applied after each optimizer step")
    parser.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience in epochs")
    parser.add_argument("--min-delta", type=float, default=1e-3, help="Minimum improvement for monitored metrics")
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_torch_model.pt"))
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rotation", type=float, default=12.0, help="Maximum absolute random rotation in degrees")
    parser.add_argument("--brightness", type=float, default=0.2, help="Brightness jitter strength in [0, 1]")
    parser.add_argument("--contrast", type=float, default=0.2, help="Contrast jitter strength in [0, 1]")
    parser.add_argument("--saturation", type=float, default=0.2, help="Saturation jitter strength in [0, 1]")
    parser.add_argument("--model-width-scale", type=float, default=0.75, help="Width multiplier for the stage-2 convolution block")
    parser.add_argument("--allow-unlabeled-root", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="If true, stream batches from disk; if false, preload all images into memory.")
    parser.add_argument("--init-from", type=str, default=None, help="Path to checkpoint .pt/.pth to initialize weights")
    parser.add_argument("--num-partitions", type=int, default=1, help="Split training set into P disjoint shards")
    parser.add_argument("--partition", type=int, default=0, help="Train only on shard id [0..P-1]")
    parser.add_argument("--auto-next-partition", action=argparse.BooleanOptionalAction, default=False, help="Rotate partition id across runs and save state to --partition-state")
    parser.add_argument("--partition-state", type=str, default=str(config.CHECKPOINT_DIR / "partition_state.json"), help="State file for --auto-next-partition")
    parser.add_argument("--class-count", type=int, default=None, help="Total number of classes. Defaults to the detected dataset class count.")
    parser.add_argument("--enforce-readonly-dataset", action=argparse.BooleanOptionalAction, default=True, help="Verify Dataset unchanged before/after run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Training device")
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
            print(md_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"Help file not found or unreadable: {exc}") from exc
        return

    if args.phase_count > args.epochs:
        raise SystemExit("--phase-count cannot be greater than --epochs")
    if not (0.0 <= float(args.label_smoothing) < 1.0):
        raise SystemExit("--label-smoothing must satisfy 0 <= value < 1")
    try:
        lr_values = validate_phase_learning_rates(args.lr, args.phase_count)
        freeze_patience, freeze_epoch_num, after_unfreeze_lr_change = validate_freeze_cycle_args(
            args.freeze_patience,
            args.freeze_epoch_num,
            args.after_unfreeze_lr_change,
        )
        augment_config = validate_augmentation_args(
            args.rotation,
            args.brightness,
            args.contrast,
            args.saturation,
        )
        mixup_alpha = validate_mixup_alpha(args.mixup_alpha)
        mixup_prob = validate_mixup_probability(args.mixup_prob)
        cutmix_ratio = validate_cutmix_ratio(args.cutmix_ratio)
        ema_decay = validate_ema_decay(args.ema_decay)
        focal_gamma = validate_focal_gamma(args.focal_gamma)
        width_scale = validate_model_width_scale(args.model_width_scale)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

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

    inverse_frequency_weights = None
    if args.class_weighting or args.focal_alpha == "auto":
        inverse_frequency_weights = make_class_weights(y_train, num_classes)
    ce_class_weights = None
    if args.class_weighting and inverse_frequency_weights is not None:
        ce_class_weights = torch.as_tensor(inverse_frequency_weights, dtype=torch.float32, device=device)
    focal_alpha_weights = None
    if args.focal_alpha == "auto" and inverse_frequency_weights is not None:
        focal_alpha_weights = torch.as_tensor(inverse_frequency_weights, dtype=torch.float32, device=device)

    base_use_focal_loss = bool(args.focal_loss)

    input_size = config.INPUT_SIZE

    model = TorchCNN(
        input_size=input_size,
        num_classes=num_classes,
        seed=args.seed,
        dropout_p=args.dropout,
        width_scale=width_scale,
    )
    model.to(device)
    print(f"Using stage2_channels={model.stage2_channels} (width_scale={model.width_scale:.3f})")
    if args.init_from:
        try:
            model.load_weights(args.init_from, map_location=device)
            print(f"Initialized weights from: {args.init_from}")
        except Exception as exc:
            print(f"[warn] Strict load failed: {exc}")
            loaded, skipped = load_weights_forgiving(model, args.init_from, skip_prefixes=("fc2.",))
            print(f"Loaded {len(loaded)} keys from checkpoint; skipped {len(skipped)} (for example, classifier or resized feature tensors)")
            model.to(device)

    batch_size = max(1, int(args.batch_size))
    x_train = None
    x_val = None
    if not args.streaming:
        print("Preloading images into memory (float32)...")
        x_train = _load_batch(train_paths, input_size)
        if y_val is not None and len(val_paths) > 0:
            x_val = _load_batch(val_paths, input_size)

    num_batches = (y_train.size + batch_size - 1) // batch_size
    phase_map = build_epoch_phase_map(args.epochs, args.phase_count)
    current_phase_index = -1
    current_phase_base_lr = lr_values[0]
    phase_lr_offset = 0.0
    phase_best_val_acc = -float("inf")
    phase_plateau_epochs = 0
    freeze_epochs_remaining = 0
    backbone_frozen = False

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_saved = False
    bad_epochs = 0
    current_lr = current_phase_base_lr

    _apply_backbone_freeze_state(model, backbone_frozen=False, freeze_bn_affine=args.freeze_bn_affine)
    optimizer: torch.optim.Optimizer = _build_optimizer(args, model, lr_value=current_phase_base_lr)

    ema: ModelEMA | None = None
    if args.ema:
        ema_model = TorchCNN(
            input_size=input_size,
            num_classes=num_classes,
            seed=args.seed,
            dropout_p=args.dropout,
            width_scale=width_scale,
        )
        ema_model.to(device)
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        ema_model.eval()
        ema = ModelEMA(ema_model, decay=ema_decay, phase_warmup_steps=max(1, num_batches))
        ema.sync_from(model)

    for epoch in range(args.epochs):
        phase_config = phase_map[epoch]
        if phase_config.phase_index != current_phase_index:
            current_phase_index = phase_config.phase_index
            current_phase_base_lr = lr_values[current_phase_index]
            phase_lr_offset = 0.0
            phase_best_val_acc = -float("inf")
            phase_plateau_epochs = 0
            freeze_epochs_remaining = 0
            backbone_frozen = False
            model.train()
            _apply_backbone_freeze_state(model, backbone_frozen=False, freeze_bn_affine=args.freeze_bn_affine)
            optimizer = _build_optimizer(args, model, lr_value=current_phase_base_lr)
            print(
                f"Phase {current_phase_index + 1}/{args.phase_count}  "
                f"base_lr={current_phase_base_lr:.6f}  "
                f"epochs={phase_config.epochs_in_phase}  "
                "cosine_restarts_per_phase=yes"
            )

        model.train()
        epoch_backbone_frozen = backbone_frozen
        _apply_backbone_freeze_state(model, backbone_frozen=epoch_backbone_frozen, freeze_bn_affine=args.freeze_bn_affine)
        permutation = rng.permutation(y_train.size)
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
                x_batch = augment_batch(x_batch, rng, augment_config)

            mix_mode = "none"
            if args.mixup:
                target_distribution = _make_target_distribution(
                    y_batch_idx,
                    num_classes,
                    label_smoothing=0.0,
                )
                x_batch, target_distribution, mix_mode, _ = apply_batch_mix(
                    x_batch,
                    target_distribution,
                    rng,
                    prob=mixup_prob,
                    beta_alpha=mixup_alpha,
                    cutmix_ratio=cutmix_ratio,
                )
                if mix_mode == "none":
                    target_distribution = _make_target_distribution(
                        y_batch_idx,
                        num_classes,
                        label_smoothing=float(args.label_smoothing),
                    )
            else:
                target_distribution = _make_target_distribution(
                    y_batch_idx,
                    num_classes,
                    label_smoothing=float(args.label_smoothing),
                )

            scheduled_lr = compute_phase_learning_rate(
                base_lr=current_phase_base_lr,
                schedule=args.lr_schedule,
                min_lr_ratio=args.min_lr_ratio,
                gamma=args.gamma,
                step_size=args.step_size,
                warmup_epochs=args.warmup_epochs,
                epoch_index_in_phase=phase_config.epoch_index_in_phase,
                epochs_in_phase=phase_config.epochs_in_phase,
                batch_index=batch_index,
                num_batches=num_batches,
            )
            current_lr = compute_effective_learning_rate(
                scheduled_lr,
                phase_lr_offset,
                current_phase_base_lr,
                args.min_lr_ratio,
            )
            _set_optimizer_lr(optimizer, current_lr)

            optimizer.zero_grad(set_to_none=True)
            x_tensor = _to_tensor_batch(x_batch, device)
            y_tensor = torch.as_tensor(target_distribution, dtype=torch.float32, device=device)
            logits = model(x_tensor)
            batch_use_focal_loss = bool(base_use_focal_loss and mix_mode == "none")
            loss = _compute_batch_loss(
                logits,
                y_tensor,
                use_focal_loss=batch_use_focal_loss,
                focal_gamma=focal_gamma,
                ce_class_weights=ce_class_weights,
                focal_alpha_weights=focal_alpha_weights,
            )
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], max_norm=args.grad_clip)
            optimizer.step()
            if ema is not None:
                phase_step = phase_config.epoch_index_in_phase * num_batches + batch_index
                ema.update(model, step_in_phase=phase_step, mix_active=(mix_mode != "none"))

            epoch_loss += float(loss.item())
            metric_targets = torch.argmax(y_tensor, dim=1)
            epoch_correct += int((torch.argmax(logits, dim=1) == metric_targets).sum().item())

        avg_loss = epoch_loss / max(1, num_batches)
        train_acc = epoch_correct / max(1, y_train.size)

        if y_val is not None and len(val_paths) > 0:
            eval_model = ema.ema_model if ema is not None else model
            eval_model.eval()
            if args.streaming or x_val is None:
                val_loss, val_acc = evaluate_streaming(
                    eval_model,
                    val_paths,
                    y_val,
                    num_classes,
                    batch_size,
                    input_size,
                    device,
                    use_focal_loss=base_use_focal_loss,
                    focal_gamma=focal_gamma,
                    ce_class_weights=ce_class_weights,
                    focal_alpha_weights=focal_alpha_weights,
                )
            else:
                val_loss, val_acc = evaluate_preloaded(
                    eval_model,
                    x_val,
                    y_val,
                    num_classes,
                    batch_size,
                    device,
                    use_focal_loss=base_use_focal_loss,
                    focal_gamma=focal_gamma,
                    ce_class_weights=ce_class_weights,
                    focal_alpha_weights=focal_alpha_weights,
                )

            improved = (val_loss < (best_val_loss - args.min_delta)) if args.early_stop_metric == "val_loss" else (val_acc > (best_val_acc + args.min_delta))
            if improved:
                if args.early_stop_metric == "val_loss":
                    best_val_loss = val_loss
                else:
                    best_val_acc = val_acc
                checkpoint_model = ema.ema_model if ema is not None else model
                checkpoint_model.save_weights(
                    args.checkpoint,
                    metadata={
                        "is_ema": bool(ema is not None),
                        "ema_decay": float(ema_decay) if ema is not None else None,
                    },
                )
                best_saved = True
                bad_epochs = 0
            else:
                bad_epochs += 1

            if epoch_backbone_frozen:
                # Even during frozen epochs, keep tracking the best validation
                # accuracy reached in this phase so later plateau checks compare
                # against the strongest result already achieved.
                if val_acc > (phase_best_val_acc + args.min_delta):
                    phase_best_val_acc = val_acc
                freeze_epochs_remaining -= 1
                if freeze_epochs_remaining <= 0:
                    next_phase_start_lr = lr_values[current_phase_index + 1] if current_phase_index + 1 < len(lr_values) else None
                    phase_lr_offset, deduction_applied = adjust_phase_lr_offset_after_unfreeze(
                        current_effective_lr=current_lr,
                        phase_lr_offset=phase_lr_offset,
                        after_unfreeze_lr_change=after_unfreeze_lr_change,
                        phase_base_lr=current_phase_base_lr,
                        min_lr_ratio=args.min_lr_ratio,
                        next_phase_start_lr=next_phase_start_lr,
                    )
                    backbone_frozen = False
                    phase_plateau_epochs = 0
                    model.train()
                    _apply_backbone_freeze_state(model, backbone_frozen=False, freeze_bn_affine=args.freeze_bn_affine)
                    optimizer = _build_optimizer(args, model, lr_value=current_lr)
                    lr_message = "reduced" if deduction_applied else "kept"
                    print(
                        f"Backbone unfrozen at epoch {epoch + 1}: cumulative lr offset {lr_message} at {phase_lr_offset:.6f}."
                    )
            else:
                if val_acc > (phase_best_val_acc + args.min_delta):
                    phase_best_val_acc = val_acc
                    phase_plateau_epochs = 0
                else:
                    phase_plateau_epochs += 1
                    if phase_plateau_epochs >= freeze_patience:
                        backbone_frozen = True
                        freeze_epochs_remaining = freeze_epoch_num
                        phase_plateau_epochs = 0
                        model.train()
                        _apply_backbone_freeze_state(model, backbone_frozen=True, freeze_bn_affine=args.freeze_bn_affine)
                        optimizer = _build_optimizer(args, model, lr_value=current_lr)
                        print(
                            f"Backbone frozen at epoch {epoch + 1}: val_acc plateaued for {freeze_patience} epochs; "
                            f"training the head for {freeze_epoch_num} epochs."
                        )

            gap = train_acc - val_acc
            marker = " *" if improved else ""
            mode_text = "head-only" if epoch_backbone_frozen else "full"
            print(
                f"Epoch {epoch + 1}/{args.epochs}  phase={current_phase_index + 1}/{args.phase_count}  "
                f"mode={mode_text}  lr={current_lr:.6f}  train_loss={avg_loss:.4f}  train_acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  gap={gap:.3f}{marker}"
            )
            if args.early_stop and bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}: metric {args.early_stop_metric} did not improve by {args.min_delta} for {args.patience} epochs.")
                break
        else:
            if epoch_backbone_frozen:
                freeze_epochs_remaining -= 1
                if freeze_epochs_remaining <= 0:
                    next_phase_start_lr = lr_values[current_phase_index + 1] if current_phase_index + 1 < len(lr_values) else None
                    phase_lr_offset, _ = adjust_phase_lr_offset_after_unfreeze(
                        current_effective_lr=current_lr,
                        phase_lr_offset=phase_lr_offset,
                        after_unfreeze_lr_change=after_unfreeze_lr_change,
                        phase_base_lr=current_phase_base_lr,
                        min_lr_ratio=args.min_lr_ratio,
                        next_phase_start_lr=next_phase_start_lr,
                    )
                    backbone_frozen = False
                    phase_plateau_epochs = 0
                    model.train()
                    _apply_backbone_freeze_state(model, backbone_frozen=False, freeze_bn_affine=args.freeze_bn_affine)
                    optimizer = _build_optimizer(args, model, lr_value=current_lr)
            mode_text = "head-only" if epoch_backbone_frozen else "full"
            print(
                f"Epoch {epoch + 1}/{args.epochs}  phase={current_phase_index + 1}/{args.phase_count}  "
                f"mode={mode_text}  lr={current_lr:.6f}  train_loss={avg_loss:.4f}  train_acc={train_acc:.3f}"
            )

    if y_val is None or len(val_paths) == 0:
        checkpoint_model = ema.ema_model if ema is not None else model
        checkpoint_model.save_weights(
            args.checkpoint,
            metadata={
                "is_ema": bool(ema is not None),
                "ema_decay": float(ema_decay) if ema is not None else None,
            },
        )
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
