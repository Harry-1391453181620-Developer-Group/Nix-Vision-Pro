"""PyTorch training backend.

This backend keeps the project data pipeline and CLI style close to the legacy
NumPy trainer, but runs the model, gradients, optimizer, and checkpointing in
PyTorch. The original NumPy trainer remains available under `--backend numpy`.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

import config
from backends.torch.model import DEFAULT_OMEGA_FEATURE_DIM, TorchCNN, load_checkpoint_state
from data.loaders import load_image
from data.preprocessing import preprocess_image
from utils.safety import install_dataset_write_guard, tree_signature
from utils.training import (
    ModelEMA,
    adjust_phase_lr_offset_after_unfreeze,
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
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:
    raise SystemExit(f"PyTorch backend is unavailable in this interpreter: {exc}") from exc


class TorchImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Stateless dataset for worker-safe Windows DataLoader multiprocessing."""

    def __init__(
        self,
        *,
        labels: np.ndarray,
        input_size: tuple[int, int],
        paths: Sequence[Path] | None = None,
        preloaded_images: np.ndarray | None = None,
        augment_config=None,
    ) -> None:
        if paths is None and preloaded_images is None:
            raise ValueError("Either paths or preloaded_images must be provided")
        if paths is not None and preloaded_images is not None and len(paths) != int(preloaded_images.shape[0]):
            raise ValueError("paths and preloaded_images must have matching lengths")
        self.labels = np.asarray(labels, dtype=np.int64)
        self.input_size = tuple(input_size)
        self.paths = tuple(Path(path) for path in paths) if paths is not None else None
        self.preloaded_images = (
            np.asarray(preloaded_images, dtype=np.float32)
            if preloaded_images is not None
            else None
        )
        self.augment_config = augment_config

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def _load_image(self, index: int) -> np.ndarray:
        if self.preloaded_images is not None:
            return np.asarray(self.preloaded_images[index], dtype=np.float32)
        assert self.paths is not None
        image = load_image(self.paths[index])
        return preprocess_image(
            image,
            target_size=self.input_size,
            normalize_to=config.NORMALIZE_TO,
            input_value_range=config.INPUT_VALUE_RANGE,
        ).astype(np.float32)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self._load_image(index)
        if self.augment_config is not None:
            seed = int(np.random.randint(0, 2**31 - 1))
            rng = np.random.default_rng(seed)
            image = augment_batch(image[np.newaxis, ...], rng, self.augment_config)[0]
        image_nchw = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_nchw, dtype=np.float32))
        return image_tensor, int(self.labels[index])


OMEGA_COLLAPSE_VARIANCE_THRESHOLD = 1e-4
OMEGA_COLLAPSE_EPOCHS = 3


@dataclass(frozen=True)
class RepresentationVarianceStats:
    mean: float
    min: float
    max: float


def _validate_omega_args(
    *,
    omega_loss: bool,
    omega_lambda: float,
    omega_projector_depth: int,
    omega_hidden_dim: int,
) -> tuple[float, int, int]:
    omega_lambda = float(omega_lambda)
    omega_projector_depth = int(omega_projector_depth)
    omega_hidden_dim = int(omega_hidden_dim)
    if omega_lambda < 0.0:
        raise ValueError("omega_lambda must be >= 0")
    if omega_projector_depth not in {1, 2}:
        raise ValueError("omega_projector_depth must be 1 or 2")
    if omega_hidden_dim <= 0:
        raise ValueError("omega_hidden_dim must be > 0")
    if omega_lambda > 0.0 and not omega_loss:
        raise ValueError("omega_lambda > 0 requires --omega-loss")
    return omega_lambda, omega_projector_depth, omega_hidden_dim


def _format_run_value(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _make_json_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _make_json_serializable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_serializable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_make_json_serializable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_make_json_serializable(payload), sort_keys=True) + "\n")


def _prepare_experiment_artifacts(
    *,
    experiment_root: Path,
    checkpoint_path: Path,
    seed: int,
    omega_lambda: float,
) -> tuple[Path, Path, Path, Path]:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = experiment_root / (
        f"{timestamp}-{checkpoint_path.stem}-seed_{int(seed)}-lambda_{_format_run_value(omega_lambda)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "epoch_metrics.jsonl"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    notes_path = run_dir / "qualitative_notes.txt"
    notes_path.write_text(
        "Phase 1 qualitative convergence notes\n"
        "===================================\n\n"
        "- smoothness:\n"
        "- oscillation:\n"
        "- divergence:\n"
        "- collapse signs:\n",
        encoding="utf-8",
    )
    return run_dir, config_path, metrics_path, summary_path


def _extract_representation_variance_stats(h: torch.Tensor | None) -> RepresentationVarianceStats:
    if h is None:
        return RepresentationVarianceStats(mean=float("nan"), min=float("nan"), max=float("nan"))
    variance = torch.var(h.detach().float(), dim=0, unbiased=False)
    return RepresentationVarianceStats(
        mean=float(variance.mean().item()),
        min=float(variance.min().item()),
        max=float(variance.max().item()),
    )


def _resolve_forward_output(
    forward_output: Any,
    *,
    omega_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if not omega_enabled:
        if isinstance(forward_output, tuple):
            return forward_output[0], None, None
        return forward_output, None, None
    if not isinstance(forward_output, tuple) or len(forward_output) != 3:
        raise ValueError("Omega-enabled forward path must return (logits, h, T(h))")
    logits, h, t_h = forward_output
    return logits, h, t_h


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


def _resolve_num_workers(streaming: bool, num_workers_arg: int | None) -> int:
    if num_workers_arg is not None:
        if int(num_workers_arg) < 0:
            raise SystemExit("--num-workers must be >= 0")
        return int(num_workers_arg)
    return 4 if streaming else 0


def _seed_data_loader_worker(worker_id: int, *, base_seed: int) -> None:
    """Keep worker init pickle-safe so Windows spawn can launch DataLoader workers."""
    worker_seed = int(base_seed) + int(worker_id)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    return partial(_seed_data_loader_worker, base_seed=int(base_seed))


def _collate_image_batch(batch: Sequence[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = zip(*batch)
    image_tensor = torch.stack(images, dim=0).contiguous()
    label_tensor = torch.as_tensor(labels, dtype=torch.int64).contiguous()
    return image_tensor, label_tensor


def _build_data_loader(
    dataset: Dataset[tuple[torch.Tensor, int]],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    base_seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(int(base_seed))
    loader_kwargs: dict[str, object] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": _collate_image_batch,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = _make_worker_init_fn(base_seed)
    return DataLoader(dataset, **loader_kwargs)


def _load_all_images(paths: Sequence[Path], input_size: tuple[int, int]) -> np.ndarray:
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


def _move_input_batch(
    x_batch: torch.Tensor,
    device: torch.device,
    *,
    use_channels_last: bool,
) -> torch.Tensor:
    x_tensor = x_batch.to(
        device=device,
        dtype=torch.float32,
        non_blocking=(device.type == "cuda"),
    )
    if use_channels_last:
        x_tensor = x_tensor.contiguous(memory_format=torch.channels_last)
    return x_tensor


def _move_label_batch(y_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    return y_batch.to(device=device, dtype=torch.long, non_blocking=(device.type == "cuda"))


def _make_target_distribution(
    labels: torch.Tensor,
    num_classes: int,
    label_smoothing: float,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    label_smoothing = float(label_smoothing)
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError("label_smoothing must satisfy 0 <= value < 1")
    labels = labels.to(dtype=torch.long).view(-1)
    targets = torch.full(
        (int(labels.shape[0]), int(num_classes)),
        fill_value=label_smoothing / num_classes,
        device=labels.device,
        dtype=dtype,
    )
    targets.scatter_(
        1,
        labels.unsqueeze(1),
        1.0 - label_smoothing + (label_smoothing / num_classes),
    )
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


def _compute_total_loss_components(
    forward_output: Any,
    targets: torch.Tensor,
    *,
    omega_enabled: bool,
    omega_lambda: float,
    use_focal_loss: bool,
    focal_gamma: float,
    ce_class_weights: torch.Tensor | None,
    focal_alpha_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, RepresentationVarianceStats]:
    logits, h, t_h = _resolve_forward_output(forward_output, omega_enabled=omega_enabled)
    ce_loss = _compute_batch_loss(
        logits,
        targets,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        ce_class_weights=ce_class_weights,
        focal_alpha_weights=focal_alpha_weights,
    )
    if not omega_enabled:
        zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
        return ce_loss, ce_loss, zero, logits, _extract_representation_variance_stats(None)

    assert h is not None and t_h is not None
    attr_loss = torch.mean(torch.square(h - t_h))
    total_loss = ce_loss + (float(omega_lambda) * attr_loss)
    return total_loss, ce_loss, attr_loss, logits, _extract_representation_variance_stats(h)


def _beta_sample_on_device(beta_alpha: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    concentration = torch.tensor(float(beta_alpha), device=device, dtype=dtype)
    return torch.distributions.Beta(concentration, concentration).sample().to(device=device, dtype=dtype)


def _apply_mixup_torch(
    x_batch: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]:
    if int(x_batch.shape[0]) < 2:
        lam = torch.ones((), device=x_batch.device, dtype=x_batch.dtype)
        return x_batch, targets, False, lam
    lam = _beta_sample_on_device(beta_alpha, device=x_batch.device, dtype=x_batch.dtype)
    permutation = torch.randperm(int(x_batch.shape[0]), device=x_batch.device)
    lam_targets = lam.to(device=targets.device, dtype=targets.dtype)
    mixed_x = lam * x_batch + (1.0 - lam) * x_batch[permutation]
    mixed_targets = lam_targets * targets + (1.0 - lam_targets) * targets[permutation]
    return mixed_x, mixed_targets, True, lam


def _apply_cutmix_torch(
    x_batch: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]:
    if int(x_batch.shape[0]) < 2:
        lam = torch.ones((), device=x_batch.device, dtype=x_batch.dtype)
        return x_batch, targets, False, lam

    batch_size, _, height, width = x_batch.shape
    lam_sample = _beta_sample_on_device(beta_alpha, device=x_batch.device, dtype=x_batch.dtype)
    cut_ratio = torch.sqrt(torch.clamp(1.0 - lam_sample, min=0.0, max=1.0))

    cut_w = torch.clamp(torch.round(torch.tensor(float(width), device=x_batch.device, dtype=x_batch.dtype) * cut_ratio), min=1.0, max=float(width)).to(dtype=torch.int64)
    cut_h = torch.clamp(torch.round(torch.tensor(float(height), device=x_batch.device, dtype=x_batch.dtype) * cut_ratio), min=1.0, max=float(height)).to(dtype=torch.int64)

    center_x = torch.randint(0, width, (1,), device=x_batch.device, dtype=torch.int64).item()
    center_y = torch.randint(0, height, (1,), device=x_batch.device, dtype=torch.int64).item()
    x1 = max(0, center_x - int(cut_w.item()) // 2)
    y1 = max(0, center_y - int(cut_h.item()) // 2)
    x2 = min(width, center_x + (int(cut_w.item()) + 1) // 2)
    y2 = min(height, center_y + (int(cut_h.item()) + 1) // 2)

    permutation = torch.randperm(batch_size, device=x_batch.device)
    mixed_x = x_batch.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x_batch[permutation, :, y1:y2, x1:x2]

    patch_area = float(max(0, y2 - y1) * max(0, x2 - x1))
    lam = torch.tensor(
        1.0 - (patch_area / float(height * width)),
        device=x_batch.device,
        dtype=x_batch.dtype,
    )
    lam_targets = lam.to(device=targets.device, dtype=targets.dtype)
    mixed_targets = lam_targets * targets + (1.0 - lam_targets) * targets[permutation]
    return mixed_x, mixed_targets, True, lam


def _apply_batch_mix_torch(
    x_batch: torch.Tensor,
    targets: torch.Tensor,
    *,
    prob: float,
    beta_alpha: float,
    cutmix_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
    if int(x_batch.shape[0]) < 2 or float(prob) <= 0.0:
        lam = torch.ones((), device=x_batch.device, dtype=x_batch.dtype)
        return x_batch, targets, "none", lam
    if torch.rand((), device=x_batch.device) >= float(prob):
        lam = torch.ones((), device=x_batch.device, dtype=x_batch.dtype)
        return x_batch, targets, "none", lam
    if torch.rand((), device=x_batch.device) < float(cutmix_ratio):
        mixed_x, mixed_targets, _, lam = _apply_cutmix_torch(x_batch, targets, beta_alpha=beta_alpha)
        return mixed_x, mixed_targets, "cutmix", lam
    mixed_x, mixed_targets, _, lam = _apply_mixup_torch(x_batch, targets, beta_alpha=beta_alpha)
    return mixed_x, mixed_targets, "mixup", lam


def _resolve_amp_dtype(device: torch.device, amp_mode: str) -> torch.dtype | None:
    if amp_mode not in {"auto", "on", "off"}:
        raise ValueError(f"Unsupported amp_mode: {amp_mode}")
    if amp_mode == "off":
        return None
    if device.type != "cuda":
        if amp_mode == "on":
            raise SystemExit("--amp-mode=on requires CUDA")
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _make_grad_scaler(device: torch.device, amp_dtype: torch.dtype | None):
    enabled = bool(device.type == "cuda" and amp_dtype == torch.float16)
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _make_autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None)


def _configure_runtime_kernels(device: torch.device, *, fixed_input_shape: bool) -> bool:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    use_channels_last = device.type == "cuda"
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = bool(device.type == "cuda" and fixed_input_shape)
    return use_channels_last


def _backward_and_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: TorchCNN,
    *,
    grad_clip: float,
    scaler,
) -> None:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        scaler.scale(loss).backward()
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        return
    loss.backward()
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=grad_clip)
    optimizer.step()


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _benchmark_train_steps(
    *,
    forward_callable: Callable[[torch.Tensor], Any],
    raw_model: TorchCNN,
    optimizer: torch.optim.Optimizer,
    scaler,
    x_batch: torch.Tensor,
    y_batch_idx: torch.Tensor,
    num_classes: int,
    label_smoothing: float,
    use_focal_loss: bool,
    focal_gamma: float,
    ce_class_weights: torch.Tensor | None,
    focal_alpha_weights: torch.Tensor | None,
    omega_enabled: bool,
    omega_lambda: float,
    grad_clip: float,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    warmup_steps: int,
    measure_steps: int,
) -> float:
    raw_model.train()
    times: list[float] = []
    targets = _make_target_distribution(
        y_batch_idx,
        num_classes,
        label_smoothing=label_smoothing,
        dtype=torch.float32,
    )
    total_steps = int(warmup_steps) + int(measure_steps)
    for step in range(total_steps):
        optimizer.zero_grad(set_to_none=True)
        _synchronize_device(device)
        start = time.perf_counter()
        with _make_autocast_context(device, amp_dtype):
            forward_output = forward_callable(x_batch)
            loss, _, _, _, _ = _compute_total_loss_components(
                forward_output,
                targets,
                omega_enabled=omega_enabled,
                omega_lambda=omega_lambda,
                use_focal_loss=use_focal_loss,
                focal_gamma=focal_gamma,
                ce_class_weights=ce_class_weights,
                focal_alpha_weights=focal_alpha_weights,
            )
        _backward_and_step(
            loss,
            optimizer,
            raw_model,
            grad_clip=grad_clip,
            scaler=scaler,
        )
        _synchronize_device(device)
        elapsed = time.perf_counter() - start
        if step >= warmup_steps:
            times.append(elapsed)
    return float(np.median(np.asarray(times, dtype=np.float64)))


def _should_keep_compiled_model(
    eager_step_median: float,
    compiled_step_median: float,
    *,
    required_speedup_ratio: float = 0.95,
) -> bool:
    return float(compiled_step_median) < (float(eager_step_median) * float(required_speedup_ratio))


def _clone_model_for_benchmark(
    model: TorchCNN,
    *,
    device: torch.device,
    use_channels_last: bool,
) -> TorchCNN:
    cloned = copy.deepcopy(model)
    if use_channels_last:
        cloned.to(device=device, memory_format=torch.channels_last)
    else:
        cloned.to(device=device)
    return cloned


def _maybe_enable_compiled_forward_model(
    *,
    model: TorchCNN,
    args: argparse.Namespace,
    benchmark_batch: tuple[torch.Tensor, torch.Tensor] | None,
    benchmark_lr: float,
    num_classes: int,
    label_smoothing: float,
    use_focal_loss: bool,
    focal_gamma: float,
    ce_class_weights: torch.Tensor | None,
    focal_alpha_weights: torch.Tensor | None,
    omega_enabled: bool,
    omega_lambda: float,
    grad_clip: float,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    use_channels_last: bool,
) -> Callable[[torch.Tensor], Any] | None:
    if args.compile_mode == "off":
        return None
    if device.type != "cuda":
        if args.compile_mode == "on":
            raise SystemExit("--compile-mode=on requires CUDA")
        return None
    if not hasattr(torch, "compile"):
        if args.compile_mode == "on":
            raise SystemExit("torch.compile is unavailable in this PyTorch build")
        print("[warn] torch.compile unavailable; continuing in eager mode.")
        return None

    if args.compile_mode == "on":
        try:
            compiled = torch.compile(model.forward_with_omega if omega_enabled else model)
            print("Enabled torch.compile (--compile-mode=on).")
            return compiled
        except Exception as exc:
            raise SystemExit(f"Failed to enable torch.compile: {exc}") from exc

    if benchmark_batch is None:
        return None

    x_cpu, y_cpu = benchmark_batch
    x_device = _move_input_batch(x_cpu, device, use_channels_last=use_channels_last)
    y_device = _move_label_batch(y_cpu, device)

    try:
        eager_model = _clone_model_for_benchmark(model, device=device, use_channels_last=use_channels_last)
        eager_optimizer = _build_optimizer(args, eager_model, lr_value=benchmark_lr)
        eager_scaler = _make_grad_scaler(device, amp_dtype)
        eager_time = _benchmark_train_steps(
            forward_callable=eager_model.forward_with_omega if omega_enabled else eager_model,
            raw_model=eager_model,
            optimizer=eager_optimizer,
            scaler=eager_scaler,
            x_batch=x_device,
            y_batch_idx=y_device,
            num_classes=num_classes,
            label_smoothing=label_smoothing,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            ce_class_weights=ce_class_weights,
            focal_alpha_weights=focal_alpha_weights,
            omega_enabled=omega_enabled,
            omega_lambda=omega_lambda,
            grad_clip=grad_clip,
            device=device,
            amp_dtype=amp_dtype,
            warmup_steps=5,
            measure_steps=10,
        )

        compiled_model = _clone_model_for_benchmark(model, device=device, use_channels_last=use_channels_last)
        compiled_forward = torch.compile(compiled_model.forward_with_omega if omega_enabled else compiled_model)
        compiled_optimizer = _build_optimizer(args, compiled_model, lr_value=benchmark_lr)
        compiled_scaler = _make_grad_scaler(device, amp_dtype)
        compiled_time = _benchmark_train_steps(
            forward_callable=compiled_forward,
            raw_model=compiled_model,
            optimizer=compiled_optimizer,
            scaler=compiled_scaler,
            x_batch=x_device,
            y_batch_idx=y_device,
            num_classes=num_classes,
            label_smoothing=label_smoothing,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            ce_class_weights=ce_class_weights,
            focal_alpha_weights=focal_alpha_weights,
            omega_enabled=omega_enabled,
            omega_lambda=omega_lambda,
            grad_clip=grad_clip,
            device=device,
            amp_dtype=amp_dtype,
            warmup_steps=5,
            measure_steps=10,
        )

        if not _should_keep_compiled_model(eager_time, compiled_time):
            print(
                f"[info] torch.compile disabled: eager median step {eager_time:.6f}s, "
                f"compiled median step {compiled_time:.6f}s"
            )
            return None

        live_compiled_forward = torch.compile(model.forward_with_omega if omega_enabled else model)
        print(
            f"Enabled torch.compile: eager median step {eager_time:.6f}s, "
            f"compiled median step {compiled_time:.6f}s"
        )
        return live_compiled_forward
    except Exception as exc:
        print(f"[warn] torch.compile disabled: {exc}")
        return None


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    device: torch.device,
    *,
    forward_callable: Callable[[torch.Tensor], Any] | None,
    use_channels_last: bool,
    amp_dtype: torch.dtype | None,
    use_focal_loss: bool,
    focal_gamma: float,
    ce_class_weights: torch.Tensor | None,
    focal_alpha_weights: torch.Tensor | None,
    omega_enabled: bool,
    omega_lambda: float,
) -> dict[str, float]:
    if len(loader.dataset) == 0:
        return {
            "loss_total": float("nan"),
            "loss_ce": float("nan"),
            "loss_attr": float("nan"),
            "acc": float("nan"),
            "h_var_mean": float("nan"),
            "h_var_min": float("nan"),
            "h_var_max": float("nan"),
        }
    total_loss = 0.0
    total_ce_loss = 0.0
    total_attr_loss = 0.0
    total_correct = 0
    total_h_var_mean = 0.0
    total_h_var_min = 0.0
    total_h_var_max = 0.0
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for x_cpu, y_cpu in loader:
            x_batch = _move_input_batch(x_cpu, device, use_channels_last=use_channels_last)
            y_batch = _move_label_batch(y_cpu, device)
            targets = _make_target_distribution(
                y_batch,
                num_classes,
                label_smoothing=0.0,
                dtype=torch.float32,
            )
            with _make_autocast_context(device, amp_dtype):
                forward_output = (forward_callable or model)(x_batch)
                loss, ce_loss, attr_loss, logits, h_stats = _compute_total_loss_components(
                    forward_output,
                    targets,
                    omega_enabled=omega_enabled,
                    omega_lambda=omega_lambda,
                    use_focal_loss=use_focal_loss,
                    focal_gamma=focal_gamma,
                    ce_class_weights=ce_class_weights,
                    focal_alpha_weights=focal_alpha_weights,
                )
            total_loss += float(loss.item())
            total_ce_loss += float(ce_loss.item())
            total_attr_loss += float(attr_loss.item())
            total_correct += int((torch.argmax(logits, dim=1) == y_batch).sum().item())
            total_h_var_mean += h_stats.mean
            total_h_var_min += h_stats.min
            total_h_var_max += h_stats.max
            num_batches += 1
    return {
        "loss_total": total_loss / max(1, num_batches),
        "loss_ce": total_ce_loss / max(1, num_batches),
        "loss_attr": total_attr_loss / max(1, num_batches),
        "acc": total_correct / max(1, len(loader.dataset)),
        "h_var_mean": total_h_var_mean / max(1, num_batches),
        "h_var_min": total_h_var_min / max(1, num_batches),
        "h_var_max": total_h_var_max / max(1, num_batches),
    }


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
    """Apply the temporary freeze policy after each `model.train()` call."""
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
    """Recreate the optimizer intentionally when freeze state changes."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=lr_value, weight_decay=args.weight_decay)
    return torch.optim.SGD(parameters, lr=lr_value, momentum=args.momentum, weight_decay=args.weight_decay)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr_value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train image classifier with the PyTorch backend")
    parser.add_argument("--help-md", action="store_true", help="Print Markdown guide and exit")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory with train images")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count (defaults to 4 for streaming, 0 for preloaded)")
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
    parser.add_argument("--experiment-dir", type=str, default="runs", help="Root directory for Omega experiment artifacts")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rotation", type=float, default=12.0, help="Maximum absolute random rotation in degrees")
    parser.add_argument("--brightness", type=float, default=0.2, help="Brightness jitter strength in [0, 1]")
    parser.add_argument("--contrast", type=float, default=0.2, help="Contrast jitter strength in [0, 1]")
    parser.add_argument("--saturation", type=float, default=0.2, help="Saturation jitter strength in [0, 1]")
    parser.add_argument("--model-width-scale", type=float, default=0.75, help="Width multiplier for the stage-2 convolution block")
    parser.add_argument("--omega-loss", action=argparse.BooleanOptionalAction, default=False, help="Enable the Phase 1 Omega-loss auxiliary branch")
    parser.add_argument("--omega-lambda", type=float, default=0.0, help="Weight applied to the Phase 1 Omega attractor loss")
    parser.add_argument("--omega-projector-depth", type=int, default=1, help="Omega projector depth: 1 or 2 linear layers")
    parser.add_argument("--omega-hidden-dim", type=int, default=DEFAULT_OMEGA_FEATURE_DIM, help="Hidden width for the 2-layer Omega projector")
    parser.add_argument("--allow-unlabeled-root", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="If true, stream batches from disk; if false, preload all images into memory.")
    parser.add_argument("--amp-mode", choices=["auto", "on", "off"], default="auto", help="AMP policy for CUDA training")
    parser.add_argument("--compile-mode", choices=["auto", "on", "off"], default="auto", help="torch.compile policy")
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
        omega_lambda, omega_projector_depth, omega_hidden_dim = _validate_omega_args(
            omega_loss=bool(args.omega_loss),
            omega_lambda=args.omega_lambda,
            omega_projector_depth=args.omega_projector_depth,
            omega_hidden_dim=args.omega_hidden_dim,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = _resolve_device(args.device)
    amp_dtype = _resolve_amp_dtype(device, args.amp_mode)
    use_channels_last = _configure_runtime_kernels(device, fixed_input_shape=True)

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
    pin_memory = bool(device.type == "cuda")
    batch_size = max(1, int(args.batch_size))
    num_workers = _resolve_num_workers(bool(args.streaming), args.num_workers)

    x_train = None
    x_val = None
    if not args.streaming:
        print("Preloading images into memory (float32)...")
        x_train = _load_all_images(train_paths, input_size)
        if y_val is not None and len(val_paths) > 0:
            x_val = _load_all_images(val_paths, input_size)

    train_dataset = TorchImageDataset(
        labels=y_train,
        input_size=input_size,
        paths=train_paths if args.streaming else None,
        preloaded_images=x_train,
        augment_config=augment_config if args.augment else None,
    )
    val_dataset = None
    if y_val is not None and len(val_paths) > 0:
        val_dataset = TorchImageDataset(
            labels=y_val,
            input_size=input_size,
            paths=val_paths if args.streaming else None,
            preloaded_images=x_val,
            augment_config=None,
        )

    train_loader = _build_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        base_seed=args.seed,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = _build_data_loader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers if args.streaming else 0,
            pin_memory=pin_memory,
            base_seed=args.seed + 10_000,
        )

    model = TorchCNN(
        input_size=input_size,
        num_classes=num_classes,
        seed=args.seed,
        dropout_p=args.dropout,
        width_scale=width_scale,
        omega_enabled=bool(args.omega_loss),
        omega_projector_depth=omega_projector_depth,
        omega_hidden_dim=omega_hidden_dim,
    )
    if use_channels_last:
        model.to(device=device, memory_format=torch.channels_last)
    else:
        model.to(device=device)
    print(f"Using stage2_channels={model.stage2_channels} (width_scale={model.width_scale:.3f})")
    if args.init_from:
        try:
            model.load_weights(args.init_from, map_location=device)
            print(f"Initialized weights from: {args.init_from}")
        except Exception as exc:
            print(f"[warn] Strict load failed: {exc}")
            loaded, skipped = load_weights_forgiving(model, args.init_from, skip_prefixes=("fc2.",))
            print(f"Loaded {len(loaded)} keys from checkpoint; skipped {len(skipped)} (for example, classifier or resized feature tensors)")
            if use_channels_last:
                model.to(device=device, memory_format=torch.channels_last)
            else:
                model.to(device=device)

    num_batches = max(1, len(train_loader))
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
    best_epoch = 0
    collapse_low_variance_epochs = 0
    final_epoch_metrics: dict[str, Any] | None = None

    _apply_backbone_freeze_state(model, backbone_frozen=False, freeze_bn_affine=args.freeze_bn_affine)
    optimizer: torch.optim.Optimizer = _build_optimizer(args, model, lr_value=current_phase_base_lr)
    scaler = _make_grad_scaler(device, amp_dtype)

    compile_probe_batch = None
    try:
        compile_probe_batch = next(iter(train_loader))
    except StopIteration:
        compile_probe_batch = None
    compiled_forward_model = _maybe_enable_compiled_forward_model(
        model=model,
        args=args,
        benchmark_batch=compile_probe_batch,
        benchmark_lr=current_phase_base_lr,
        num_classes=num_classes,
        label_smoothing=float(args.label_smoothing),
        use_focal_loss=base_use_focal_loss,
        focal_gamma=focal_gamma,
        ce_class_weights=ce_class_weights,
        focal_alpha_weights=focal_alpha_weights,
        omega_enabled=bool(args.omega_loss),
        omega_lambda=omega_lambda,
        grad_clip=float(args.grad_clip),
        device=device,
        amp_dtype=amp_dtype,
        use_channels_last=use_channels_last,
    )

    ema: ModelEMA | None = None
    if args.ema:
        ema_model = TorchCNN(
            input_size=input_size,
            num_classes=num_classes,
            seed=args.seed,
            dropout_p=args.dropout,
            width_scale=width_scale,
            omega_enabled=bool(args.omega_loss),
            omega_projector_depth=omega_projector_depth,
            omega_hidden_dim=omega_hidden_dim,
        )
        if use_channels_last:
            ema_model.to(device=device, memory_format=torch.channels_last)
        else:
            ema_model.to(device=device)
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        ema_model.eval()
        ema = ModelEMA(ema_model, decay=ema_decay, phase_warmup_steps=max(1, num_batches))
        ema.sync_from(model)

    checkpoint_metadata = {
        "class_names": list(resolved_class_names),
        "is_ema": False,
        "ema_decay": None,
    }
    experiment_run_dir: Path | None = None
    experiment_metrics_path: Path | None = None
    experiment_summary_path: Path | None = None
    if args.omega_loss:
        experiment_run_dir, experiment_config_path, experiment_metrics_path, experiment_summary_path = _prepare_experiment_artifacts(
            experiment_root=Path(args.experiment_dir),
            checkpoint_path=Path(args.checkpoint),
            seed=args.seed,
            omega_lambda=omega_lambda,
        )
        _write_json(
            experiment_config_path,
            {
                "args": vars(args),
                "resolved_device": str(device),
                "resolved_num_classes": int(num_classes),
                "resolved_class_names": list(resolved_class_names),
                "resolved_width_scale": float(width_scale),
                "resolved_stage2_channels": int(model.stage2_channels),
                "omega_enabled": bool(args.omega_loss),
                "omega_lambda": float(omega_lambda),
                "omega_projector_depth": int(omega_projector_depth),
                "omega_hidden_dim": int(omega_hidden_dim),
                "checkpoint_path": str(Path(args.checkpoint).resolve()),
            },
        )

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
            scaler = _make_grad_scaler(device, amp_dtype)
            print(
                f"Phase {current_phase_index + 1}/{args.phase_count}  "
                f"base_lr={current_phase_base_lr:.6f}  "
                f"epochs={phase_config.epochs_in_phase}  "
                "cosine_restarts_per_phase=yes"
            )

        model.train()
        epoch_backbone_frozen = backbone_frozen
        _apply_backbone_freeze_state(model, backbone_frozen=epoch_backbone_frozen, freeze_bn_affine=args.freeze_bn_affine)
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_attr_loss = 0.0
        epoch_correct = 0
        epoch_h_var_mean = 0.0
        epoch_h_var_min = float("inf")
        epoch_h_var_max = -float("inf")

        for batch_index, (x_cpu, y_cpu) in enumerate(train_loader):
            y_batch_idx = _move_label_batch(y_cpu, device)
            x_batch = _move_input_batch(x_cpu, device, use_channels_last=use_channels_last)

            mix_mode = "none"
            if args.mixup:
                target_distribution = _make_target_distribution(
                    y_batch_idx,
                    num_classes,
                    label_smoothing=0.0,
                    dtype=torch.float32,
                )
                x_batch, target_distribution, mix_mode, _ = _apply_batch_mix_torch(
                    x_batch,
                    target_distribution,
                    prob=mixup_prob,
                    beta_alpha=mixup_alpha,
                    cutmix_ratio=cutmix_ratio,
                )
                if mix_mode == "none":
                    target_distribution = _make_target_distribution(
                        y_batch_idx,
                        num_classes,
                        label_smoothing=float(args.label_smoothing),
                        dtype=torch.float32,
                    )
            else:
                target_distribution = _make_target_distribution(
                    y_batch_idx,
                    num_classes,
                    label_smoothing=float(args.label_smoothing),
                    dtype=torch.float32,
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
            forward_callable: Callable[[torch.Tensor], Any]
            if bool(args.omega_loss):
                forward_callable = compiled_forward_model if compiled_forward_model is not None else model.forward_with_omega
            else:
                forward_callable = compiled_forward_model if compiled_forward_model is not None else model
            with _make_autocast_context(device, amp_dtype):
                batch_use_focal_loss = bool(base_use_focal_loss and mix_mode == "none")
                forward_output = forward_callable(x_batch)
                loss, ce_loss, attr_loss, logits, h_stats = _compute_total_loss_components(
                    forward_output,
                    target_distribution,
                    omega_enabled=bool(args.omega_loss),
                    omega_lambda=omega_lambda,
                    use_focal_loss=batch_use_focal_loss,
                    focal_gamma=focal_gamma,
                    ce_class_weights=ce_class_weights,
                    focal_alpha_weights=focal_alpha_weights,
                )
            _backward_and_step(
                loss,
                optimizer,
                model,
                grad_clip=float(args.grad_clip),
                scaler=scaler,
            )
            if ema is not None:
                phase_step = phase_config.epoch_index_in_phase * num_batches + batch_index
                ema.update(model, step_in_phase=phase_step, mix_active=(mix_mode != "none"))

            epoch_loss += float(loss.item())
            epoch_ce_loss += float(ce_loss.item())
            epoch_attr_loss += float(attr_loss.item())
            metric_targets = torch.argmax(target_distribution, dim=1)
            epoch_correct += int((torch.argmax(logits, dim=1) == metric_targets).sum().item())
            epoch_h_var_mean += h_stats.mean
            if not np.isnan(h_stats.min):
                epoch_h_var_min = min(epoch_h_var_min, h_stats.min)
            if not np.isnan(h_stats.max):
                epoch_h_var_max = max(epoch_h_var_max, h_stats.max)

        avg_loss = epoch_loss / max(1, num_batches)
        avg_ce_loss = epoch_ce_loss / max(1, num_batches)
        avg_attr_loss = epoch_attr_loss / max(1, num_batches)
        train_acc = epoch_correct / max(1, len(train_dataset))
        train_h_var_mean = epoch_h_var_mean / max(1, num_batches)
        train_h_var_min = epoch_h_var_min if epoch_h_var_min != float("inf") else float("nan")
        train_h_var_max = epoch_h_var_max if epoch_h_var_max != -float("inf") else float("nan")

        if val_loader is not None and y_val is not None and len(val_paths) > 0:
            eval_model = ema.ema_model if ema is not None else model
            eval_model.eval()
            eval_forward_callable: Callable[[torch.Tensor], Any] | None = None
            if eval_model is model and compiled_forward_model is not None:
                eval_forward_callable = compiled_forward_model
            elif bool(args.omega_loss):
                eval_forward_callable = eval_model.forward_with_omega
            val_metrics = evaluate_loader(
                eval_model,
                val_loader,
                num_classes,
                device,
                forward_callable=eval_forward_callable,
                use_channels_last=use_channels_last,
                amp_dtype=amp_dtype,
                use_focal_loss=base_use_focal_loss,
                focal_gamma=focal_gamma,
                ce_class_weights=ce_class_weights,
                focal_alpha_weights=focal_alpha_weights,
                omega_enabled=bool(args.omega_loss),
                omega_lambda=omega_lambda,
            )
            val_loss = val_metrics["loss_total"]
            val_acc = val_metrics["acc"]

            improved = (val_loss < (best_val_loss - args.min_delta)) if args.early_stop_metric == "val_loss" else (val_acc > (best_val_acc + args.min_delta))
            if improved:
                if args.early_stop_metric == "val_loss":
                    best_val_loss = val_loss
                else:
                    best_val_acc = val_acc
                best_epoch = epoch + 1
                checkpoint_model = ema.ema_model if ema is not None else model
                checkpoint_metadata["is_ema"] = bool(ema is not None)
                checkpoint_metadata["ema_decay"] = float(ema_decay) if ema is not None else None
                checkpoint_model.save_weights(args.checkpoint, metadata=checkpoint_metadata)
                best_saved = True
                bad_epochs = 0
            else:
                bad_epochs += 1
            if args.early_stop_metric != "val_loss":
                best_val_loss = min(best_val_loss, val_loss)
            if args.early_stop_metric != "val_acc":
                best_val_acc = max(best_val_acc, val_acc)

            if epoch_backbone_frozen:
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
                    scaler = _make_grad_scaler(device, amp_dtype)
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
                        scaler = _make_grad_scaler(device, amp_dtype)
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
            final_epoch_metrics = {
                "epoch": epoch + 1,
                "phase": current_phase_index + 1,
                "mode": mode_text,
                "lr": float(current_lr),
                "train_loss_total": float(avg_loss),
                "train_loss_ce": float(avg_ce_loss),
                "train_loss_attr": float(avg_attr_loss),
                "train_acc": float(train_acc),
                "val_loss_total": float(val_metrics["loss_total"]),
                "val_loss_ce": float(val_metrics["loss_ce"]),
                "val_loss_attr": float(val_metrics["loss_attr"]),
                "val_acc": float(val_acc),
                "generalization_gap": float(gap),
                "h_var_mean": float(train_h_var_mean),
                "h_var_min": float(train_h_var_min),
                "h_var_max": float(train_h_var_max),
                "val_h_var_mean": float(val_metrics["h_var_mean"]),
                "val_h_var_min": float(val_metrics["h_var_min"]),
                "val_h_var_max": float(val_metrics["h_var_max"]),
                "improved": bool(improved),
            }
            if args.omega_loss:
                collapse_low_variance_epochs = (
                    collapse_low_variance_epochs + 1
                    if train_h_var_mean < OMEGA_COLLAPSE_VARIANCE_THRESHOLD
                    else 0
                )
                final_epoch_metrics["representation_collapse_warning"] = bool(
                    collapse_low_variance_epochs >= OMEGA_COLLAPSE_EPOCHS
                )
                assert experiment_metrics_path is not None
                _append_jsonl(experiment_metrics_path, final_epoch_metrics)
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
                    scaler = _make_grad_scaler(device, amp_dtype)
            mode_text = "head-only" if epoch_backbone_frozen else "full"
            print(
                f"Epoch {epoch + 1}/{args.epochs}  phase={current_phase_index + 1}/{args.phase_count}  "
                f"mode={mode_text}  lr={current_lr:.6f}  train_loss={avg_loss:.4f}  train_acc={train_acc:.3f}"
            )
            final_epoch_metrics = {
                "epoch": epoch + 1,
                "phase": current_phase_index + 1,
                "mode": mode_text,
                "lr": float(current_lr),
                "train_loss_total": float(avg_loss),
                "train_loss_ce": float(avg_ce_loss),
                "train_loss_attr": float(avg_attr_loss),
                "train_acc": float(train_acc),
                "val_loss_total": float("nan"),
                "val_loss_ce": float("nan"),
                "val_loss_attr": float("nan"),
                "val_acc": float("nan"),
                "generalization_gap": float("nan"),
                "h_var_mean": float(train_h_var_mean),
                "h_var_min": float(train_h_var_min),
                "h_var_max": float(train_h_var_max),
                "val_h_var_mean": float("nan"),
                "val_h_var_min": float("nan"),
                "val_h_var_max": float("nan"),
                "improved": False,
            }
            if args.omega_loss:
                collapse_low_variance_epochs = (
                    collapse_low_variance_epochs + 1
                    if train_h_var_mean < OMEGA_COLLAPSE_VARIANCE_THRESHOLD
                    else 0
                )
                final_epoch_metrics["representation_collapse_warning"] = bool(
                    collapse_low_variance_epochs >= OMEGA_COLLAPSE_EPOCHS
                )
                assert experiment_metrics_path is not None
                _append_jsonl(experiment_metrics_path, final_epoch_metrics)

    if val_loader is None or y_val is None or len(val_paths) == 0:
        checkpoint_model = ema.ema_model if ema is not None else model
        checkpoint_metadata["is_ema"] = bool(ema is not None)
        checkpoint_metadata["ema_decay"] = float(ema_decay) if ema is not None else None
        checkpoint_model.save_weights(args.checkpoint, metadata=checkpoint_metadata)
        best_saved = True
    if best_saved:
        print(f"Saved checkpoint: {args.checkpoint}")

    if args.omega_loss and experiment_summary_path is not None:
        representation_collapse_warning = bool(
            final_epoch_metrics is not None
            and final_epoch_metrics.get("representation_collapse_warning", False)
        )
        _write_json(
            experiment_summary_path,
            {
                "run_dir": str(experiment_run_dir) if experiment_run_dir is not None else None,
                "checkpoint_path": str(Path(args.checkpoint).resolve()),
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
                "best_val_acc": float(best_val_acc),
                "final_epoch_metrics": final_epoch_metrics or {},
                "representation_collapse_warning": representation_collapse_warning,
                "collapse_variance_threshold": float(OMEGA_COLLAPSE_VARIANCE_THRESHOLD),
                "collapse_consecutive_epochs": int(OMEGA_COLLAPSE_EPOCHS),
                "omega_lambda": float(omega_lambda),
                "omega_projector_depth": int(omega_projector_depth),
                "omega_hidden_dim": int(omega_hidden_dim),
                "contraction_is_empirical": True,
            },
        )
        print(f"Saved Omega experiment artifacts: {experiment_run_dir}")

    if args.enforce_readonly_dataset and pre_signature is not None:
        post_signature = tree_signature(data_dir)
        if post_signature != pre_signature:
            raise SystemExit("Dataset changed during run; aborting (read-only enforcement).")

    print("Training finished.")


if __name__ == "__main__":
    main()
