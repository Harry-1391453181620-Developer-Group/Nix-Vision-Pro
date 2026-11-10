"""PyTorch training backend for MAOL (CNN-only) and the CE linear control.

MAOL workflow:
1. Warm-up on the prototype head with epoch-wise class means and CE only.
2. Save a warm-up checkpoint containing backbone, wells, statistics and margins.
3. Optional scale probe from that checkpoint (100 steps, six loss terms + grads).
4. Full MAOL training from the warm-up checkpoint with tuned lambdas.

CE control (`--maol-mode none`):
- raw ``fc1 + ReLU`` embedding, ``Linear(256, N)`` head, CE only;
- no warm-up/probe, optional EMA, same schedule/augmentation as MAOL.
"""

from __future__ import annotations

import argparse
from collections.abc import Sized
from functools import partial
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import config
from backends.torch.model import TorchCNN, load_checkpoint_state, resolve_checkpoint_runtime_config
from data.loaders import load_image
from data.nested import DatasetIndex, discover_nested_dataset, split_dataset, stratified_split
from data.preprocessing import preprocess_image
from maol.losses import compute_maol_losses
from maol.stats import (
    WarmupStats,
    apply_warmup_stats,
    compute_class_means,
    compute_warmup_statistics,
)
from utils.safety import install_dataset_write_guard, tree_signature
from utils.training import (
    ModelEMA,
    augment_batch,
    build_epoch_phase_map,
    compute_effective_learning_rate,
    compute_phase_learning_rate,
    parse_bool_flag,
    validate_augmentation_args,
    validate_ema_decay,
    validate_freeze_cycle_args,
    validate_model_width_scale,
    validate_phase_learning_rates,
)

install_dataset_write_guard()

DEFAULT_LAMBDAS = {
    "intra": 1.0,
    "sep": 1.0,
    "nest": 1.0,
    "cov": 1.0,
    "center": 1.0,
}
MAOL_WARMUP_TAU_STEPS = 100


class TorchImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Image dataset used by the MAOL trainer."""

    def __init__(
        self,
        paths: Sequence[Path],
        labels: np.ndarray,
        input_size: tuple[int, int],
        augment_config=None,
    ) -> None:
        if len(paths) != int(labels.shape[0]):
            raise ValueError("paths and labels must have matching lengths")
        self.paths = tuple(Path(path) for path in paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.augment_config = augment_config

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = load_image(self.paths[index])
        image = preprocess_image(
            image,
            target_size=self.input_size,
            normalize_to=config.NORMALIZE_TO,
            input_value_range=config.INPUT_VALUE_RANGE,
        ).astype(np.float32)
        if self.augment_config is not None:
            seed = int(np.random.randint(0, 2**31 - 1))
            image = augment_batch(image[None, ...], np.random.default_rng(seed), self.augment_config)[0]
        image_nchw = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(image_nchw, dtype=np.float32)), int(self.labels[index])


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    seed = int(base_seed) + int(worker_id)
    np.random.seed(seed)
    random.seed(seed)


def _collate_image_batch(batch: Sequence[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = zip(*batch)
    return (
        torch.stack(images, dim=0).contiguous(),
        torch.as_tensor(labels, dtype=torch.int64).contiguous(),
    )


def _build_loader(
    dataset: Dataset[tuple[torch.Tensor, int]],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": _collate_image_batch,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
        kwargs["worker_init_fn"] = partial(_seed_worker, base_seed=int(seed))
    return DataLoader[tuple[torch.Tensor, torch.Tensor]](dataset, **kwargs)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device=cuda requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    amp_module = getattr(torch, "amp", None)
    grad_scaler_cls = getattr(amp_module, "GradScaler", None) if amp_module is not None else None
    if grad_scaler_cls is not None:
        return grad_scaler_cls("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, amp_dtype: torch.dtype | None):
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr_value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr_value)


def _build_optimizer(
    args: argparse.Namespace,
    model: TorchCNN,
    lr_value: float,
) -> torch.optim.Optimizer:
    regular: list[torch.nn.Parameter] = []
    prototype: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("head.centers") or name.startswith("head.raw_radii"):
            prototype.append(parameter)
        else:
            regular.append(parameter)
    groups: list[dict[str, Any]] = [{"params": regular, "weight_decay": float(args.weight_decay)}]
    if prototype:
        groups.append({"params": prototype, "weight_decay": 0.0})
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=float(lr_value))
    return torch.optim.SGD(groups, lr=float(lr_value), momentum=float(args.momentum))


def _apply_backbone_freeze_state(model: TorchCNN, frozen: bool, freeze_bn_affine: bool) -> None:
    for parameter in model.iter_head_parameters():
        parameter.requires_grad = True
    for parameter in model.iter_backbone_parameters():
        parameter.requires_grad = not frozen
    for bn_layer in model.backbone_batchnorm_layers():
        if frozen:
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


def _set_prototype_trainable(model: TorchCNN, trainable: bool) -> None:
    model.head.set_trainable(trainable)


def _grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach().float(), ord=2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return float("nan")
    return float(torch.linalg.vector_norm(torch.stack(norms), ord=2).item())


def _backward_and_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: TorchCNN,
    *,
    grad_clip: float,
    scaler,
) -> float:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = _grad_norm(parameters)
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
        return norm
    loss.backward()
    norm = _grad_norm(parameters)
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=float(grad_clip))
    optimizer.step()
    return norm


def _class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    weights = np.zeros((num_classes,), dtype=np.float64)
    present = counts > 0
    if not np.any(present):
        return np.ones((num_classes,), dtype=np.float64)
    weights[present] = labels.size / (np.sum(present) * counts[present])
    weights[present] /= np.mean(weights[present])
    return weights


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_serializable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_serializable(payload), sort_keys=True) + "\n")


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serializable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _prepare_run_dir(args: argparse.Namespace, tag: str, index: DatasetIndex) -> Path:
    root = Path(args.json_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{timestamp}-{tag}-seed_{int(args.seed)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "config.json",
        {
            "args": vars(args),
            "tag": tag,
            "class_names": list(index.class_names),
            "tree": index.tree.to_dict(),
            "class_counts": index.class_counts.tolist(),
            "node_weights": index.node_weights.tolist(),
        },
    )
    return run_dir


def _move_module_to_device(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    module.to(device=device)
    if device.type == "cuda":
        module.to(memory_format=torch.channels_last)
    return module


def _move_batch(x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    if device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    y = y.to(device=device, dtype=torch.long, non_blocking=device.type == "cuda")
    return x, y


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    correct = int((torch.argmax(logits, dim=1) == labels).sum().item())
    return correct, int(labels.numel())


def _empty_metrics() -> dict[str, float]:
    return {
        "loss_total": float("nan"),
        "loss_ce": float("nan"),
        "loss_intra": float("nan"),
        "loss_sep": float("nan"),
        "loss_nest": float("nan"),
        "loss_cov": float("nan"),
        "loss_center": float("nan"),
        "acc": float("nan"),
        "gradient_norm": float("nan"),
    }


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    device: torch.device,
    *,
    lambdas: Mapping[str, float],
    gamma: float,
    eta: float,
    class_weights: torch.Tensor | None,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    if loader is None or len(cast(Sized, loader.dataset)) == 0:
        return _empty_metrics()
    model.eval()
    totals = {key: 0.0 for key in ("total", "ce", "intra", "sep", "nest", "cov", "center")}
    correct = 0
    count = 0
    for x_cpu, y_cpu in loader:
        x, y = _move_batch(x_cpu, y_cpu, device)
        with _autocast(device, amp_dtype):
            h = cast(TorchCNN, model).forward_embedding(x)
            terms = compute_maol_losses(
                cast(TorchCNN, model).head,
                h,
                y,
                lambdas=lambdas,
                gamma=gamma,
                eta=eta,
                class_weights=class_weights,
            )
        totals["total"] += float(terms.total.item())
        totals["ce"] += float(terms.ce.item())
        totals["intra"] += float(terms.intra.item())
        totals["sep"] += float(terms.sep.item())
        totals["nest"] += float(terms.nest.item())
        totals["cov"] += float(terms.cov.item())
        totals["center"] += float(terms.center.item())
        batch_correct, batch_count = _accuracy(terms.logits, y)
        correct += batch_correct
        count += batch_count
    batches = max(1, len(loader))
    return {
        "loss_total": totals["total"] / batches,
        "loss_ce": totals["ce"] / batches,
        "loss_intra": totals["intra"] / batches,
        "loss_sep": totals["sep"] / batches,
        "loss_nest": totals["nest"] / batches,
        "loss_cov": totals["cov"] / batches,
        "loss_center": totals["center"] / batches,
        "acc": correct / max(1, count),
        "gradient_norm": float("nan"),
    }


@torch.no_grad()
def evaluate_warmup_ce(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    device: torch.device,
    *,
    class_weights: torch.Tensor | None,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    if loader is None or len(cast(Sized, loader.dataset)) == 0:
        return {"loss_ce": float("nan"), "acc": float("nan")}
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    batches = 0
    for x_cpu, y_cpu in loader:
        x, y = _move_batch(x_cpu, y_cpu, device)
        with _autocast(device, amp_dtype):
            h = cast(TorchCNN, model).forward_embedding(x)
            logits = cast(TorchCNN, model).head.logits(h)
            if class_weights is None:
                loss = F.cross_entropy(logits, y)
            else:
                per_sample = F.cross_entropy(logits, y, reduction="none")
                weights = class_weights.to(device=logits.device, dtype=logits.dtype)[y]
                loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-12)
        total_loss += float(loss.item())
        batch_correct, batch_count = _accuracy(logits, y)
        correct += batch_correct
        count += batch_count
        batches += 1
    return {"loss_ce": total_loss / max(1, batches), "acc": correct / max(1, count)}


def _expanded_labels(counts: torch.Tensor) -> list[int]:
    labels: list[int] = []
    for class_index, count in enumerate(counts.detach().cpu().tolist()):
        labels.extend([class_index] * int(count))
    return labels


def _resolve_warmup_tau(
    tau_raw: float,
    tau_floor: float,
    global_step: int,
) -> tuple[float, float, bool]:
    """Resolve the warm-up temperature with a fallback, not a hard clamp.

    The document says tau=1 for the first 100 iterations and then
    ``mean ||h-c||^2``.  If the class means have collapsed, that estimate is
    meaningless (e.g. 1e-6), so fall back to 1.0 for stability.  A tiny but
    nonzero floor would still massively amplify the logits.
    """
    raw = float(tau_raw)
    fallback = (not np.isfinite(raw)) or raw < float(tau_floor)
    if int(global_step) < MAOL_WARMUP_TAU_STEPS:
        return 1.0, raw, bool(fallback)
    if fallback:
        return 1.0, raw, True
    return max(raw, 0.0), raw, False


def _summarize_values(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _parameter_groups(model: TorchCNN) -> dict[str, list[torch.nn.Parameter]]:
    groups: dict[str, list[torch.nn.Parameter]] = {
        "backbone": [],
        "fc1": [],
        "centers": [],
        "radii": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("head.centers"):
            groups["centers"].append(parameter)
        elif name.startswith("head.raw_radii"):
            groups["radii"].append(parameter)
        elif name.startswith("fc1"):
            groups["fc1"].append(parameter)
        else:
            groups["backbone"].append(parameter)
    return groups


def _build_warmup_diagnostics(
    stats: WarmupStats,
    val_acc: float,
    num_classes: int,
    *,
    tau_floor: float,
    acc_factor: float,
    max_q_ratio: float,
    min_d_inter_ratio: float,
    max_tau0: float,
    tau_fallback_epochs: int = 0,
) -> dict[str, Any]:
    q = stats.q.detach().float().cpu().numpy()
    q_median = float(np.median(q)) if q.size else 0.0
    q_max = float(np.max(q)) if q.size else 0.0
    q_ratio_value = float(q_max / q_median) if q_median > 0.0 else float("inf")
    d_ratio_value = (
        float(stats.d_inter / stats.d_intra) if stats.d_intra > 0.0 else float("inf")
    )
    acc_threshold = float(acc_factor) / max(1, int(num_classes))

    checks = {
        "val_acc": {
            "value": float(val_acc),
            "threshold": acc_threshold,
            "passed": bool(np.isfinite(val_acc) and val_acc >= acc_threshold),
        },
        "d_inter_over_d_intra": {
            "value": d_ratio_value,
            "threshold": float(min_d_inter_ratio),
            "passed": bool(np.isfinite(d_ratio_value) and d_ratio_value >= float(min_d_inter_ratio)),
        },
        "q_max_over_median": {
            "value": q_ratio_value,
            "threshold": float(max_q_ratio),
            "passed": bool(np.isfinite(q_ratio_value) and q_ratio_value <= float(max_q_ratio)),
        },
        "tau0_ge_floor": {
            "value": float(stats.tau0),
            "threshold": float(tau_floor),
            "passed": bool(stats.tau0 >= float(tau_floor)),
        },
        "tau0_le_ceiling": {
            "value": float(stats.tau0),
            "threshold": float(max_tau0),
            "passed": bool(np.isfinite(stats.tau0) and stats.tau0 <= float(max_tau0)),
        },
    }
    failures: list[str] = []
    if not checks["val_acc"]["passed"]:
        failures.append(
            f"val_acc={val_acc:.4f} < {acc_threshold:.4f} (2/{num_classes})"
        )
    if not checks["d_inter_over_d_intra"]["passed"]:
        failures.append(
            f"d_inter/d_intra={d_ratio_value:.3f} < {float(min_d_inter_ratio):.3f}"
        )
    if not checks["q_max_over_median"]["passed"]:
        failures.append(
            f"q_max/q_median={q_ratio_value:.2f} > {float(max_q_ratio):.2f}"
        )
    if not checks["tau0_ge_floor"]["passed"]:
        failures.append(f"tau0={stats.tau0:.6g} < tau_floor={float(tau_floor):.6g}")
    if not checks["tau0_le_ceiling"]["passed"]:
        failures.append(f"tau0={stats.tau0:.6g} > tau_ceiling={float(max_tau0):.6g}")

    return {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "metrics": {
            "val_acc": float(val_acc),
            "num_classes": int(num_classes),
            "d_intra": float(stats.d_intra),
            "d_inter": float(stats.d_inter),
            "d_inter_over_d_intra": d_ratio_value,
            "q_median": q_median,
            "q_max": q_max,
            "q_max_over_median": q_ratio_value,
            "tau0": float(stats.tau0),
            "tau_warm_raw": float(stats.tau_warm),
            "tau_fallback_epochs": int(tau_fallback_epochs),
            "acc_factor": float(acc_factor),
            "max_q_ratio": float(max_q_ratio),
            "min_d_inter_ratio": float(min_d_inter_ratio),
            "max_tau0": float(max_tau0),
            "tau_floor": float(tau_floor),
        },
    }


def _check_warmup_or_abort(
    stats: WarmupStats,
    val_acc: float,
    num_classes: int,
    args: argparse.Namespace,
    run_dir: Path,
    *,
    tau_fallback_epochs: int = 0,
) -> dict[str, Any]:
    diagnostics = _build_warmup_diagnostics(
        stats,
        val_acc,
        num_classes,
        tau_floor=float(args.maol_tau_floor),
        acc_factor=float(args.maol_warmup_min_acc_factor),
        max_q_ratio=float(args.maol_warmup_max_q_ratio),
        min_d_inter_ratio=float(args.maol_warmup_min_d_inter_ratio),
        max_tau0=float(args.maol_warmup_max_tau0),
        tau_fallback_epochs=tau_fallback_epochs,
    )
    _write_json(run_dir / "warmup_diagnostics.json", diagnostics)
    if args.maol_warmup_sanity and not diagnostics["passed"]:
        _write_json(run_dir / "warmup_rejected.json", diagnostics)
        print("MAOL warm-up sanity check FAILED:")
        for failure in diagnostics["failures"]:
            print(f"  - {failure}")
        print(f"Diagnostics: {run_dir / 'warmup_rejected.json'}")
        raise SystemExit(2)
    return diagnostics


def run_warmup(
    model: TorchCNN,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    clean_train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    args: argparse.Namespace,
    device: torch.device,
    *,
    class_weights: torch.Tensor | None,
) -> tuple[WarmupStats, list[dict[str, Any]]]:
    """Warm-up: frozen prototype centers, tau=1 for 100 steps, then tau_warm, CE only."""
    head = model.head
    head.set_trainable(False)
    head.tau.fill_(1.0)
    warmup_lr = (
        float(args.maol_warmup_lr)
        if args.maol_warmup_lr is not None
        else 2.0 * float(args.lr[0])
    )
    print(f"Warm-up LR: {warmup_lr:.6f}")
    optimizer = _build_optimizer(args, model, lr_value=warmup_lr)
    amp_dtype = _resolve_amp_dtype(device, args.amp_mode)
    scaler = _make_grad_scaler(device, amp_dtype)

    rows: list[dict[str, Any]] = []
    global_step = 0
    tau_warm = 1.0
    tau_fallback_epochs = 0
    num_batches = max(1, len(train_loader))
    for epoch in range(int(args.maol_warmup_epochs)):
        model.eval()
        means, counts, tau_epoch = compute_class_means(
            model.forward_embedding,
            clean_train_loader,
            device,
            model.num_classes,
            model.embedding_dim,
        )
        head.set_centers(means)
        head.set_node_weights(model.tree.node_weights_from_labels(_expanded_labels(counts)))
        tau_warm, tau_raw, tau_fallback = _resolve_warmup_tau(
            tau_epoch,
            float(args.maol_tau_floor),
            global_step,
        )
        if tau_fallback:
            tau_fallback_epochs += 1
        head.tau.fill_(1.0 if global_step < MAOL_WARMUP_TAU_STEPS else tau_warm)
        model.train()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_count = 0
        for batch_index, (x_cpu, y_cpu) in enumerate(train_loader):
            x, y = _move_batch(x_cpu, y_cpu, device)
            _set_optimizer_lr(optimizer, warmup_lr)
            optimizer.zero_grad(set_to_none=True)
            if global_step >= MAOL_WARMUP_TAU_STEPS and abs(float(head.tau) - tau_warm) > 1e-9:
                head.tau.fill_(tau_warm)
            with _autocast(device, amp_dtype):
                h = model.forward_embedding(x)
                logits = head.logits(h)
                if class_weights is None:
                    loss = F.cross_entropy(logits, y)
                else:
                    per_sample = F.cross_entropy(logits, y, reduction="none")
                    weights = class_weights.to(device=logits.device, dtype=logits.dtype)[y]
                    loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-12)
            _backward_and_step(
                loss,
                optimizer,
                model,
                grad_clip=float(args.grad_clip),
                scaler=scaler,
            )
            global_step += 1
            epoch_loss += float(loss.item())
            batch_correct, batch_count = _accuracy(logits, y)
            epoch_correct += batch_correct
            epoch_count += batch_count

        val = evaluate_warmup_ce(model, val_loader, device, class_weights=class_weights, amp_dtype=amp_dtype)
        row = {
            "epoch": epoch + 1,
            "loss_ce": epoch_loss / max(1, num_batches),
            "train_acc": epoch_correct / max(1, epoch_count),
            "val_loss_ce": val["loss_ce"],
            "val_acc": val["acc"],
            "tau_raw": float(tau_raw),
            "tau_used": float(head.tau.detach().cpu().item()),
            "tau_fallback": bool(tau_fallback),
            "tau_fallback_epochs": int(tau_fallback_epochs),
            "warmup_lr": float(warmup_lr),
            "global_step": global_step,
        }
        rows.append(row)
        print(
            f"[warmup {epoch + 1}/{args.maol_warmup_epochs}] "
            f"loss_ce={row['loss_ce']:.4f} train_acc={row['train_acc']:.3f} "
            f"val_acc={row['val_acc']:.3f} tau_raw={row['tau_raw']:.4g} "
            f"tau_used={row['tau_used']:.4g}"
            + (" FALLBACK" if row["tau_fallback"] else "")
        )

    model.eval()
    stats = compute_warmup_statistics(
        model.forward_embedding,
        head,
        clean_train_loader,
        val_loader,
        device,
        model.tree,
        margin_scale=float(args.maol_margin_scale),
        mbase_scale=float(args.maol_mbase_scale),
        mmax_scale=float(args.maol_mmax_scale),
        n_min=int(args.maol_nmin),
    )
    apply_warmup_stats(head, stats)
    head.set_trainable(True)
    return stats, rows


@torch.no_grad()
def evaluate_linear_loader(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    device: torch.device,
    *,
    class_weights: torch.Tensor | None,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    """CE-only evaluation for the no-MAOL linear control."""
    if loader is None or len(cast(Sized, loader.dataset)) == 0:
        return _empty_metrics()
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    for x_cpu, y_cpu in loader:
        x, y = _move_batch(x_cpu, y_cpu, device)
        with _autocast(device, amp_dtype):
            logits = cast(TorchCNN, model)(x)
            loss = F.cross_entropy(logits, y, weight=class_weights)
        total_loss += float(loss.item())
        batch_correct, batch_count = _accuracy(logits, y)
        correct += batch_correct
        count += batch_count
    batches = max(1, len(loader))
    mean_loss = total_loss / batches
    return {
        "loss_total": mean_loss,
        "loss_ce": mean_loss,
        "loss_intra": 0.0,
        "loss_sep": 0.0,
        "loss_nest": 0.0,
        "loss_cov": 0.0,
        "loss_center": 0.0,
        "acc": correct / max(1, count),
        "gradient_norm": float("nan"),
    }


def _checkpoint_metadata(
    model: TorchCNN,
    stats: WarmupStats,
    args: argparse.Namespace,
    *,
    is_ema: bool,
    best_epoch: int,
) -> dict[str, Any]:
    return {
        "stage": "maol",
        "maol_mode": "full",
        "is_ema": bool(is_ema),
        "ema_decay": float(args.ema_decay) if is_ema else None,
        "best_epoch": int(best_epoch),
        "lambdas": {
            "intra": float(args.maol_lambda_intra),
            "sep": float(args.maol_lambda_sep),
            "nest": float(args.maol_lambda_nest),
            "cov": float(args.maol_lambda_cov),
            "center": float(args.maol_lambda_center),
        },
        "warmup_stats": stats.to_metadata(),
        "maol": {
            "gamma": float(args.maol_gamma),
            "eta": float(args.maol_eta),
            "margin_scale": float(args.maol_margin_scale),
            "mbase_scale": float(args.maol_mbase_scale),
            "mmax_scale": float(args.maol_mmax_scale),
        },
    }


def _linear_checkpoint_metadata(
    args: argparse.Namespace,
    *,
    is_ema: bool,
    best_epoch: int,
) -> dict[str, Any]:
    return {
        "stage": "ce",
        "theory": "CE",
        "maol_mode": "none",
        "maol": False,
        "is_ema": bool(is_ema),
        "ema_decay": float(args.ema_decay) if is_ema else None,
        "best_epoch": int(best_epoch),
        "class_weighting": bool(args.class_weighting),
    }


def _lambdas(args: argparse.Namespace) -> dict[str, float]:
    return {
        "intra": float(args.maol_lambda_intra),
        "sep": float(args.maol_lambda_sep),
        "nest": float(args.maol_lambda_nest),
        "cov": float(args.maol_lambda_cov),
        "center": float(args.maol_lambda_center),
    }


def _save_warmup(
    model: TorchCNN,
    stats: WarmupStats,
    path: Path,
    args: argparse.Namespace,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> None:
    model.save_weights(
        path,
        metadata={
            "stage": "maol_warmup",
            "is_ema": False,
            "warmup_stats": stats.to_metadata(),
            "warmup_diagnostics": dict(diagnostics or {}),
            "lambdas": DEFAULT_LAMBDAS,
            "maol": {
                "gamma": float(args.maol_gamma),
                "eta": float(args.maol_eta),
                "margin_scale": float(args.maol_margin_scale),
                "mbase_scale": float(args.maol_mbase_scale),
                "mmax_scale": float(args.maol_mmax_scale),
            },
        },
    )


def _load_warmup(
    path: Path,
    index: DatasetIndex,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TorchCNN, WarmupStats]:
    config_info = resolve_checkpoint_runtime_config(path, map_location=device)
    if config_info.tree is None:
        raise SystemExit(f"{path} is not a MAOL warm-up checkpoint (missing tree metadata)")
    if config_info.head_type != "prototype" or config_info.embedding_norm != "l2":
        raise SystemExit(
            f"{path} predates the L2-normalized MAOL head; retrain it with the current code"
        )
    if tuple(config_info.class_names) != tuple(index.class_names):
        raise SystemExit("Warm-up checkpoint class order does not match the current nested dataset")
    if config_info.tree.to_dict() != index.tree.to_dict():
        raise SystemExit("Warm-up checkpoint tree does not match the current nested dataset")
    raw_stats = config_info.metadata.get("warmup_stats")
    if not isinstance(raw_stats, Mapping):
        raise SystemExit(f"{path} is missing warm-up statistics")
    stats = WarmupStats.from_metadata(raw_stats)
    if tuple(stats.c_ema.shape) != (index.tree.num_leaves, config_info.embedding_dim):
        raise SystemExit("Warm-up statistics do not match the checkpoint embedding dimension")
    model = TorchCNN(
        input_size=config_info.input_size,
        num_classes=config_info.num_classes,
        tree=index.tree,
        seed=args.seed,
        width_scale=config_info.width_scale,
        embedding_dim=config_info.embedding_dim,
        node_weights=index.node_weights,
    )
    model.load_weights(path, map_location=device, strict=True)
    apply_warmup_stats(model.head, stats)
    model.head.set_trainable(True)
    _move_module_to_device(model, device)
    return model, stats


def _probe(
    model: TorchCNN,
    stats: WarmupStats,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
) -> Path:
    """Run a short scale probe.  No model weights are saved."""
    del stats
    model.head.set_trainable(True)
    no_update = bool(args.maol_probe_no_update)
    if no_update:
        model.eval()
    else:
        model.train()
    optimizer = _build_optimizer(args, model, lr_value=float(args.lr[0]))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    group_params = [(name, params) for name, params in _parameter_groups(model).items() if params]
    group_parameters = [parameter for _, params in group_params for parameter in params]

    rows_path = run_dir / "probe_metrics.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text("", encoding="utf-8")
    iterator = iter(train_loader)
    rows: list[dict[str, Any]] = []
    for step in range(int(args.maol_probe_steps)):
        try:
            x_cpu, y_cpu = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x_cpu, y_cpu = next(iterator)
        x, y = _move_batch(x_cpu, y_cpu, device)
        optimizer.zero_grad(set_to_none=True)
        h = model.forward_embedding(x)
        if not no_update:
            model.head.update_ema_stats(
                h.detach(),
                y,
                m=0.9,
                m_q=float(args.maol_mq),
                n_min=int(args.maol_nmin),
            )
        terms = compute_maol_losses(
            model.head,
            h,
            y,
            lambdas=DEFAULT_LAMBDAS,
            gamma=float(args.maol_gamma),
            eta=float(args.maol_eta),
            class_weights=None,
        )
        terms.total.backward(retain_graph=True)
        total_grad_norm = _grad_norm(parameters)
        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=float(args.grad_clip))
        total_grad_norm_postclip = _grad_norm(parameters)

        grad_h: dict[str, float] = {}
        grad_param: dict[str, dict[str, float]] = {}
        for name, term in terms.named().items():
            if name == "total":
                continue
            try:
                gradients = torch.autograd.grad(
                    term,
                    (h, *group_parameters),
                    retain_graph=True,
                    allow_unused=True,
                )
            except RuntimeError:
                gradients = None
            h_gradient = gradients[0] if gradients is not None else None
            grad_h[name] = (
                float(torch.linalg.vector_norm(h_gradient).item())
                if h_gradient is not None
                else 0.0
            )
            if args.maol_probe_grad_params:
                grad_param[name] = {}
                offset = 0
                for group_name, params in group_params:
                    group_norm = 0.0
                    if gradients is not None:
                        group_grads = gradients[1 + offset : 1 + offset + len(params)]
                        norms = [
                            torch.linalg.vector_norm(gradient.detach().float(), ord=2)
                            for gradient in group_grads
                            if gradient is not None
                        ]
                        if norms:
                            group_norm = float(torch.linalg.vector_norm(torch.stack(norms), ord=2).item())
                    grad_param[name][group_name] = group_norm
                    offset += len(params)

        if not no_update:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        row = {
            "step": step,
            "loss": {
                "CE": float(terms.ce.item()),
                "intra": float(terms.intra.item()),
                "sep": float(terms.sep.item()),
                "nest": float(terms.nest.item()),
                "cov": float(terms.cov.item()),
                "center": float(terms.center.item()),
            },
            "grad_h": {key.upper() if key == "ce" else key: value for key, value in grad_h.items()},
            "grad_param": {
                key.upper() if key == "ce" else key: value
                for key, value in grad_param.items()
            },
            "total_grad_norm": float(total_grad_norm),
            "total_grad_norm_postclip": float(total_grad_norm_postclip),
            "tau": float(model.head.tau.detach().cpu().item()),
            "lr": float(args.lr[0]),
            "no_update": bool(no_update),
        }
        rows.append(row)
        _append_jsonl(rows_path, row)

    loss_keys = ("CE", "intra", "sep", "nest", "cov", "center")
    loss_stats = {key: _summarize_values([row["loss"][key] for row in rows]) for key in loss_keys}
    grad_h_stats = {key: _summarize_values([row["grad_h"][key] for row in rows]) for key in loss_keys}
    total_stats = _summarize_values([row["total_grad_norm"] for row in rows])
    total_post_stats = _summarize_values([row["total_grad_norm_postclip"] for row in rows])
    grad_param_stats: dict[str, dict[str, dict[str, float]]] = {}
    if args.maol_probe_grad_params and rows:
        for term_name in loss_keys:
            grad_param_stats[term_name] = {}
            for group_name, _ in group_params:
                grad_param_stats[term_name][group_name] = _summarize_values(
                    [row["grad_param"].get(term_name, {}).get(group_name, 0.0) for row in rows]
                )

    summary = {
        "steps": len(rows),
        "no_update": bool(no_update),
        "grad_clip": float(args.grad_clip),
        "loss_mean": {key: float(np.mean([row["loss"][key] for row in rows])) if rows else float("nan") for key in loss_keys},
        "grad_h_mean": {key: float(np.mean([row["grad_h"][key] for row in rows])) if rows else float("nan") for key in loss_keys},
        "total_grad_norm_mean": float(total_stats["mean"]),
        "total_grad_norm_postclip_mean": float(total_post_stats["mean"]),
        "loss_stats": loss_stats,
        "grad_h_stats": grad_h_stats,
        "grad_param_stats": grad_param_stats,
        "total_grad_norm_stats": total_stats,
        "total_grad_norm_postclip_stats": total_post_stats,
        "tau": float(model.head.tau.detach().cpu().item()),
        "lr": float(args.lr[0]),
        "note": "Scale probe only; no model weights were written.",
    }
    summary_path = run_dir / "probe_summary.json"
    _write_json(summary_path, summary)
    print(f"Probe finished: {len(rows)} steps -> {rows_path}")
    print(f"Probe summary: {summary_path}")
    return summary_path


def _build_plotter(metrics_path: Path, args: argparse.Namespace):
    try:
        from plot import EpochMetricsPlotter, resolve_plot_output_path
    except Exception as exc:
        raise SystemExit(f"Failed to initialize metric plotting: {exc}") from exc
    plotter = EpochMetricsPlotter(
        metrics_path=metrics_path,
        output_path=resolve_plot_output_path(
            metrics_path,
            output_dir=Path(args.plot_output_dir) if args.plot_output_dir else None,
            output_format=args.plot_output_format,
        ),
    )
    if args.plot_real_time:
        if plotter.interactive:
            print(f"[plot] live metric window enabled (matplotlib backend: {plotter.backend}).")
        else:
            print(
                f"[plot] matplotlib backend '{plotter.backend}' cannot open a window; "
                f"real-time updates stay in memory and the final plot will be saved to {plotter.output_path}"
            )
    return plotter


def _finalize_plotting(
    plotter: Any | None,
    metric_rows: list[dict[str, Any]],
    metrics_path: Path,
    args: argparse.Namespace,
) -> None:
    """Always write the final plot; show a window only when one can exist."""
    if not metric_rows or not (args.plot_once or args.plot_real_time):
        return
    if plotter is None:
        plotter = _build_plotter(metrics_path, args)
    plotter.save(metric_rows)
    print(f"[plot] metrics plot saved to: {plotter.output_path}")
    if args.plot_once:
        if plotter.interactive:
            plotter.show(block=True)
        else:
            print(
                f"[plot] matplotlib backend '{plotter.backend}' cannot show a GUI window; "
                f"open the saved image instead: {plotter.output_path}"
            )


def _run_training(
    model: TorchCNN,
    stats: WarmupStats,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    *,
    class_weights: torch.Tensor | None,
) -> dict[str, Any]:
    lambdas = _lambdas(args)
    metrics_path = run_dir / "epoch_metrics.jsonl"
    summary_path = run_dir / "summary.json"
    if metrics_path.exists():
        metrics_path.unlink()
    plotter = _build_plotter(metrics_path, args) if args.plot_real_time else None
    metric_rows: list[dict[str, Any]] = []

    ema: ModelEMA | None = None
    if args.ema:
        ema_model = TorchCNN(
            input_size=model.input_size,
            num_classes=model.num_classes,
            tree=model.tree,
            seed=args.seed,
            width_scale=model.width_scale,
            embedding_dim=model.embedding_dim,
            node_weights=stats.node_weights.tolist(),
        )
        _move_module_to_device(ema_model, device)
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        ema_model.eval()
        ema = ModelEMA(
            ema_model,
            decay=float(args.ema_decay),
            phase_warmup_steps=max(1, len(train_loader)),
            skip_prefixes=(
                "head.c_ema",
                "head.q",
                "head.tau",
                "head.node_weights",
                "head.margins",
                "head.sep_weights",
            ),
        )
        ema.sync_from(model)

    optimizer = _build_optimizer(args, model, lr_value=float(args.lr[0]))
    amp_dtype = _resolve_amp_dtype(device, args.amp_mode)
    scaler = _make_grad_scaler(device, amp_dtype)
    phase_map = build_epoch_phase_map(int(args.epochs), int(args.phase_count))
    current_phase = -1
    phase_base_lr = float(args.lr[0])
    phase_offset = 0.0
    phase_best_acc = -float("inf")
    plateau_epochs = 0
    freeze_remaining = 0
    backbone_frozen = False

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    final_metrics: dict[str, Any] = {}
    num_batches = max(1, len(train_loader))
    current_lr = phase_base_lr

    for epoch in range(int(args.epochs)):
        phase = phase_map[epoch]
        if phase.phase_index != current_phase:
            current_phase = phase.phase_index
            phase_base_lr = float(args.lr[current_phase])
            phase_offset = 0.0
            phase_best_acc = -float("inf")
            plateau_epochs = 0
            freeze_remaining = 0
            backbone_frozen = False
            model.train()
            _apply_backbone_freeze_state(model, False, bool(args.freeze_bn_affine))
            optimizer = _build_optimizer(args, model, lr_value=phase_base_lr)
            scaler = _make_grad_scaler(device, amp_dtype)
            print(f"[phase {current_phase + 1}/{args.phase_count}] base_lr={phase_base_lr:.6f}")

        model.train()
        epoch_totals = {key: 0.0 for key in ("total", "ce", "intra", "sep", "nest", "cov", "center")}
        epoch_correct = 0
        epoch_count = 0
        grad_norm_total = 0.0
        grad_norm_count = 0
        hidden_norm_total = 0.0
        hidden_norm_count = 0

        for batch_index, (x_cpu, y_cpu) in enumerate(train_loader):
            x, y = _move_batch(x_cpu, y_cpu, device)
            scheduled_lr = compute_phase_learning_rate(
                base_lr=phase_base_lr,
                schedule=args.lr_schedule,
                min_lr_ratio=float(args.min_lr_ratio),
                gamma=float(args.gamma),
                step_size=int(args.step_size),
                warmup_epochs=int(args.warmup_epochs),
                epoch_index_in_phase=phase.epoch_index_in_phase,
                epochs_in_phase=phase.epochs_in_phase,
                batch_index=batch_index,
                num_batches=num_batches,
            )
            current_lr = compute_effective_learning_rate(
                scheduled_lr,
                phase_offset,
                phase_base_lr,
                float(args.min_lr_ratio),
            )
            _set_optimizer_lr(optimizer, current_lr)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp_dtype):
                h = model.forward_embedding(x)
                model.head.update_ema_stats(
                    h.detach(),
                    y,
                    m=0.9,
                    m_q=float(args.maol_mq),
                    n_min=int(args.maol_nmin),
                )
                terms = compute_maol_losses(
                    model.head,
                    h,
                    y,
                    lambdas=lambdas,
                    gamma=float(args.maol_gamma),
                    eta=float(args.maol_eta),
                    class_weights=class_weights,
                )
            gradient_norm = _backward_and_step(
                terms.total,
                optimizer,
                model,
                grad_clip=float(args.grad_clip),
                scaler=scaler,
            )
            if not np.isnan(gradient_norm):
                grad_norm_total += gradient_norm
                grad_norm_count += 1
            hidden_norm_total += float(torch.linalg.vector_norm(h.detach().float(), dim=-1).mean().item())
            hidden_norm_count += 1
            if ema is not None:
                ema.update(model, step_in_phase=phase.epoch_index_in_phase * num_batches + batch_index)
            for key, value in (
                ("total", terms.total),
                ("ce", terms.ce),
                ("intra", terms.intra),
                ("sep", terms.sep),
                ("nest", terms.nest),
                ("cov", terms.cov),
                ("center", terms.center),
            ):
                epoch_totals[key] += float(value.item())
            batch_correct, batch_count = _accuracy(terms.logits, y)
            epoch_correct += batch_correct
            epoch_count += batch_count

        train_acc = epoch_correct / max(1, epoch_count)
        avg = {key: value / num_batches for key, value in epoch_totals.items()}
        eval_model = ema.ema_model if ema is not None else model
        val = evaluate_loader(
            eval_model,
            val_loader,
            device,
            lambdas=lambdas,
            gamma=float(args.maol_gamma),
            eta=float(args.maol_eta),
            class_weights=class_weights,
            amp_dtype=amp_dtype,
        )
        val_acc = val["acc"]
        val_loss = val["loss_total"]
        previous_best_acc = best_val_acc
        if not np.isnan(val_acc):
            best_val_acc = max(best_val_acc, val_acc)
        improved = (
            val_loss < (best_val_loss - float(args.min_delta))
            if args.early_stop_metric == "val_loss"
            else val_acc > (previous_best_acc + float(args.min_delta))
        )
        if improved:
            best_epoch = epoch + 1
            bad_epochs = 0
            checkpoint_model = eval_model
            checkpoint_model.save_weights(
                Path(args.checkpoint),
                metadata=_checkpoint_metadata(model, stats, args, is_ema=ema is not None, best_epoch=best_epoch),
            )
        else:
            bad_epochs += 1
        best_val_loss = min(best_val_loss, val_loss)
        best_val_acc = max(best_val_acc, val_acc)

        if args.freeze_enabled and val_loader is not None:
            if backbone_frozen:
                if val_acc > phase_best_acc + float(args.min_delta):
                    phase_best_acc = val_acc
                freeze_remaining -= 1
                if freeze_remaining <= 0:
                    backbone_frozen = False
                    _apply_backbone_freeze_state(model, False, bool(args.freeze_bn_affine))
                    optimizer = _build_optimizer(args, model, lr_value=current_lr)
                    scaler = _make_grad_scaler(device, amp_dtype)
            else:
                if val_acc > phase_best_acc + float(args.min_delta):
                    phase_best_acc = val_acc
                    plateau_epochs = 0
                else:
                    plateau_epochs += 1
                    if plateau_epochs >= int(args.freeze_patience):
                        backbone_frozen = True
                        freeze_remaining = int(args.freeze_epoch_num)
                        plateau_epochs = 0
                        _apply_backbone_freeze_state(model, True, bool(args.freeze_bn_affine))
                        optimizer = _build_optimizer(args, model, lr_value=current_lr)
                        scaler = _make_grad_scaler(device, amp_dtype)

        gap = train_acc - val_acc
        final_metrics = {
            "epoch": epoch + 1,
            "phase": current_phase + 1,
            "lr": float(current_lr),
            "train_loss_total": float(avg["total"]),
            "train_loss_ce": float(avg["ce"]),
            "train_loss_intra": float(avg["intra"]),
            "train_loss_sep": float(avg["sep"]),
            "train_loss_nest": float(avg["nest"]),
            "train_loss_cov": float(avg["cov"]),
            "train_loss_center": float(avg["center"]),
            "train_acc": float(train_acc),
            "val_loss_total": float(val_loss),
            "val_loss_ce": float(val["loss_ce"]),
            "val_loss_intra": float(val["loss_intra"]),
            "val_loss_sep": float(val["loss_sep"]),
            "val_loss_nest": float(val["loss_nest"]),
            "val_loss_cov": float(val["loss_cov"]),
            "val_loss_center": float(val["loss_center"]),
            "val_acc": float(val_acc),
            "generalization_gap": float(gap),
            "gradient_norm": float(grad_norm_total / max(1, grad_norm_count)),
            "hidden_norm": float(hidden_norm_total / max(1, hidden_norm_count)),
            "tau": float(model.head.tau.detach().cpu().item()),
            "d_intra": float(stats.d_intra),
            "d_inter": float(stats.d_inter),
            "margin_cap": float(stats.margin_cap),
            "mean_leaf_radius": float(model.head.radii.mean().detach().cpu().item()),
            "mean_leaf_center_norm": float(torch.linalg.vector_norm(model.head.centers, dim=-1).mean().item()),
            "lambda_intra": float(lambdas["intra"]),
            "lambda_sep": float(lambdas["sep"]),
            "lambda_nest": float(lambdas["nest"]),
            "lambda_cov": float(lambdas["cov"]),
            "lambda_center": float(lambdas["center"]),
            "improved": bool(improved),
            "backbone_frozen": bool(backbone_frozen),
        }
        metric_rows.append(final_metrics)
        _append_jsonl(metrics_path, final_metrics)
        if plotter is not None:
            plotter.update(metric_rows)
        print(
            f"[epoch {epoch + 1}/{args.epochs}] lr={current_lr:.6f} "
            f"train_loss={avg['total']:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} gap={gap:.3f}"
            + (" *" if improved else "")
        )
        if args.early_stop and bad_epochs >= int(args.patience):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if val_loader is None or len(cast(Sized, val_loader.dataset)) == 0:
        model.save_weights(
            Path(args.checkpoint),
            metadata=_checkpoint_metadata(model, stats, args, is_ema=False, best_epoch=best_epoch),
        )

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "maol_mode": "full",
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "final_metrics": final_metrics,
        "warmup_stats": stats.to_metadata(),
        "lambdas": lambdas,
        "class_names": list(model.class_names),
        "tree": model.tree.to_dict(),
        "note": "MAOL run; MixUp, CutMix, focal loss and label smoothing are not used.",
    }
    _write_json(summary_path, summary)
    _finalize_plotting(plotter, metric_rows, metrics_path, args)
    return summary


def _run_linear_training(
    model: TorchCNN,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    *,
    class_weights: torch.Tensor | None,
) -> dict[str, Any]:
    """CE-only linear-head control (no wells, no geometry losses, no L2)."""
    metrics_path = run_dir / "epoch_metrics.jsonl"
    summary_path = run_dir / "summary.json"
    if metrics_path.exists():
        metrics_path.unlink()
    plotter = _build_plotter(metrics_path, args) if args.plot_real_time else None
    metric_rows: list[dict[str, Any]] = []

    ema: ModelEMA | None = None
    if args.ema:
        ema_model = TorchCNN(
            input_size=model.input_size,
            num_classes=model.num_classes,
            tree=model.tree,
            maol=False,
            class_names=model.class_names,
            seed=args.seed,
            width_scale=model.width_scale,
            embedding_dim=model.embedding_dim,
        )
        _move_module_to_device(ema_model, device)
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        ema_model.eval()
        ema = ModelEMA(
            ema_model,
            decay=float(args.ema_decay),
            phase_warmup_steps=max(1, len(train_loader)),
            skip_prefixes=(),
        )
        ema.sync_from(model)

    optimizer = _build_optimizer(args, model, lr_value=float(args.lr[0]))
    amp_dtype = _resolve_amp_dtype(device, args.amp_mode)
    scaler = _make_grad_scaler(device, amp_dtype)
    phase_map = build_epoch_phase_map(int(args.epochs), int(args.phase_count))
    current_phase = -1
    phase_base_lr = float(args.lr[0])
    phase_offset = 0.0
    current_lr = phase_base_lr

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    final_metrics: dict[str, Any] = {}
    num_batches = max(1, len(train_loader))

    for epoch in range(int(args.epochs)):
        phase = phase_map[epoch]
        if phase.phase_index != current_phase:
            current_phase = phase.phase_index
            phase_base_lr = float(args.lr[current_phase])
            phase_offset = 0.0
            model.train()
            optimizer = _build_optimizer(args, model, lr_value=phase_base_lr)
            scaler = _make_grad_scaler(device, amp_dtype)
            print(f"[phase {current_phase + 1}/{args.phase_count}] base_lr={phase_base_lr:.6f}")

        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_count = 0
        grad_norm_total = 0.0
        grad_norm_count = 0
        hidden_norm_total = 0.0
        hidden_norm_count = 0

        for batch_index, (x_cpu, y_cpu) in enumerate(train_loader):
            x, y = _move_batch(x_cpu, y_cpu, device)
            scheduled_lr = compute_phase_learning_rate(
                base_lr=phase_base_lr,
                schedule=args.lr_schedule,
                min_lr_ratio=float(args.min_lr_ratio),
                gamma=float(args.gamma),
                step_size=int(args.step_size),
                warmup_epochs=int(args.warmup_epochs),
                epoch_index_in_phase=phase.epoch_index_in_phase,
                epochs_in_phase=phase.epochs_in_phase,
                batch_index=batch_index,
                num_batches=num_batches,
            )
            current_lr = compute_effective_learning_rate(
                scheduled_lr,
                phase_offset,
                phase_base_lr,
                float(args.min_lr_ratio),
            )
            _set_optimizer_lr(optimizer, current_lr)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp_dtype):
                h = model.forward_embedding(x)
                logits = model.fc2(h)
                loss = F.cross_entropy(logits, y, weight=class_weights)
            gradient_norm = _backward_and_step(
                loss,
                optimizer,
                model,
                grad_clip=float(args.grad_clip),
                scaler=scaler,
            )
            if not np.isnan(gradient_norm):
                grad_norm_total += gradient_norm
                grad_norm_count += 1
            hidden_norm_total += float(torch.linalg.vector_norm(h.detach().float(), dim=-1).mean().item())
            hidden_norm_count += 1
            if ema is not None:
                ema.update(model, step_in_phase=phase.epoch_index_in_phase * num_batches + batch_index)

            epoch_loss += float(loss.item())
            batch_correct, batch_count = _accuracy(logits, y)
            epoch_correct += batch_correct
            epoch_count += batch_count

        train_acc = epoch_correct / max(1, epoch_count)
        train_loss = epoch_loss / num_batches
        eval_model = ema.ema_model if ema is not None else model
        val = evaluate_linear_loader(
            eval_model,
            val_loader,
            device,
            class_weights=class_weights,
            amp_dtype=amp_dtype,
        )
        val_acc = val["acc"]
        val_loss = val["loss_total"]
        previous_best_acc = best_val_acc
        if not np.isnan(val_acc):
            best_val_acc = max(best_val_acc, val_acc)
        improved = (
            val_loss < (best_val_loss - float(args.min_delta))
            if args.early_stop_metric == "val_loss"
            else val_acc > (previous_best_acc + float(args.min_delta))
        )
        if improved:
            best_epoch = epoch + 1
            bad_epochs = 0
            eval_model.save_weights(
                Path(args.checkpoint),
                metadata=_linear_checkpoint_metadata(
                    args,
                    is_ema=ema is not None,
                    best_epoch=best_epoch,
                ),
            )
        else:
            bad_epochs += 1
        best_val_loss = min(best_val_loss, val_loss)
        best_val_acc = max(best_val_acc, val_acc)

        gap = train_acc - val_acc
        final_metrics = {
            "epoch": epoch + 1,
            "phase": current_phase + 1,
            "maol_mode": "none",
            "lr": float(current_lr),
            "train_loss_total": float(train_loss),
            "train_loss_ce": float(train_loss),
            "train_loss_intra": 0.0,
            "train_loss_sep": 0.0,
            "train_loss_nest": 0.0,
            "train_loss_cov": 0.0,
            "train_loss_center": 0.0,
            "train_acc": float(train_acc),
            "val_loss_total": float(val_loss),
            "val_loss_ce": float(val_loss),
            "val_loss_intra": 0.0,
            "val_loss_sep": 0.0,
            "val_loss_nest": 0.0,
            "val_loss_cov": 0.0,
            "val_loss_center": 0.0,
            "val_acc": float(val_acc),
            "generalization_gap": float(gap),
            "gradient_norm": float(grad_norm_total / max(1, grad_norm_count)),
            "hidden_norm": float(hidden_norm_total / max(1, hidden_norm_count)),
            "tau": 0.0,
            "d_intra": 0.0,
            "d_inter": 0.0,
            "margin_cap": 0.0,
            "mean_leaf_radius": 0.0,
            "mean_leaf_center_norm": 0.0,
            "lambda_intra": 0.0,
            "lambda_sep": 0.0,
            "lambda_nest": 0.0,
            "lambda_cov": 0.0,
            "lambda_center": 0.0,
            "improved": bool(improved),
            "backbone_frozen": False,
        }
        metric_rows.append(final_metrics)
        _append_jsonl(metrics_path, final_metrics)
        if plotter is not None:
            plotter.update(metric_rows)
        print(
            f"[epoch {epoch + 1}/{args.epochs}] lr={current_lr:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} gap={gap:.3f}"
            + (" *" if improved else "")
        )
        if args.early_stop and bad_epochs >= int(args.patience):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if val_loader is None or len(cast(Sized, val_loader.dataset)) == 0:
        model.save_weights(
            Path(args.checkpoint),
            metadata=_linear_checkpoint_metadata(args, is_ema=False, best_epoch=best_epoch),
        )

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "maol_mode": "none",
        "head_type": "linear",
        "ema": bool(args.ema),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "final_metrics": final_metrics,
        "warmup_stats": None,
        "lambdas": {key: 0.0 for key in DEFAULT_LAMBDAS},
        "class_names": list(model.class_names),
        "tree": model.tree.to_dict() if model.tree is not None else None,
        "note": "CE linear-head control; no MAOL wells or geometry losses.",
    }
    _write_json(summary_path, summary)
    _finalize_plotting(plotter, metric_rows, metrics_path, args)
    return summary


def _resolve_ema_args(args: argparse.Namespace) -> None:
    """Normalize ``--ema``/``--no-ema``/``--ema-decay`` into a bool + decay.

    ``--ema`` without a value uses ``--ema-decay`` (default 0.9993).
    ``--ema 0.999`` explicitly sets the decay; ``--no-ema`` disables EMA.
    """
    raw = getattr(args, "ema", None)
    if raw is None or raw is False:
        args.ema = False
    elif raw is True:
        args.ema = True
    else:
        args.ema = True
        args.ema_decay = float(raw)
    try:
        args.ema_decay = validate_ema_decay(args.ema_decay)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0")
    if not (0.0 <= args.val_split < 1.0):
        raise SystemExit("--val-split must satisfy 0 <= value < 1")
    if args.maol_mode == "none":
        if args.maol_probe_only:
            raise SystemExit("--maol-probe-only is MAOL-only; use --maol-mode full")
        if args.maol_warmup_only:
            raise SystemExit("--maol-warmup-only is MAOL-only; use --maol-mode full")
        if args.maol_force_warmup:
            raise SystemExit("--maol-force-warmup is MAOL-only; use --maol-mode full")
    elif args.maol_warmup_epochs < 5:
        raise SystemExit("--maol-warmup-epochs must be >= 5")
    if args.maol_probe_steps <= 0:
        raise SystemExit("--maol-probe-steps must be > 0")
    if args.maol_probe_only and args.maol_warmup_only:
        raise SystemExit("--maol-probe-only and --maol-warmup-only are mutually exclusive")
    if args.gamma <= 0.0:
        raise SystemExit("--gamma must be > 0")
    if not (0.0 < args.maol_gamma < 1.0):
        raise SystemExit("--maol-gamma must satisfy 0 < value < 1")
    if args.maol_eta < 0.0:
        raise SystemExit("--maol-eta must be >= 0")
    if not (0.0 <= args.maol_mq < 1.0):
        raise SystemExit("--maol-mq must satisfy 0 <= value < 1")
    if args.maol_nmin < 1:
        raise SystemExit("--maol-nmin must be >= 1")
    if args.maol_tau_floor <= 0.0:
        raise SystemExit("--maol-tau-floor must be > 0")
    if args.maol_warmup_min_acc_factor <= 0.0:
        raise SystemExit("--maol-warmup-min-acc-factor must be > 0")
    if args.maol_warmup_max_q_ratio <= 0.0:
        raise SystemExit("--maol-warmup-max-q-ratio must be > 0")
    if args.maol_warmup_min_d_inter_ratio <= 0.0:
        raise SystemExit("--maol-warmup-min-d-inter-ratio must be > 0")
    if args.maol_warmup_max_tau0 <= 0.0:
        raise SystemExit("--maol-warmup-max-tau0 must be > 0")
    if args.maol_warmup_lr is not None and args.maol_warmup_lr <= 0.0:
        raise SystemExit("--maol-warmup-lr must be > 0 when provided")
    for name in (
        "maol_margin_scale",
        "maol_mbase_scale",
        "maol_mmax_scale",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be > 0")
    for name in ("maol_lambda_intra", "maol_lambda_sep", "maol_lambda_nest", "maol_lambda_cov", "maol_lambda_center"):
        if float(getattr(args, name)) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 0")
    if args.phase_count > args.epochs:
        raise SystemExit("--phase-count cannot be greater than --epochs")
    try:
        validate_augmentation_args(args.rotation, args.brightness, args.contrast, args.saturation)
        validate_ema_decay(args.ema_decay)
        validate_model_width_scale(args.model_width_scale)
        validate_freeze_cycle_args(args.freeze_patience, args.freeze_epoch_num, args.after_unfreeze_lr_change)
        validate_phase_learning_rates(args.lr, args.phase_count)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _verify_dataset_unchanged(data_dir: Path, pre_signature, enabled: bool) -> None:
    if enabled and pre_signature is not None and tree_signature(data_dir) != pre_signature:
        raise SystemExit("Dataset changed during run; aborting (read-only enforcement).")


def _choose_partition(args: argparse.Namespace) -> int:
    parts = max(1, int(args.num_partitions))
    if not args.auto_next_partition:
        return max(0, min(int(args.partition), parts - 1))
    state_path = Path(args.partition_state)
    state = {"num_partitions": parts, "next": 0}
    try:
        if state_path.is_file():
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if int(loaded.get("num_partitions", -1)) == parts:
                state = loaded
    except Exception:
        pass
    part = int(state.get("next", 0)) % parts
    state["num_partitions"] = parts
    state["next"] = (part + 1) % parts
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass
    return part


def _partition_index(path: Path, parts: int) -> int:
    digest = hashlib.md5(str(path.resolve()).lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a MAOL CNN classifier")
    parser.add_argument("--data-dir", type=str, default=str(config.DATA_DIR))
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, nargs="+", default=[config.LEARNING_RATE])
    parser.add_argument("--phase-count", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=0, help="LR warm-up epochs inside each phase")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--lr-schedule", choices=["cosine", "step", "constant"], default="cosine")
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--early-stop-metric", choices=["val_loss", "val_acc"], default="val_acc")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--class-weighting", action=argparse.BooleanOptionalAction, default=False)
    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument(
        "--ema",
        nargs="?",
        const=True,
        default=None,
        type=float,
        metavar="DECAY",
        help="Enable EMA; optional decay in (0,1), e.g. --ema 0.999. Bare --ema uses --ema-decay.",
    )
    ema_group.add_argument("--no-ema", dest="ema", action="store_const", const=False, help="Disable EMA")
    parser.add_argument("--ema-decay", type=float, default=0.9993, help="Decay used by bare --ema")
    parser.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "maol_model.pt"))
    parser.add_argument("--json-dir", type=str, default="runs")
    parser.add_argument(
        "--plot-once",
        action="store_true",
        help="Save the metrics plot after training and show a window when the backend supports one",
    )
    parser.add_argument(
        "--plot-real-time",
        action="store_true",
        help="Update a live metrics window during training when the backend supports one; the final plot is always saved",
    )
    parser.add_argument("--plot-output-format", choices=["png", "jpg", "jpeg"], default="png")
    parser.add_argument("--plot-output-dir", type=str, default=None)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rotation", type=float, default=20.0)
    parser.add_argument("--brightness", type=float, default=0.25)
    parser.add_argument("--contrast", type=float, default=0.25)
    parser.add_argument("--saturation", type=float, default=0.25)
    parser.add_argument("--model-width-scale", type=float, default=1.5)
    parser.add_argument("--maol-mode", choices=["full", "none"], default="full", help="full = MAOL wells/geometry; none = CE linear-head control")
    parser.add_argument("--no-maol", dest="maol_mode", action="store_const", const="none", help="Alias for --maol-mode none")
    parser.add_argument("--num-partitions", type=int, default=1)
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--auto-next-partition", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--partition-state", type=str, default=str(config.CHECKPOINT_DIR / "partition_state.json"))
    parser.add_argument("--enforce-readonly-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp-mode", choices=["auto", "on", "off"], default="auto")

    parser.add_argument("--maol-warmup-epochs", type=int, default=15)
    parser.add_argument("--maol-warmup-only", action="store_true")
    parser.add_argument("--maol-force-warmup", action="store_true")
    parser.add_argument("--maol-warmup-ckpt", type=str, default=None)
    parser.add_argument("--maol-warmup-lr", type=float, default=None, help="Override warm-up LR; default is lr[0] from a pretrained init, 2*lr[0] from scratch")
    parser.add_argument("--maol-warmup-augment", action=argparse.BooleanOptionalAction, default=False, help="Use RandAugment during warm-up (default: off)")
    parser.add_argument("--maol-tau-floor", type=float, default=1e-2, help="If tau_warm falls below this, fall back to tau=1.0")
    parser.add_argument("--maol-warmup-sanity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--maol-warmup-min-acc-factor", type=float, default=2.0, help="Minimum warm-up val_acc as a multiple of 1/num_classes")
    parser.add_argument("--maol-warmup-max-q-ratio", type=float, default=10.0, help="Maximum q_max/q_median allowed after warm-up")
    parser.add_argument("--maol-warmup-min-d-inter-ratio", type=float, default=1.3, help="Minimum d_inter/d_intra allowed after warm-up")
    parser.add_argument("--maol-warmup-max-tau0", type=float, default=10.0, help="Maximum tau0 allowed after warm-up (unit-sphere scale)")
    parser.add_argument("--maol-probe-only", action="store_true")
    parser.add_argument("--maol-probe-steps", type=int, default=100)
    parser.add_argument("--maol-probe-output", type=str, default=None)
    probe_update_group = parser.add_mutually_exclusive_group()
    probe_update_group.add_argument(
        "--maol-probe-no-update",
        dest="maol_probe_no_update",
        action="store_true",
        default=True,
        help="Measure only; never call optimizer.step (default)",
    )
    probe_update_group.add_argument(
        "--maol-probe-update",
        dest="maol_probe_no_update",
        action="store_false",
        help="Allow the probe to update weights (gradients are clipped first)",
    )
    parser.add_argument("--maol-probe-grad-params", action=argparse.BooleanOptionalAction, default=True, help="Record per-term gradients for backbone/fc1/centers/radii")
    parser.add_argument("--maol-lambda-intra", type=float, default=1.0)
    parser.add_argument("--maol-lambda-sep", type=float, default=1.0)
    parser.add_argument("--maol-lambda-nest", type=float, default=1.0)
    parser.add_argument("--maol-lambda-cov", type=float, default=1.0)
    parser.add_argument("--maol-lambda-center", type=float, default=1.0)
    parser.add_argument("--maol-gamma", type=float, default=0.5)
    parser.add_argument("--maol-eta", type=float, default=0.1)
    parser.add_argument("--maol-mq", type=float, default=0.9)
    parser.add_argument("--maol-nmin", type=int, default=5)
    parser.add_argument("--maol-margin-scale", type=float, default=1.25)
    parser.add_argument("--maol-mbase-scale", type=float, default=0.9)
    parser.add_argument("--maol-mmax-scale", type=float, default=0.8)
    parser.add_argument("--freeze-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-patience", type=int, default=8)
    parser.add_argument("--freeze-epoch-num", type=int, default=10)
    parser.add_argument("--after-unfreeze-lr-change", type=float, default=1e-4)
    parser.add_argument(
        "--freeze-bn-affine",
        type=parse_bool_flag,
        nargs="?",
        const=True,
        default=True,
    )
    args = parser.parse_args()
    _resolve_ema_args(args)
    _validate_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = _resolve_device(args.device)
    data_dir = Path(args.data_dir).resolve()
    pre_signature = tree_signature(data_dir) if args.enforce_readonly_dataset else None

    index = discover_nested_dataset(data_dir)
    train_indices, val_indices = stratified_split(index.labels, float(args.val_split), seed=int(args.seed))
    train_paths, y_train, val_paths, y_val = split_dataset(index, train_indices, val_indices)

    num_partitions = max(1, int(args.num_partitions))
    if num_partitions > 1:
        partition = _choose_partition(args)
        keep = [_partition_index(path, num_partitions) == partition for path in train_paths]
        train_paths = [path for path, flag in zip(train_paths, keep) if flag]
        y_train = y_train[np.asarray(keep, dtype=bool)]
        if not train_paths:
            raise SystemExit("Selected partition contains no samples")

    if len(train_paths) == 0:
        raise SystemExit("No training images found")
    class_counts = np.bincount(y_train, minlength=index.tree.num_leaves)
    print(f"Using {len(index.class_names)} classes, {len(train_paths)} train / {len(val_paths)} val")
    print(f"Class counts: {class_counts.tolist()}")
    print(index.tree.describe())

    checkpoint_path = Path(args.checkpoint)
    warmup_ckpt = (
        Path(args.maol_warmup_ckpt)
        if args.maol_warmup_ckpt
        else checkpoint_path.with_name(checkpoint_path.stem + "_warmup.pt")
    )

    augment_config = (
        validate_augmentation_args(args.rotation, args.brightness, args.contrast, args.saturation)
        if args.augment
        else None
    )
    train_dataset = TorchImageDataset(train_paths, y_train, config.INPUT_SIZE, augment_config=augment_config)
    clean_train_dataset = TorchImageDataset(train_paths, y_train, config.INPUT_SIZE, augment_config=None)
    val_dataset = (
        TorchImageDataset(val_paths, y_val, config.INPUT_SIZE, augment_config=None)
        if len(val_paths) > 0
        else None
    )
    pin_memory = device.type == "cuda"
    train_loader = _build_loader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        seed=int(args.seed),
    )
    clean_train_loader = _build_loader(
        clean_train_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        seed=int(args.seed) + 1,
    )
    val_loader = (
        _build_loader(
            val_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=pin_memory,
            seed=int(args.seed) + 2,
        )
        if val_dataset is not None
        else None
    )
    warmup_train_loader = _build_loader(
        train_dataset if args.maol_warmup_augment else clean_train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        seed=int(args.seed) + 3,
    )

    class_weights_tensor = None
    if args.class_weighting:
        class_weights_tensor = torch.as_tensor(
            _class_weights(y_train, index.tree.num_leaves), dtype=torch.float32, device=device
        )

    node_weights = index.tree.node_weights_from_labels(y_train.tolist())
    model = TorchCNN(
        input_size=config.INPUT_SIZE,
        num_classes=index.tree.num_leaves,
        tree=index.tree,
        maol=args.maol_mode != "none",
        class_names=index.class_names,
        seed=int(args.seed),
        width_scale=float(args.model_width_scale),
        embedding_dim=256,
        node_weights=node_weights,
    )
    _move_module_to_device(model, device)
    print(f"Using stage_channels={model.stage_channels} width_scale={model.width_scale:.3f}")
    print(f"MAOL mode: {args.maol_mode}")

    if args.maol_mode == "none":
        run_tag = "ce"
    else:
        run_tag = "maol-probe" if args.maol_probe_only else ("maol-warmup" if args.maol_warmup_only else "maol")
    run_dir = Path(args.maol_probe_output) if args.maol_probe_output else _prepare_run_dir(args, run_tag, index)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.maol_mode == "none":
        summary = _run_linear_training(
            model,
            train_loader,
            val_loader,
            args,
            device,
            run_dir,
            class_weights=class_weights_tensor,
        )
        print(f"Run summary: {run_dir / 'summary.json'}")
        print(f"Best epoch: {summary['best_epoch']}  best_val_acc={summary['best_val_acc']:.4f}")
        if args.enforce_readonly_dataset and pre_signature is not None:
            _verify_dataset_unchanged(data_dir, pre_signature, True)
        return

    if args.maol_probe_only:
        if not warmup_ckpt.is_file():
            raise SystemExit(f"Probe requires a warm-up checkpoint: {warmup_ckpt}")
        model, stats = _load_warmup(warmup_ckpt, index, args=args, device=device)
        loaded_val = evaluate_warmup_ce(
            model,
            val_loader,
            device,
            class_weights=class_weights_tensor,
            amp_dtype=_resolve_amp_dtype(device, args.amp_mode),
        )
        _check_warmup_or_abort(
            stats,
            loaded_val["acc"],
            model.num_classes,
            args,
            run_dir,
        )
        _probe(model, stats, train_loader, args, device, run_dir)
        _verify_dataset_unchanged(data_dir, pre_signature, bool(args.enforce_readonly_dataset))
        return

    if warmup_ckpt.is_file() and not args.maol_warmup_only and not args.maol_force_warmup:
        print(f"Loading warm-up checkpoint: {warmup_ckpt}")
        model, stats = _load_warmup(warmup_ckpt, index, args=args, device=device)
        loaded_val = evaluate_warmup_ce(
            model,
            val_loader,
            device,
            class_weights=class_weights_tensor,
            amp_dtype=_resolve_amp_dtype(device, args.amp_mode),
        )
        _check_warmup_or_abort(
            stats,
            loaded_val["acc"],
            model.num_classes,
            args,
            run_dir,
        )
    else:
        print(f"Running MAOL warm-up for {args.maol_warmup_epochs} epochs")
        print(f"Warm-up augmentation: {bool(args.maol_warmup_augment)}")
        stats, warmup_rows = run_warmup(
            model,
            warmup_train_loader,
            clean_train_loader,
            val_loader,
            args,
            device,
            class_weights=class_weights_tensor,
        )
        _write_json(run_dir / "warmup_stats.json", stats.to_metadata())
        for row in warmup_rows:
            _append_jsonl(run_dir / "warmup_metrics.jsonl", row)
        diagnostics = _check_warmup_or_abort(
            stats,
            float(warmup_rows[-1]["val_acc"]) if warmup_rows else float("nan"),
            model.num_classes,
            args,
            run_dir,
            tau_fallback_epochs=int(warmup_rows[-1].get("tau_fallback_epochs", 0)) if warmup_rows else 0,
        )
        _save_warmup(model, stats, warmup_ckpt, args, diagnostics=diagnostics)
        print(f"Saved warm-up checkpoint: {warmup_ckpt}")
        if args.maol_warmup_only:
            _verify_dataset_unchanged(data_dir, pre_signature, bool(args.enforce_readonly_dataset))
            return

    summary = _run_training(
        model,
        stats,
        train_loader,
        val_loader,
        args,
        device,
        run_dir,
        class_weights=class_weights_tensor,
    )
    print(f"Run summary: {run_dir / 'summary.json'}")
    print(f"Best epoch: {summary['best_epoch']}  best_val_acc={summary['best_val_acc']:.4f}")

    if args.enforce_readonly_dataset and pre_signature is not None:
        _verify_dataset_unchanged(data_dir, pre_signature, True)


if __name__ == "__main__":
    main()
