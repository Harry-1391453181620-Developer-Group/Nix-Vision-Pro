"""MAOL inference backend (CNN prototype head only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from backends.torch.model import TorchCNN, resolve_checkpoint_runtime_config
from data.loaders import load_image
from data.preprocessing import preprocess_image
from utils.safety import install_dataset_write_guard

install_dataset_write_guard()

import config


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device=cuda requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_maol_model(weights_path: str | Path, device: torch.device) -> TorchCNN:
    config_info = resolve_checkpoint_runtime_config(weights_path, map_location=device)
    if config_info.head_type != "prototype" or config_info.tree is None:
        raise SystemExit(
            "This build supports MAOL prototype checkpoints only. "
            "Old MAOIDL checkpoints are intentionally not compatible."
        )
    if config_info.embedding_norm != "l2":
        raise SystemExit(
            "This checkpoint predates the L2-normalized MAOL head; retrain it with the current code."
        )
    model = TorchCNN(
        input_size=config_info.input_size,
        num_classes=config_info.num_classes,
        tree=config_info.tree,
        width_scale=config_info.width_scale,
        embedding_dim=config_info.embedding_dim,
        seed=42,
    )
    model.load_weights(weights_path, map_location=device, strict=True)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one image with a MAOL prototype checkpoint")
    parser.add_argument("image", type=str, help="Path to image file")
    parser.add_argument("--weights", type=str, required=True, help="MAOL .pt/.pth checkpoint")
    parser.add_argument("--probabilities", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")
    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise SystemExit(f"Weights not found: {weights_path}")

    device = _resolve_device(args.device)
    model = load_maol_model(weights_path, device)
    image = load_image(image_path)
    x = preprocess_image(
        image,
        target_size=model.input_size,
        normalize_to=config.NORMALIZE_TO,
        input_value_range=config.INPUT_VALUE_RANGE,
    )
    x_tensor = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    predicted = int(probs.argmax())
    class_names = tuple(model.class_names)
    print(f"Predicted class: {predicted} ({class_names[predicted]})")
    if args.probabilities:
        top_k = max(1, min(int(args.top_k), int(probs.shape[0])))
        for index in probs.argsort()[::-1][:top_k]:
            print(f"  {class_names[int(index)]}: {float(probs[int(index)]):.4f}")


if __name__ == "__main__":
    main()
