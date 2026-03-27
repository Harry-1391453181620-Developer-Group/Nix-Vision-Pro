"""PyTorch inference backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from utils.safety import install_dataset_write_guard

install_dataset_write_guard()

import config
from backends.torch.model import TorchCNN
from data.loaders import load_image
from data.preprocessing import preprocess_image


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device=cuda requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict class for an image with the PyTorch CNN")
    parser.add_argument("image", type=str, help="Path to image file")
    parser.add_argument("--weights", type=str, default=None, help="Path to saved weights .pt/.pth (optional)")
    parser.add_argument("--probabilities", action="store_true", help="Print class probabilities")
    parser.add_argument("--top-k", type=int, default=3, help="Show top-k predictions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used if no weights)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root used to resolve class labels")
    parser.add_argument("--class-count", type=int, default=None, help="Optional class-count override for model output size")
    parser.add_argument("--model-width-scale", type=float, default=0.75, help="Width multiplier for the stage-2 convolution block")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    class_source_dir = Path(args.data_dir) if args.data_dir else None
    try:
        class_names = config.get_class_names(class_source_dir, class_count=args.class_count, require_images=False)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    num_classes = len(class_names)

    device = _resolve_device(args.device)
    model = TorchCNN(input_size=config.INPUT_SIZE, num_classes=num_classes, seed=args.seed, width_scale=args.model_width_scale)
    model.to(device)
    if args.weights:
        weights_path = Path(args.weights)
        if weights_path.suffix not in {".pt", ".pth"}:
            raise SystemExit("Weights file must use .pt or .pth extension")
        model.load_weights(weights_path, map_location=device)
    model.eval()

    image = load_image(image_path)
    x = preprocess_image(
        image,
        target_size=config.INPUT_SIZE,
        normalize_to=config.NORMALIZE_TO,
        input_value_range=config.INPUT_VALUE_RANGE,
    )
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    pred_class = int(probs.argmax())

    print(f"Predicted class: {pred_class} ({class_names[pred_class]})")
    if args.probabilities:
        top_k = max(1, min(args.top_k, num_classes))
        top_indices = probs.argsort()[::-1][:top_k]
        print(f"Top {top_k} probabilities:")
        for index in top_indices:
            print(f"  {class_names[index]}: {probs[index]:.4f}")


if __name__ == "__main__":
    main()
