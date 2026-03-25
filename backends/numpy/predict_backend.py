"""
Inference script: load one image, preprocess, model forward, and print class labels.

The runtime class list is resolved from the dataset so prediction stays aligned
with the active project labels instead of a fixed config snapshot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.safety import install_dataset_write_guard
install_dataset_write_guard()

import config
from data.loaders import load_image
from data.preprocessing import preprocess_image
from backends.numpy.model import CNN
from nn.activations import softmax


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict class for an image with NumPy-only CNN")
    parser.add_argument("image", type=str, help="Path to image file")
    parser.add_argument("--weights", type=str, default=None, help="Path to saved weights .npz (optional)")
    parser.add_argument("--probabilities", action="store_true", help="Print class probabilities")
    parser.add_argument("--top-k", type=int, default=3, help="Show top-k predictions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used if no weights)")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root used to resolve class labels")
    parser.add_argument("--class-count", type=int, default=None, help="Optional class-count override for model output size")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    input_size = config.INPUT_SIZE
    class_source_dir = Path(args.data_dir) if args.data_dir else None
    try:
        class_names = config.get_class_names(class_source_dir, class_count=args.class_count, require_images=False)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    num_classes = len(class_names)

    image = load_image(image_path)
    x = preprocess_image(
        image,
        target_size=input_size,
        normalize_to=config.NORMALIZE_TO,
        input_value_range=config.INPUT_VALUE_RANGE,
    )
    x = x[np.newaxis, ...]  # (1, H, W, 3)

    model = CNN(input_size=input_size, num_classes=num_classes, seed=args.seed)
    if args.weights:
        weights_path = Path(args.weights)
        if weights_path.suffix != ".npz":
            raise SystemExit("Weights file must use .npz extension")
        model.load_weights(weights_path)
    model.eval()

    logits = model.forward(x)
    probs = softmax(logits, axis=-1)[0]
    pred_class = int(np.argmax(probs))

    print(f"Predicted class: {pred_class} ({class_names[pred_class]})")
    if args.probabilities:
        top_k = max(1, min(args.top_k, num_classes))
        top_indices = np.argsort(probs)[::-1][:top_k]
        print(f"Top {top_k} probabilities:")
        for index in top_indices:
            print(f"  {class_names[index]}: {probs[index]:.4f}")


if __name__ == "__main__":
    main()
