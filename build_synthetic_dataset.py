"""
Generate a balanced synthetic image classification dataset using Pillow only.

The active class list is resolved from `Dataset/` when possible so the builder
tracks the rest of the project automatically. Only built-in drawable classes are
supported by this generator.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import config


def rand_color(rng: random.Random, low: int = 20, high: int = 235) -> tuple[int, int, int]:
    return (rng.randint(low, high), rng.randint(low, high), rng.randint(low, high))


def make_background(size: int, rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (size, size), rand_color(rng, 5, 40))
    draw = ImageDraw.Draw(img)
    c1 = rand_color(rng, 15, 80)
    c2 = rand_color(rng, 80, 200)
    for y in range(size):
        t = y / max(1, size - 1)
        col = (
            int(c1[0] * (1 - t) + c2[0] * t),
            int(c1[1] * (1 - t) + c2[1] * t),
            int(c1[2] * (1 - t) + c2[2] * t),
        )
        draw.line([(0, y), (size, y)], fill=col)

    # Add small noisy speckles.
    for _ in range(size * size // 150):
        x = rng.randint(0, size - 1)
        y = rng.randint(0, size - 1)
        draw.point((x, y), fill=rand_color(rng, 0, 255))
    return img


def jitter_points(points: list[tuple[float, float]], rng: random.Random, amount: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for x, y in points:
        out.append((x + rng.uniform(-amount, amount), y + rng.uniform(-amount, amount)))
    return out


def draw_airplane(draw: ImageDraw.ImageDraw, s: int, rng: random.Random, color: tuple[int, int, int]) -> None:
    cx, cy = s * 0.5, s * 0.55
    w, h = s * 0.55, s * 0.25
    pts = [
        (cx - 0.45 * w, cy),
        (cx + 0.35 * w, cy),
        (cx + 0.50 * w, cy - 0.07 * h),
        (cx + 0.35 * w, cy + 0.07 * h),
        (cx - 0.45 * w, cy + 0.02 * h),
        (cx - 0.18 * w, cy + 0.23 * h),
        (cx - 0.02 * w, cy + 0.23 * h),
        (cx - 0.05 * w, cy + 0.02 * h),
        (cx + 0.06 * w, cy + 0.02 * h),
        (cx + 0.20 * w, cy + 0.20 * h),
        (cx + 0.35 * w, cy + 0.20 * h),
        (cx + 0.12 * w, cy + 0.02 * h),
        (cx + 0.12 * w, cy - 0.02 * h),
        (cx + 0.35 * w, cy - 0.20 * h),
        (cx + 0.20 * w, cy - 0.20 * h),
        (cx + 0.06 * w, cy - 0.02 * h),
        (cx - 0.05 * w, cy - 0.02 * h),
        (cx - 0.02 * w, cy - 0.23 * h),
        (cx - 0.18 * w, cy - 0.23 * h),
        (cx - 0.45 * w, cy),
    ]
    draw.polygon(jitter_points(pts, rng, s * 0.008), fill=color)


def draw_automobile(draw: ImageDraw.ImageDraw, s: int, rng: random.Random, color: tuple[int, int, int]) -> None:
    x1, y1 = s * 0.18, s * 0.55
    x2, y2 = s * 0.82, s * 0.76
    draw.rounded_rectangle((x1, y1, x2, y2), radius=int(s * 0.05), fill=color)
    draw.polygon(
        jitter_points(
            [(s * 0.28, s * 0.55), (s * 0.40, s * 0.43), (s * 0.65, s * 0.43), (s * 0.76, s * 0.55)],
            rng,
            s * 0.01,
        ),
        fill=color,
    )
    tire = rand_color(rng, 15, 50)
    rim = rand_color(rng, 180, 230)
    for cx in (s * 0.33, s * 0.70):
        r = s * 0.07
        draw.ellipse((cx - r, s * 0.70 - r, cx + r, s * 0.70 + r), fill=tire)
        draw.ellipse((cx - r * 0.45, s * 0.70 - r * 0.45, cx + r * 0.45, s * 0.70 + r * 0.45), fill=rim)


def draw_bird(draw: ImageDraw.ImageDraw, s: int, rng: random.Random, color: tuple[int, int, int]) -> None:
    body = (s * 0.27, s * 0.42, s * 0.74, s * 0.72)
    draw.ellipse(body, fill=color)
    wing = jitter_points([(s * 0.44, s * 0.56), (s * 0.58, s * 0.40), (s * 0.64, s * 0.62)], rng, s * 0.01)
    draw.polygon(wing, fill=rand_color(rng, 40, 120))
    beak = jitter_points([(s * 0.74, s * 0.56), (s * 0.88, s * 0.52), (s * 0.74, s * 0.62)], rng, s * 0.008)
    draw.polygon(beak, fill=(240, 190, 40))
    eye_r = s * 0.015
    draw.ellipse((s * 0.61 - eye_r, s * 0.51 - eye_r, s * 0.61 + eye_r, s * 0.51 + eye_r), fill=(0, 0, 0))


def draw_cat(draw: ImageDraw.ImageDraw, s: int, rng: random.Random, color: tuple[int, int, int]) -> None:
    draw.ellipse((s * 0.26, s * 0.32, s * 0.74, s * 0.78), fill=color)
    ear_l = jitter_points([(s * 0.33, s * 0.34), (s * 0.42, s * 0.12), (s * 0.50, s * 0.34)], rng, s * 0.01)
    ear_r = jitter_points([(s * 0.50, s * 0.34), (s * 0.58, s * 0.12), (s * 0.67, s * 0.34)], rng, s * 0.01)
    draw.polygon(ear_l, fill=color)
    draw.polygon(ear_r, fill=color)
    for ex in (s * 0.43, s * 0.57):
        draw.ellipse((ex - s * 0.02, s * 0.48 - s * 0.02, ex + s * 0.02, s * 0.48 + s * 0.02), fill=(0, 0, 0))
    draw.polygon([(s * 0.50, s * 0.55), (s * 0.47, s * 0.60), (s * 0.53, s * 0.60)], fill=(220, 120, 120))


def draw_deer(draw: ImageDraw.ImageDraw, s: int, rng: random.Random, color: tuple[int, int, int]) -> None:
    draw.ellipse((s * 0.30, s * 0.33, s * 0.70, s * 0.78), fill=color)
    for x in (s * 0.42, s * 0.58):
        draw.rectangle((x - s * 0.015, s * 0.70, x + s * 0.015, s * 0.88), fill=color)
    horn_color = rand_color(rng, 120, 210)
    for x in (s * 0.40, s * 0.60):
        draw.line((x, s * 0.30, x - s * 0.05, s * 0.15), fill=horn_color, width=max(2, s // 40))
        draw.line((x, s * 0.24, x + s * 0.05, s * 0.12), fill=horn_color, width=max(2, s // 45))
    for ex in (s * 0.45, s * 0.55):
        draw.ellipse((ex - s * 0.014, s * 0.50 - s * 0.014, ex + s * 0.014, s * 0.50 + s * 0.014), fill=(0, 0, 0))


def draw_dog(draw: ImageDraw.ImageDraw, s: int, rng: random.Random, color: tuple[int, int, int]) -> None:
    draw.ellipse((s * 0.26, s * 0.30, s * 0.74, s * 0.78), fill=color)
    ear_l = jitter_points([(s * 0.24, s * 0.40), (s * 0.36, s * 0.30), (s * 0.32, s * 0.56)], rng, s * 0.01)
    ear_r = jitter_points([(s * 0.76, s * 0.40), (s * 0.64, s * 0.30), (s * 0.68, s * 0.56)], rng, s * 0.01)
    draw.polygon(ear_l, fill=rand_color(rng, 25, 95))
    draw.polygon(ear_r, fill=rand_color(rng, 25, 95))
    draw.ellipse((s * 0.44, s * 0.58, s * 0.56, s * 0.70), fill=rand_color(rng, 40, 110))
    for ex in (s * 0.43, s * 0.57):
        draw.ellipse((ex - s * 0.015, s * 0.50 - s * 0.015, ex + s * 0.015, s * 0.50 + s * 0.015), fill=(0, 0, 0))


def draw_frog(draw: ImageDraw.ImageDraw, s: int, rng: random.Random, color: tuple[int, int, int]) -> None:
    draw.ellipse((s * 0.22, s * 0.38, s * 0.78, s * 0.80), fill=color)
    for ex in (s * 0.39, s * 0.61):
        draw.ellipse((ex - s * 0.07, s * 0.30 - s * 0.07, ex + s * 0.07, s * 0.30 + s * 0.07), fill=color)
        draw.ellipse((ex - s * 0.025, s * 0.30 - s * 0.025, ex + s * 0.025, s * 0.30 + s * 0.025), fill=(0, 0, 0))
    draw.arc((s * 0.38, s * 0.56, s * 0.62, s * 0.72), start=10, end=170, fill=(20, 20, 20), width=max(2, s // 60))


def draw_horse(draw: ImageDraw.ImageDraw, s: int, rng: random.Random, color: tuple[int, int, int]) -> None:
    body = jitter_points(
        [(s * 0.20, s * 0.62), (s * 0.62, s * 0.62), (s * 0.66, s * 0.52), (s * 0.74, s * 0.50),
         (s * 0.70, s * 0.40), (s * 0.58, s * 0.42), (s * 0.54, s * 0.50), (s * 0.20, s * 0.50)],
        rng,
        s * 0.01,
    )
    draw.polygon(body, fill=color)
    for x in (s * 0.27, s * 0.40, s * 0.52, s * 0.60):
        draw.rectangle((x - s * 0.015, s * 0.62, x + s * 0.015, s * 0.86), fill=color)
    draw.line((s * 0.20, s * 0.52, s * 0.10, s * 0.42), fill=rand_color(rng, 25, 90), width=max(2, s // 45))


def draw_ship(draw: ImageDraw.ImageDraw, s: int, rng: random.Random, color: tuple[int, int, int]) -> None:
    water = rand_color(rng, 20, 90)
    draw.rectangle((0, s * 0.66, s, s), fill=water)
    hull = jitter_points([(s * 0.15, s * 0.66), (s * 0.78, s * 0.66), (s * 0.68, s * 0.82), (s * 0.22, s * 0.82)], rng, s * 0.01)
    draw.polygon(hull, fill=color)
    draw.rectangle((s * 0.33, s * 0.52, s * 0.56, s * 0.66), fill=rand_color(rng, 160, 230))
    draw.line((s * 0.45, s * 0.32, s * 0.45, s * 0.52), fill=rand_color(rng, 110, 210), width=max(2, s // 55))
    draw.polygon(jitter_points([(s * 0.45, s * 0.34), (s * 0.62, s * 0.46), (s * 0.45, s * 0.46)], rng, s * 0.008), fill=rand_color(rng, 170, 245))


def draw_truck(draw: ImageDraw.ImageDraw, s: int, rng: random.Random, color: tuple[int, int, int]) -> None:
    draw.rectangle((s * 0.14, s * 0.53, s * 0.64, s * 0.75), fill=color)
    cab = rand_color(rng, 120, 240)
    draw.rectangle((s * 0.64, s * 0.58, s * 0.82, s * 0.75), fill=cab)
    draw.polygon(jitter_points([(s * 0.64, s * 0.58), (s * 0.72, s * 0.50), (s * 0.82, s * 0.58)], rng, s * 0.007), fill=cab)
    tire = rand_color(rng, 8, 45)
    for cx in (s * 0.26, s * 0.55, s * 0.74):
        r = s * 0.065
        draw.ellipse((cx - r, s * 0.75 - r, cx + r, s * 0.75 + r), fill=tire)


DRAWERS: dict[str, Callable[[ImageDraw.ImageDraw, int, random.Random, tuple[int, int, int]], None]] = {
    "airplane": draw_airplane,
    "automobile": draw_automobile,
    "bird": draw_bird,
    "cat": draw_cat,
    "deer": draw_deer,
    "dog": draw_dog,
    "frog": draw_frog,
    "horse": draw_horse,
    "ship": draw_ship,
    "truck": draw_truck,
}

DEFAULT_SYNTHETIC_CLASS_NAMES = list(DRAWERS.keys())


def resolve_synthetic_class_names(data_dir: str | Path | None = None) -> list[str]:
    """
    Resolve which classes the synthetic generator should build for this project.

    Dataset directory names take priority so the generator follows the active
    project labels, but generation is still limited to classes that have a
    corresponding built-in drawing routine.
    """
    class_names = config.get_class_names(
        data_dir=data_dir,
        require_images=False,
        fallback=DEFAULT_SYNTHETIC_CLASS_NAMES,
    )
    unsupported = [class_name for class_name in class_names if class_name not in DRAWERS]
    if unsupported:
        supported = ", ".join(sorted(DRAWERS))
        missing = ", ".join(unsupported)
        raise SystemExit(
            f"Synthetic generator has no drawer for class(es): {missing}. Supported classes: {supported}"
        )
    return class_names


def draw_sample(class_name: str, size: int, rng: random.Random) -> Image.Image:
    bg = make_background(size, rng)
    draw = ImageDraw.Draw(bg)
    base_color = rand_color(rng, 45, 225)
    DRAWERS[class_name](draw, size, rng, base_color)

    # Mild global augmentations for intra-class diversity.
    angle = rng.uniform(-18, 18)
    bg = bg.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=rand_color(rng, 10, 80))
    if rng.random() < 0.35:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.3)))
    if rng.random() < 0.35:
        bg = ImageEnhance.Contrast(bg).enhance(rng.uniform(0.85, 1.3))
    if rng.random() < 0.35:
        bg = ImageEnhance.Color(bg).enhance(rng.uniform(0.8, 1.3))
    return bg


# ---- SAFETY GUARD: never write into the project "Dataset" directory ----
from pathlib import Path as _P
_PROJ_ROOT = _P(__file__).resolve().parent
_RESERVED_DATASET = (_PROJ_ROOT / "Dataset").resolve()

def _assert_safe_out_dir(out_dir: Path) -> None:
    """Refuse writing into the reserved project Dataset directory."""
    out_r = out_dir.resolve()
    s = str(out_r).lower()
    if out_r == _RESERVED_DATASET or s.endswith("\\\\dataset") or s.endswith("/dataset"):
        raise SystemExit(
            "Safety: refusing to write into 'Dataset'. Choose a different --out-dir (e.g., 'SyntheticDataset')."
        )

def generate_dataset(
    out_dir: Path,
    images_per_class: int,
    size: int,
    seed: int,
    clean: bool,
    class_names: list[str] | None = None,
) -> None:
    """
    Generate a synthetic dataset under `out_dir` with class subfolders.
    Renders `images_per_class` images per class at square `size` using a fixed RNG `seed`.
    If `clean` is True, existing class folders and metadata files are removed first.
    """
    _assert_safe_out_dir(out_dir)
    active_class_names = list(class_names or resolve_synthetic_class_names())

    # Optionally clear previous outputs (never for reserved directory — guarded above).
    if clean and out_dir.exists():
        for cls in active_class_names:
            cls_dir = out_dir / cls
            if cls_dir.exists():
                shutil.rmtree(cls_dir, ignore_errors=True)
        for meta_name in ("synthetic_dataset_info.json", "metadata.jsonl"):
            meta_path = out_dir / meta_name
            if meta_path.exists():
                try:
                    meta_path.unlink()
                except Exception:
                    pass

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for class_name in active_class_names:
        cls_dir = out_dir / class_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(cls_dir.glob("*.jpg")))
        for i in range(existing, images_per_class):
            img = draw_sample(class_name, size=size, rng=rng)
            out_path = cls_dir / f"{i + 1:04d}.jpg"
            img.save(out_path, format="JPEG", quality=94, optimize=True)
        print(f"[class] {class_name}: {images_per_class} images")

    info = {
        "type": "synthetic",
        "generator": "build_synthetic_dataset.py",
        "classes": active_class_names,
        "images_per_class": images_per_class,
        "image_size": [size, size],
        "seed": seed,
    }
    (out_dir / "synthetic_dataset_info.json").write_text(json_dumps(info), encoding="utf-8")
    print(f"[done] dataset generated at: {out_dir}")


def json_dumps(data: dict) -> str:
    import json

    return json.dumps(data, indent=2, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic image classification dataset.")
    parser.add_argument("--out-dir", type=str, required=True, help="Dataset root directory.")
    parser.add_argument("--images-per-class", type=int, default=500, help="Number of images per class.")
    parser.add_argument("--size", type=int, default=256, help="Image size (square).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--clean", action="store_true", help="Clear existing class folders before generation.")
    args = parser.parse_args()

    if args.images_per_class <= 0:
        raise SystemExit("--images-per-class must be > 0")
    if args.size < 64:
        raise SystemExit("--size must be >= 64")
    class_names = resolve_synthetic_class_names()

    generate_dataset(
        out_dir=Path(args.out_dir),
        images_per_class=args.images_per_class,
        size=args.size,
        seed=args.seed,
        clean=args.clean,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
