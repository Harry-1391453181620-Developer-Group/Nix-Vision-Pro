"""
Build a high-quality image dataset from Wikimedia Commons category pages.

Output layout:
    <out_dir>/
        airplane/
        automobile/
        ...
        metadata.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image

import config

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "Image_Identify_CNN_DatasetBuilder/1.0 (Wikimedia Commons crawler)"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Supported Wikimedia category seeds. The active subset is resolved from
# `Dataset/` so the downloader follows the project class layout automatically.
DEFAULT_CLASS_SOURCES: Dict[str, List[str]] = {
    "airplane": ["Airliners", "Airplanes in flight", "Commercial aircraft"],
    "automobile": ["Automobiles", "Cars"],
    "bird": ["Birds", "Birds in flight"],
    "cat": ["Cats", "Domestic cats"],
    "deer": ["Deer"],
    "dog": ["Dogs", "Domestic dogs"],
    "frog": ["Frogs"],
    "horse": ["Horses", "Racehorses"],
    "ship": ["Ships", "Cargo ships"],
    "truck": ["Trucks", "Lorries"],
}

BAN_TITLE_TOKENS = (
    "logo",
    "icon",
    "diagram",
    "map",
    "flag",
    "coat of arms",
    "drawing",
    "symbol",
    "scheme",
)


@dataclass
class Candidate:
    cls: str
    category: str
    title: str
    url: str
    download_url: str
    description_url: str
    width: int
    height: int
    license: str


def resolve_wikimedia_class_sources(data_dir: str | Path | None = None) -> Dict[str, List[str]]:
    """
    Resolve which Wikimedia class sources should be active for this project.

    Dataset directory names take priority so this builder stays aligned with the
    rest of the project, but every requested class must still have a supported
    Wikimedia category seed in `DEFAULT_CLASS_SOURCES`.
    """
    class_names = config.get_class_names(
        data_dir=data_dir,
        require_images=False,
        fallback=tuple(DEFAULT_CLASS_SOURCES.keys()),
    )
    unsupported = [class_name for class_name in class_names if class_name not in DEFAULT_CLASS_SOURCES]
    if unsupported:
        supported = ", ".join(sorted(DEFAULT_CLASS_SOURCES))
        missing = ", ".join(unsupported)
        raise SystemExit(
            f"Wikimedia builder has no category seeds for class(es): {missing}. Supported classes: {supported}"
        )
    return {class_name: list(DEFAULT_CLASS_SOURCES[class_name]) for class_name in class_names}


def commons_api_query(params: Dict[str, str], retries: int = 3) -> dict:
    query = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        **params,
    }
    url = f"{API_URL}?{urllib.parse.urlencode(query)}"
    headers = {"User-Agent": USER_AGENT}
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=40) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                time.sleep(1.0 * attempt)
    raise RuntimeError(f"Commons API request failed: {url}") from last_exc


def fetch_candidates_for_category(category: str, max_pages: int = 800) -> List[Candidate]:
    out: List[Candidate] = []
    cont: Dict[str, str] = {}

    while len(out) < max_pages:
        params = {
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": "500",
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": "1024",
        }
        params.update(cont)
        data = commons_api_query(params)
        pages = data.get("query", {}).get("pages", [])

        for page in pages:
            title = str(page.get("title", ""))
            info = (page.get("imageinfo") or [{}])[0]
            url = str(info.get("url", ""))
            if not url:
                continue
            ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
            if ext not in ALLOWED_EXTS:
                continue
            title_l = title.lower()
            if any(tok in title_l for tok in BAN_TITLE_TOKENS):
                continue
            width = int(info.get("width", 0) or 0)
            height = int(info.get("height", 0) or 0)
            license_name = (
                info.get("extmetadata", {})
                .get("LicenseShortName", {})
                .get("value", "Unknown")
            )
            out.append(
                Candidate(
                    cls="",
                    category=category,
                    title=title,
                    url=url,
                    download_url=str(info.get("thumburl") or url),
                    description_url=str(info.get("descriptionurl", "")),
                    width=width,
                    height=height,
                    license=str(license_name),
                )
            )

        if "continue" not in data:
            break
        cont = {
            k: v
            for k, v in data["continue"].items()
            if k != "continue"
        }

    return out


def iter_candidates_for_class(cls: str, categories: Iterable[str]) -> List[Candidate]:
    merged: List[Candidate] = []
    seen_urls: set[str] = set()
    for cat in categories:
        try:
            cat_candidates = fetch_candidates_for_category(cat)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] class={cls} category={cat} fetch failed: {exc}")
            continue
        for cand in cat_candidates:
            if cand.url in seen_urls:
                continue
            seen_urls.add(cand.url)
            cand.cls = cls
            merged.append(cand)
    return merged


def decode_image_to_rgb(image_bytes: bytes) -> Image.Image | None:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Force decode before leaving the context.
            img.load()
            if img.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                alpha = img.getchannel("A")
                bg.paste(img.convert("RGB"), mask=alpha)
                return bg
            if img.mode != "RGB":
                return img.convert("RGB")
            return img.copy()
    except Exception:  # noqa: BLE001
        return None


def download_bytes(url: str, timeout: int = 40) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def save_dataset(
    out_dir: Path,
    per_class: int,
    min_width: int,
    min_height: int,
    seed: int,
    class_sources: Dict[str, List[str]] | None = None,
) -> None:
    active_class_sources = class_sources or resolve_wikimedia_class_sources()
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.jsonl"
    global_hashes: set[str] = set()
    existing_lines = 0
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                for _ in f:
                    existing_lines += 1
        except Exception:  # noqa: BLE001
            existing_lines = 0

    # Seed dedup set with already-downloaded images for resume runs.
    for cls in active_class_sources:
        cls_dir = out_dir / cls
        if not cls_dir.exists():
            continue
        for p in cls_dir.glob("*.jpg"):
            try:
                with Image.open(p) as img:
                    img = img.convert("RGB")
                    global_hashes.add(hashlib.sha1(img.tobytes()).hexdigest())
            except Exception:  # noqa: BLE001
                continue

    total_saved = 0
    with metadata_path.open("a", encoding="utf-8") as meta_f:
        for cls, categories in active_class_sources.items():
            cls_dir = out_dir / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            existing_files = sorted(cls_dir.glob("*.jpg"))
            already_saved = len(existing_files)
            if already_saved >= per_class:
                print(f"[class] {cls}: already {already_saved}/{per_class}, skipping")
                continue

            candidates = iter_candidates_for_class(cls, categories)
            candidates = [
                c
                for c in candidates
                if c.width >= min_width and c.height >= min_height
            ]
            rng.shuffle(candidates)

            saved = already_saved
            for cand in candidates:
                if saved >= per_class:
                    break
                try:
                    raw = download_bytes(cand.download_url)
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    continue
                if len(raw) < 20_000:  # tiny files are usually unusable.
                    continue

                img = decode_image_to_rgb(raw)
                if img is None:
                    continue
                w, h = img.size
                if w < min_width or h < min_height:
                    continue

                content_hash = hashlib.sha1(img.tobytes()).hexdigest()
                if content_hash in global_hashes:
                    continue
                global_hashes.add(content_hash)

                out_name = f"{saved + 1:04d}.jpg"
                out_path = cls_dir / out_name
                try:
                    img.save(out_path, format="JPEG", quality=92, optimize=True)
                except Exception:  # noqa: BLE001
                    continue

                saved += 1
                total_saved += 1
                meta = {
                    "class": cls,
                    "file": str(out_path),
                    "source_url": cand.url,
                    "download_url": cand.download_url,
                    "description_url": cand.description_url,
                    "title": cand.title,
                    "category": cand.category,
                    "license": cand.license,
                    "width": w,
                    "height": h,
                    "sha1": content_hash,
                }
                meta_f.write(json.dumps(meta, ensure_ascii=True) + "\n")

            print(f"[class] {cls}: saved {saved}/{per_class}")
            if saved < per_class:
                print(
                    f"[warn] class {cls} has only {saved} images. "
                    "Try reducing --per-class or adding more source categories."
                )

    print(f"[done] dataset directory: {out_dir}")
    print(f"[done] total images saved: {total_saved}")
    print(f"[done] metadata: {metadata_path} (appended from line {existing_lines + 1})")


# ---- SAFETY GUARD: never write into the project "Dataset" directory ----
from pathlib import Path as _P
_PROJ_ROOT = _P(__file__).resolve().parent
_RESERVED_DATASET = (_PROJ_ROOT / "Dataset").resolve()

def _assert_safe_out_dir(out_dir: Path) -> None:
    out_r = out_dir.resolve()
    s = str(out_r).lower()
    if out_r == _RESERVED_DATASET or s.endswith("\\dataset") or s.endswith("/dataset"):
        raise SystemExit(
            "Safety: refusing to write into 'Dataset'. Choose a different --out-dir (e.g., 'WikimediaDataset')."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an image dataset from Wikimedia Commons.")
    parser.add_argument("--out-dir", type=str, required=True, help="Dataset root directory.")
    parser.add_argument("--per-class", type=int, default=500, help="Maximum images to save per class.")
    parser.add_argument("--min-width", type=int, default=192, help="Minimum accepted image width.")
    parser.add_argument("--min-height", type=int, default=192, help="Minimum accepted image height.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    _assert_safe_out_dir(out_dir)

    if args.per_class <= 0:
        raise SystemExit("--per-class must be > 0")
    if args.min_width <= 0 or args.min_height <= 0:
        raise SystemExit("--min-width and --min-height must be > 0")
    class_sources = resolve_wikimedia_class_sources()

    save_dataset(
        out_dir=out_dir,
        per_class=args.per_class,
        min_width=args.min_width,
        min_height=args.min_height,
        seed=args.seed,
        class_sources=class_sources,
    )


if __name__ == "__main__":
    main()

