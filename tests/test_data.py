"""Unit tests for data loaders and preprocessing."""

import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from data.loaders import load_image, load_images_from_dir
from data.preprocessing import batch_preprocess, normalize, preprocess_image, resize


_TEST_TMP_ROOT = Path('.pytest-tmp') / 'data_tests'
_TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def test_load_image_creates_ndarray():
    """load_image returns ndarray (H, W, 3) with values in [0, 255]."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img = Image.new('RGB', (10, 8), color=(100, 150, 200))
        img.save(f.name)
        path = f.name
    try:
        arr = load_image(path)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (8, 10, 3)
        assert arr.dtype == np.float64
        assert arr.min() >= 0 and arr.max() <= 255
    finally:
        Path(path).unlink(missing_ok=True)


def test_resize_shape():
    """resize produces target (H, W, 3)."""
    image = np.random.randint(0, 256, (20, 30, 3), dtype=np.uint8).astype(np.float64)
    out = resize(image, (10, 15))
    assert out.shape == (10, 15, 3)


def test_normalize_range():
    """normalize maps [0,255] to [0,1] by default."""
    image = np.array([[[0, 127.5, 255]]], dtype=np.float64)
    out = normalize(image, scale=(0.0, 1.0), input_range=(0.0, 255.0))
    assert out.shape == (1, 1, 3)
    np.testing.assert_allclose(out[0, 0], [0.0, 0.5, 1.0], atol=1e-6)


def test_preprocess_image_shape_and_range():
    """preprocess_image returns (H, W, 3) in [0, 1]."""
    image = np.random.randint(0, 256, (24, 32, 3), dtype=np.uint8).astype(np.float64)
    out = preprocess_image(image, target_size=(16, 16), normalize_to=(0.0, 1.0))
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.float64
    assert out.min() >= 0 and out.max() <= 1


def test_batch_preprocess_stacks():
    """batch_preprocess returns (N, H, W, 3)."""
    images = [
        np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8).astype(np.float64),
        np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8).astype(np.float64),
    ]
    out = batch_preprocess(images, target_size=(8, 8))
    assert out.shape == (2, 8, 8, 3)


def test_load_image_nonexistent_raises():
    """load_image raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_image('/nonexistent/path/image.png')


def test_load_images_from_dir_empty_or_missing():
    """load_images_from_dir on empty dir returns empty lists."""
    temp_dir = _TEST_TMP_ROOT / uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        images, paths = load_images_from_dir(temp_dir)
        assert images == []
        assert paths == []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    with pytest.raises(NotADirectoryError):
        load_images_from_dir('/nonexistent/dir')
