import os
import sys
from pathlib import Path
import pytest

# Ensure we import project utils
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.safety import install_dataset_write_guard, DATASET_DIR


def test_dataset_open_write_is_blocked():
    install_dataset_write_guard()
    target = DATASET_DIR / '___guard_test___'
    with pytest.raises(PermissionError):
        open(target, 'w').close()


def test_dataset_remove_is_blocked():
    install_dataset_write_guard()
    target = DATASET_DIR / '___guard_test___to_remove'
    with pytest.raises(PermissionError):
        os.remove(target)
