"""Pytest session configuration for workspace-local temporary directories."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = ROOT / '.pytest-tmp'
SYSTEM_TEMP_ROOT = TEST_TMP_ROOT / 'system-temp'

for directory in (TEST_TMP_ROOT, SYSTEM_TEMP_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

for env_name in ('TMPDIR', 'TEMP', 'TMP'):
    os.environ[env_name] = str(SYSTEM_TEMP_ROOT)
tempfile.tempdir = str(SYSTEM_TEMP_ROOT)
