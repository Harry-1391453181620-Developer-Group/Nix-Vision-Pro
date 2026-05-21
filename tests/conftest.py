"""Pytest session configuration for workspace-local temporary directories."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = ROOT / '.pytest-tmp'


def _resolve_writable_tmp_root() -> Path:
    candidates = (
        TEST_TMP_ROOT,
        ROOT / 'runs' / '_pytest_tmp',
    )
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / '.write_probe'
            probe.write_text('ok', encoding='utf-8')
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise RuntimeError('No writable pytest temporary directory is available inside the workspace')


TEST_TMP_ROOT = _resolve_writable_tmp_root()
SYSTEM_TEMP_ROOT = TEST_TMP_ROOT / 'system-temp'

for directory in (TEST_TMP_ROOT, SYSTEM_TEMP_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

for env_name in ('TMPDIR', 'TEMP', 'TMP'):
    os.environ[env_name] = str(SYSTEM_TEMP_ROOT)
tempfile.tempdir = str(SYSTEM_TEMP_ROOT)


def workspace_tmp_path(*parts: str) -> Path:
    path = TEST_TMP_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
