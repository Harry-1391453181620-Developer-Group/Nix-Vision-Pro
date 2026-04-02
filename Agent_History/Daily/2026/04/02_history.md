# 2026-04-02 History

## Request

Audited the whole repository, removed stale non-runtime code and tracked junk, fixed the full test workflow, and aligned project docs with the actual supported training path.

## Work completed

- Removed stale legacy code that was no longer part of the supported runtime surface:
  - `build_synthetic_dataset.py`
  - `build_wikimedia_dataset.py`
  - `nn/mamba.py`
  - `backends/torch/mamba.py`
  - `backends/numpy/nn/mamba.py`
  - `tests/test_mamba.py`
  - `debug-eccffb.log`
- Updated `backends/numpy/nn/__init__.py` so the NumPy NN export surface no longer exposes the removed Mamba block.
- Fixed `data/loaders.py` so `load_image()` closes Pillow file handles promptly by using a context manager.
- Fixed the full test workflow in this environment by adding `tests/conftest.py` to redirect Python temp usage into the workspace instead of the blocked Windows temp root.
- Simplified `tests/test_dataset_guard.py` so it no longer requests the unused `tmp_path` fixture.
- Reworked `tests/test_data.py` to use workspace-local temporary directories instead of `TemporaryDirectory()`.
- Reworked `tests/test_runtime_class_config.py` to cover the supported dataset-driven config behavior only, after removing the stale dataset-builder utilities.
- Updated `README.md`, `Image_Identify_CNN.md`, and `CONTRIBUTING.md` so they now describe:
  - training from the real `Dataset/` image tree
  - removal of stale synthetic/Wikimedia builder scripts
  - removal of the dead Mamba implementation
  - the current test command

## Validation

- `.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider`
- Result: `53 passed`

## Notes

- The removed dataset-builder scripts only supported an older small class subset and were not part of the actual training or inference path for the current 62-class dataset.
- Some ignored local temp/cache directories inside the workspace remain permission-restricted in this environment, but no tracked project files depend on them.
- Passive security review: the cleanup reduced attack surface by removing dead code paths and an old debug log, and the loader fix reduced the risk of descriptor leaks during long-running image loads.
