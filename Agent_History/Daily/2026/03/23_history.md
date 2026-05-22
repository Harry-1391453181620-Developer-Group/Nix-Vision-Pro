# Daily History - 2026-03-23

## Summary
- Fixed the synthetic dataset builder regression and restored the dataset generation path.
- Added training improvements to the original NumPy path, including BatchNorm support, stronger augmentation, scheduler options, label smoothing, and early stopping controls.
- Repaired project regressions introduced during those changes.
- Split the project into explicit `backends/torch/` and `backends/numpy/` directories while preserving the original structure.
- Made the PyTorch backend the default runtime path without deleting the NumPy backend.
- Updated the project documentation to match the new backend layout and the final Python 3.14 `.venv` decision.
- Repaired the Python 3.14 `.venv` so it now reuses globally installed runtime packages and contains project test tooling.

## Code Changes
- Fixed the recursion bug in `build_synthetic_dataset.py` and restored `generate_dataset()`.
- Added or completed class weighting, label smoothing, BatchNorm integration, stronger augmentation, scheduler selection, and early stopping controls in the training flow.
- Restored the broken `build_wikimedia_dataset.py` entrypoint.
- Restored the missing `DepthwiseConv2D` declaration in the NumPy layer stack.
- Fixed checkpoint serialization so BatchNorm state is saved and loaded correctly.
- Reorganized the codebase so:
  - `backends/numpy/` contains the preserved original NumPy model, training, prediction, GUI, and `nn` primitives.
  - `backends/torch/` contains the PyTorch model, Mamba block, training, prediction, and GUI backends.
  - top-level `train.py`, `predict.py`, `gui.py`, and `model.py` act as compatibility and dispatch layers.
  - top-level `nn/` remains as a compatibility export for the NumPy backend modules.

## Environment Changes
- Confirmed the project target remains Python 3.14.
- Verified the global Python 3.14 installation contains `torch`, `numpy`, `Pillow`, and `opencv-python`.
- Repaired `.venv\pyvenv.cfg` so `.venv` now uses `include-system-site-packages = true`.
- Installed `pytest` into the Python 3.14 project venv.
- Updated `requirements.txt` to cover the full current project dependency set: `torch`, `numpy`, `pillow`, `opencv-python`, and `pytest`.

## Documentation Updates
- Rewrote `README.md` to describe the new backend split, the Python 3.14 `.venv` workflow, and the current training and inference commands.
- Rewrote `CONTRIBUTING.md` to match the current repository structure, backend expectations, validation flow, and project rules.
- Rewrote `Image_Identify_CNN.md` so it now documents the final Python 3.14 `.venv` decision, the backend layout, the PyTorch default path, and the full recommended training command.
- Reviewed `rulesForAgents.md` and `Agent_History/Always/Project_Final_Goal.md`; no content changes were needed because both already match the current project policy.

## Verification
- Verified `.venv\Scripts\python.exe` now resolves `torch`, `numpy`, `Pillow`, `opencv-python`, `pip`, and `pytest`.
- Verified `train.py --help` works from the Python 3.14 project venv with the PyTorch backend as default.
- Verified `predict.py --help` works from the Python 3.14 project venv.
- Previously verified repository compile checks, NumPy regression tests, and PyTorch smoke tests during the backend migration.

## Notes
- The GUI help path is not a practical non-interactive validation target because `gui.py` launches the application rather than behaving like a standard CLI tool.
- The NumPy backend remains intentionally preserved for compatibility, regression checking, and architecture comparison.
