# Contributing

## Environment

- Python version: `3.14`
- Preferred environment: project-local `.venv`
- Current `.venv` model: `python -m venv --system-site-packages .venv`

Setup:

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Repository Structure

- `backends/torch/` contains the active PyTorch backend.
- `backends/numpy/` contains the preserved legacy NumPy backend.
- `utils/training.py` contains shared augmentation and phase-schedule policy.
- Top-level `train.py`, `predict.py`, `gui.py`, and `model.py` are compatibility and dispatch entrypoints.
- The supported dataset contract is `Dataset/<class_name>/*`; keep training and inference aligned with that layout.

## Contribution Rules

- Keep the original project structure intact; do not remove the NumPy backend.
- Preserve the current high-level CNN architecture unless a change is explicitly requested.
- Write clear, detailed comments in code changes.
- Keep security and reliability ahead of convenience.
- Update the daily history file under `Agent_History/Daily/` for every substantive change.
- Keep documentation aligned with the actual runtime behavior.

## Backend Expectations

- PyTorch is the default backend for new training and inference work.
- NumPy remains available for regression checks, legacy checkpoints, and architecture comparison.
- Shared training policy should stay aligned across both backends unless a backend-specific deviation is explicitly required.
- Temporary backbone freeze now uses a timed freeze window and may reduce the current phase base LR after unfreeze, but it must never undercut the next phase start LR.
- Cosine scheduling restarts per phase by design and should stay documented when changed.

## Dependencies

The repository currently depends on:
- `torch`
- `numpy`
- `pillow`
- `opencv-python`
- `pytest`

Add new dependencies only when they are justified by the project goal.

## Validation

Before handing off changes, run the narrowest useful checks first.

Examples:

```powershell
.\.venv\Scripts\python.exe train.py --help
.\.venv\Scripts\python.exe predict.py --help
.\.venv\Scripts\python.exe -m pytest tests -v -p no:cacheprovider
```

## Documentation

If you change behavior, update the relevant documents:
- `README.md`
- `Image_Identify_CNN.md`
- `Agent_History/docs/plans/YYYY-MM-DD-*.md` when a design is approved
- `Agent_History/Daily/YYYY/MM/DD_history.md`
- `Agent_History/Monthly/YYYY/MM_summary.md` when writing the month-end summary

Keep instruction documents unchanged unless the project owner explicitly changes project policy.
