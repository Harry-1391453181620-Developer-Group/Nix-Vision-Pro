# 2026-05-01 History

## Request

Added Internal Dynamics Stability Indicator tracking from the updated `smart.docx` concept, added live plotting for run metrics, updated docs and ignore rules, installed/verified matplotlib for `.venv312`, and committed the work.

## Work completed

- Added `IDSI` calculation to the torch training metrics path.
- Logged `IDSI` into Omega `epoch_metrics.jsonl` files under `runs/` for each epoch.
- Added `plot.py` for live matplotlib line plots with one marker per epoch.
- Included the requested metric set in the plotter:
  `generalization_gap`, representation variance metrics, `lr`, train/validation accuracy, CE/attractor/total losses, and `IDSI`.
- Updated `README.md` with the new dependency, IDSI metric, and plotting workflow.
- Added `matplotlib` to `requirements.txt`.
- Corrected `.gitignore` so `best_train_commands.txt`, `runs/`, and the `smart.docx` shortcut are local-only artifacts.

## Validation

- Verified `matplotlib` is available through `.venv312`.
- Ran no-bytecode syntax checks for `train.py`, `backends/torch/train_backend.py`, `plot.py`, and `tests/test_training_policies.py`.
- Ran a one-shot `plot.py` smoke render against an existing `epoch_metrics.jsonl`.
- Ran `tests/test_training_policies.py::test_omega_loss_backpropagates_to_projector_and_representation_path`.
