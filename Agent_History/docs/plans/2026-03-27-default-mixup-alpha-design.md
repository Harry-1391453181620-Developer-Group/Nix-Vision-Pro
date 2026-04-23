# 2026-03-27 Default MixUp Alpha Design

## Goal

Change both training backends so MixUp is enabled by default and expose the Beta parameter as a CLI flag.

## Approved Policy

- `--mixup` defaults to enabled.
- `--mixup-alpha` is added with default `0.2`.
- `--mixup-prob` keeps default `0.5`.
- If MixUp is active for the run, focal loss remains disabled automatically and the trainers report cross-entropy-based train and validation loss.

## Implementation Scope

- Add shared validation for `mixup_alpha` in `utils/training.py`.
- Update both backend parsers and both MixUp call sites to consume the validated CLI value.
- Update tests so invalid alpha values fail fast.
- Update user-facing docs and example commands to show the new defaults.

## Validation

- `python -m py_compile` on the touched modules.
- Targeted `pytest` coverage for MixUp helper validation and policy behavior.
