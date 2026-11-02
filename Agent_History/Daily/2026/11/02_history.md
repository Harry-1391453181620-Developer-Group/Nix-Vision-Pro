# 2026-11-02

## Task

Correct the class-count facts used in the MAOL vs CE-control comparison.

## Correction

- All MAOL training runs are **18 classes**.
- The old 93.6% MAOIDL model was **12 classes** and used a token-transformer head.
- My earlier notes wrongly called the CE control "17-class" and then wrongly called
  the old baseline "18-class".

## Consequences

- 93.6% (12 classes) and 90.08% (18 classes) are not comparable; the class-count
  difference cannot be separated from the head/architecture difference.
- The valid A/B is 18-class MAOL vs 18-class CE linear, both with `--ema`.
- Lambdas from `runs/20261102-015422-maol-probe-seed_42` stay
  `intra=1.0, sep=2.0, nest=1.0, cov=0.5, center=0.5`.

## Files Updated

- `README.md`
- `Agent_History/Daily/2026/11/01_history.md`
- `Agent_History/Monthly/2026/11_summary.md`
- `Agent_History/Monthly/2026/10_summary.md`
- `best_train_commands.txt`
- `Agent_History/docs/plans/2026-11-02-class-count-correction.md` (new)

## Validation

- Documentation-only change; no code modified.
- `python -m pytest tests -q -p no:cacheprovider`: still 38 passed.
