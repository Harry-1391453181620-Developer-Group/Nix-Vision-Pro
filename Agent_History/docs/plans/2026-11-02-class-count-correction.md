# 2026-11-02 Class-Count Correction

## What was wrong

Earlier notes treated the CE linear control as a 17-class experiment and later
implied the old 93.6% MAOIDL model was 18-class.  Both were wrong.

Verified facts (from run `config.json` files and the user's correction):

- every MAOL run on the current dataset is **18 classes**;
- the old 93.6% pure-CE MAOIDL model was **12 classes** with a
  token-transformer head.

## Consequence

- `93.6%` (12 classes, random baseline 8.3%) and `90.08%` (18 classes, random
  baseline 5.6%) are **not comparable**; the class-count difference is not
  removable from the historical number.
- The valid comparison is 18-class MAOL vs 18-class CE linear control, both with
  `--ema`, same backbone, data, augmentation, epochs and LR.
- Lambdas from `runs/20261102-015422-maol-probe-seed_42` are unchanged:
  `intra=1.0, sep=2.0, nest=1.0, cov=0.5, center=0.5`.

## Files updated

- `README.md`: CE control section states the 18-class setup and the 12-class
  historical baseline.
- `Agent_History/Daily/2026/11/01_history.md`: notes corrected.
- `Agent_History/Monthly/2026/11_summary.md`: 18-class control, corrected
  comparison guidance.
- `Agent_History/Monthly/2026/10_summary.md`: 93.6% labelled 12-class; next-work
  comparison updated.
- `best_train_commands.txt`: CE control comment updated to 18-class.
