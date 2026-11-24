# Full optimization report

## Protocol

- Model: XGBoost with 144 leakage-safe network features
- Objective: maximize validation F1 at threshold 0.5
- Budget: 32 objective calls per optimizer
- Selection data: train 2015–2021, validation 2022
- Test data: January–June 2023, sealed until the winner was selected
- Test was evaluated once after refitting the winner on train + validation

## Optimizer comparison

| Rank | Optimizer | Best validation F1 | Model fits | Cache hits | Time (min) |
|---:|---|---:|---:|---:|---:|
| 1 | Bayesian/TPE | 0.433229 | 32 | 0 | 9.19 |
| 2 | Simulated Annealing | 0.431350 | 32 | 0 | 8.56 |
| 3 | Genetic Algorithm | 0.430916 | 32 | 0 | 10.29 |
| 4 | Tabu Search | 0.429512 | 25 | 7 | 6.90 |
| 5 | Hill Climbing | 0.428913 | 25 | 7 | 11.98 |
| 6 | Particle Swarm | 0.427076 | 30 | 2 | 14.52 |
| 7 | Random Search | 0.425455 | 32 | 0 | 18.74 |
| 8 | Grid Search | 0.423994 | 32 | 0 | 20.03 |
| 9 | SciPy/Nelder-Mead (SOpt) | 0.420976 | 32 | 0 | 9.32 |

Total measured optimizer runtime: 6571.645 seconds (109.5 minutes).

## Validation-selected winner

Bayesian/TPE selected:

```text
max_depth          = 5
learning_rate      = 0.1780156469
min_child_weight   = 5.6048389722
subsample          = 0.9438582034
scale_pos_weight   = 4.3965848136
```

Validation F1: **0.433229**.

## One-time final test evaluation

The winner was refitted on 278,393 train + validation records and evaluated once
on 22,339 test records.

| Metric | Score |
|---|---:|
| Precision | 0.308014 |
| Recall | 0.681596 |
| F1 | 0.424291 |
| ROC-AUC | 0.745772 |
| Accuracy | 0.678320 |

Confusion matrix (`[[TN, FP], [FN, TP]]`):

```text
[[12505, 5949],
 [ 1237, 2648]]
```

The optimized model detects 68.16% of delayed flights. Precision remains modest
because it predicts delay for 38.48% of flights while the observed delayed rate is
17.39%. The threshold was intentionally fixed at 0.5 throughout optimization.

## Verification

- Test was not available to any optimizer.
- Saved predictions independently reproduce the stored metrics.
- Preprocessor, XGBoost sklearn model, and raw Booster reload successfully.
- All 11 automated tests pass.
- `pip check` reports no broken requirements.
