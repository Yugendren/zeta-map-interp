# zeta-map-interp

Can a small transformer learn the zeta map on Dyck paths, and does what it learns generalise across path length?

- `src/zetamap/dyck.py` — Dyck path generation and the zeta map (winning reading convention found by exhaustive search over an 8-way family; see the docstring in `model.py`).
- `src/zetamap/model.py` — the transformer.
- `src/zetamap/interp.py` — interpretability probes.
- `results/train-metrics.json`, `results/generalization.json` — training curves and the length-generalisation sweep.
- `checkpoints/model.pt` — trained weights.

## Result

Held-out validation at the training length (n = 13): 99.5% exact match.
Every other length tested (n = 11, 12 exhaustive; n = 14, 15, 16 sampled): 0% exact match.

The model fits the training-length distribution and does not learn the map. This is recorded as a negative result on length generalisation; see `results/generalization.json` for the exact counts.

`make` runs training and evaluation; `tests/` holds the unit tests.
