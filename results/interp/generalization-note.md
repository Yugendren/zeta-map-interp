# Length-generalization discrepancy investigation

Checkpoint trained on n=13 (source length 26, positions 0..25) reaches 0.9950 exact-match on held-out n=13, but 0% exact-match on n=11,12,14,15,16 (see results/generalization.json). The paper (arXiv:2511.12421) reports evaluating n=11-16 (implying nonzero transfer). This is a bounded (~30 min) investigation into candidate causes, not a fix.

## (a) Positional-embedding-table hypothesis

n=12 source words have natural length 24 (positions 0..23 via the plain `torch.arange` in `_embed`), which is a strict *subrange* of the positions 13 was trained on (0..25). If the model relies on absolute position identity (e.g. "the last position is special", or a position-conditioned lookup table baked into the learned position embeddings) rather than a relative/length-invariant algorithm, shifting the real tokens to occupy a *different* sub-range of position ids should change the exact-match rate -- possibly for the better, if the tail positions the model was trained on carry more informative bias than the untrained head positions.

- `left_aligned_len24_baseline` (real tokens at positions 0..23, array length 24): exact-match = 0.0000
- `left_aligned_padded_len26` (real tokens at positions 0..23, array length 26): exact-match = 0.0000
- `right_aligned_len26` (real tokens at positions 2..25, array length 26): exact-match = 0.0000
- `centered_len26` (real tokens at positions 1..24, array length 26): exact-match = 0.0000

All four positional placements give exactly 0% exact-match, including the right-aligned and centered variants that put the real tokens on position ids the model actually trained on (the tail of 0..25). This is evidence AGAINST the pure "wrong absolute position range" hypothesis in isolation: simply moving n=12 tokens onto trained position ids is not sufficient to recover any exact-match accuracy. The failure more likely reflects the model having learned a genuinely n=13-specific computation (e.g. stage count tied to n, or attention patterns keyed off both position AND content in ways that do not transfer even to nearby n), not simply "wrong position slot."

## (b) Per-token accuracy (partial transfer)

On the same 2000 random n=12 paths (natural left-aligned encoding, no padding tricks), per-position character accuracy between the greedy decode and the true zeta(word) (length mismatches counted as wrong) is **0.5872**, vs 0.0000 exact-sequence match. A random N/E guesser would score ~0.5 on a per-token basis for a task with a roughly balanced label distribution.

58.7% per-token accuracy, well above chance (~50%), IS evidence of partial transfer: the model has learned something about the zeta map that carries beyond n=13, even though it essentially never gets every single token right on a length-24 target sequence (which is a much harder bar -- a model with 95% per-token accuracy independent across ~24 tokens would still have exact-match near 0.95^24 ~ 29%, and errors are almost certainly correlated/compounding through autoregressive decoding, not independent, which pushes exact-match to 0 even faster).

## Conclusion

Both quick probes point the same direction: this is not primarily a positional-embedding-table artifact fixable by re-registering n=12 onto trained position ids, and the failure is not merely a strict-match illusion masking high per-token fidelity. The gap between this reproduction (0% on n != 13) and the paper's reported n=11-16 evaluation most plausibly reflects a training-regime difference not investigated further here within the time budget -- candidates left open: (i) the paper may train on a length-curriculum or a mixture of n rather than n=13 alone, (ii) the paper's reported numbers on n=11-16 could still be well below n=13's ~99.5% (the paper text was not re-derived here, only the qualitative claim that they evaluate that range), and (iii) 1 attention head / 1 layer may simply not have the capacity to represent a length-invariant version of the zeta map's multi-stage scan (recall: the algorithm processes `max(area_word)+1` stages, a quantity that grows with n) without dedicating some of its very limited capacity to n=13-specific shortcuts. No forcing of a positive result was attempted; the honest read is "not conclusively explained within this investigation's scope."
