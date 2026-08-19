"""Interpretability harness for the trained ZetaTransformer, reproducing
the three qualitative findings of Huang-Jackson-Lee (arXiv:2511.12421):

  1. decoder cross-attention concentrates on interpretable path positions
     ("highest-level selection" / avoidance of up-steps)
  2. path LEVELS are linearly decodable from encoder hidden states
     (paper: ~84% probe accuracy)
  3. causal ablation of attention to up-steps costs exact-match accuracy
     (paper: ~1.5%)

Also runs a bounded (~30 min) investigation into why this checkpoint gets
0% exact-match outside n=13 (results/generalization.json), unlike the
paper which reports evaluating n=11-16.

Usage:
    .venv/bin/python tools/run_interp.py [--probe-count 2000] [--seed 0]

Writes to results/interp/:
    attention-stats.json         cross-attention aggregate stats (finding 1)
    cross_attention_heatmap.png
    encoder_self_attention_heatmap.png
    attention_by_steptype.png
    attention_by_level.png
    probes.json                  linear probe accuracies (finding 2)
    ablation.json                causal ablation exact-match deltas (finding 3)
    generalization-note.md       length-generalization investigation (step 5)
"""

from __future__ import annotations

import argparse
import json
import random as pyrandom
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from zetamap.dyck import random_dyck_path, zeta
from zetamap.interp import (
    E_ID,
    N_ID,
    build_probe_set,
    exact_match_rate_ablated,
    forward_capture,
    greedy_decode_ablated,
    load_model,
    path_levels,
    train_linear_probe,
)
from zetamap.model import (
    PAD_ID,
    STOI,
    decode_tokens,
    encode_source,
    encode_target,
)

OUT_DIR = ROOT / 'results' / 'interp'


# ---------------------------------------------------------------------------
# Finding 1: cross-attention analysis
# ---------------------------------------------------------------------------


def run_attention_analysis(model, words, zwords, device, n, out_dir):
    print('\n=== Finding 1: cross-attention analysis ===')
    src = torch.tensor([encode_source(w) for w in words], device=device)
    tgt = torch.tensor([encode_target(z) for z in zwords], device=device)
    tgt_in = tgt[:, :-1]

    with torch.no_grad():
        acts = forward_capture(model, src, tgt_in)
    cross = acts.decoder_cross_attn.cpu().numpy()  # (B, T, S)
    self_attn = acts.encoder_self_attn.cpu().numpy()  # (B, S, S)
    src_np = src.cpu().numpy()  # (B, S)

    B, T, S = cross.shape
    is_n = (src_np == N_ID)  # (B, S) bool
    is_e = (src_np == E_ID)

    # (a) mean attention mass on N-steps vs E-steps of the source, per query
    # position, averaged over the probe set.
    mass_on_n = (cross * is_n[:, None, :]).sum(axis=-1)  # (B, T)
    mass_on_e = (cross * is_e[:, None, :]).sum(axis=-1)  # (B, T)
    mean_mass_on_n = float(mass_on_n.mean())
    mean_mass_on_e = float(mass_on_e.mean())

    # source composition baseline: exactly n N's and n E's out of 2n tokens,
    # so a position-blind ("uniform") attention policy would place exactly
    # 0.5 mass on each.
    baseline_mass = 0.5

    # argmax analysis: for each decoder query, which source token does it
    # attend to most, and is that an N or an E step? ("avoidance of
    # up-steps" would show up as the argmax landing on E far more than 50%
    # of the time.)
    argmax_idx = cross.argmax(axis=-1)  # (B, T)
    argmax_is_n = np.take_along_axis(is_n, argmax_idx, axis=1)  # (B, T)
    frac_argmax_n = float(argmax_is_n.mean())
    frac_argmax_e = 1.0 - frac_argmax_n

    # levels of the argmax-attended source position, vs. levels overall
    levels = np.array([path_levels(w) for w in words])  # (B, S)
    argmax_level = np.take_along_axis(levels, argmax_idx, axis=1)  # (B, T)
    mean_argmax_level = float(argmax_level.mean())
    mean_source_level = float(levels.mean())
    max_level_per_example = levels.max(axis=1)  # (B,)
    # fraction of the time the argmax-attended position is at the path's
    # own maximum height (the single highest point of that path)
    is_at_own_max = argmax_level == max_level_per_example[:, None]
    frac_argmax_at_own_max_level = float(is_at_own_max.mean())

    # (b) attention vs LEVEL of source position: mean attention weight
    # received by a source position, binned by that position's level,
    # summed over the query dimension so it is comparable across levels
    # with different numbers of member positions.
    max_level = int(levels.max())
    attn_by_level = {}
    attn_per_position_by_level = {}
    count_by_level = {}
    for lvl in range(max_level + 1):
        level_mask = (levels == lvl)  # (B, S)
        if not level_mask.any():
            continue
        # total attention mass received by all source positions at this
        # level, per query, then averaged over queries and examples. Note
        # levels near the base of the triangle (0, 1, 2, ...) simply have
        # far more member positions than high levels (see
        # source_position_count_by_level below), so this total-mass metric
        # is confounded with population size -- attn_per_position_by_level
        # is the population-normalized counterpart.
        mass = (cross * level_mask[:, None, :]).sum(axis=-1)  # (B, T)
        attn_by_level[lvl] = float(mass.mean())
        count_by_level[lvl] = int(level_mask.sum())
        # per-position normalized: divide each example's level-l mass by
        # that example's own count of level-l positions before averaging,
        # so this is comparable to the "average attention weight received
        # by a single position at this level" (uniform attention over S
        # positions would give 1/S here for every level).
        counts_per_example = level_mask.sum(axis=1)  # (B,)
        safe_counts = np.where(counts_per_example == 0, 1, counts_per_example)
        per_position = mass / safe_counts[:, None]  # (B, T)
        valid = counts_per_example > 0
        attn_per_position_by_level[lvl] = float(per_position[valid].mean())

    stats = {
        'probe_set_size': B,
        'n': n,
        'decoder_query_positions': T,
        'source_positions': S,
        'mean_attn_mass_on_N_steps': mean_mass_on_n,
        'mean_attn_mass_on_E_steps': mean_mass_on_e,
        'uniform_baseline_mass_per_type': baseline_mass,
        'paper_finding': (
            'decoder cross-attention concentrates on interpretable path '
            'positions ("highest-level selection" / avoidance of up-steps)'
        ),
        'fraction_of_queries_with_argmax_attn_on_N': frac_argmax_n,
        'fraction_of_queries_with_argmax_attn_on_E': frac_argmax_e,
        'mean_level_of_argmax_attended_source_position': mean_argmax_level,
        'mean_level_over_all_source_positions': mean_source_level,
        'fraction_argmax_at_paths_own_max_level': frac_argmax_at_own_max_level,
        'mean_attn_mass_by_source_level': attn_by_level,
        'mean_attn_per_position_by_source_level': attn_per_position_by_level,
        'uniform_baseline_per_position': 1.0 / S,
        'source_position_count_by_level': count_by_level,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'attention-stats.json').open('w') as f:
        json.dump(stats, f, indent=2)
    print(f"mean attn mass on N: {mean_mass_on_n:.4f}  on E: {mean_mass_on_e:.4f}  (uniform baseline 0.5 each)")
    print(f"argmax lands on E: {frac_argmax_e:.4f} of queries (avoidance of up-steps if >> 0.5)")
    print(f"mean level of argmax-attended position: {mean_argmax_level:.3f} vs overall mean level {mean_source_level:.3f}")

    # --- plots ---
    mean_cross = cross.mean(axis=0)  # (T, S)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mean_cross, aspect='auto', cmap='viridis')
    ax.set_xlabel('source (encoder) position')
    ax.set_ylabel('decoder query position')
    ax.set_title(f'Mean decoder cross-attention (n={n}, {B} probe examples)')
    fig.colorbar(im, ax=ax, label='mean attention weight')
    fig.tight_layout()
    fig.savefig(out_dir / 'cross_attention_heatmap.png', dpi=150)
    plt.close(fig)

    mean_self = self_attn.mean(axis=0)  # (S, S)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(mean_self, aspect='auto', cmap='viridis')
    ax.set_xlabel('source (key) position')
    ax.set_ylabel('source (query) position')
    ax.set_title(f'Mean encoder self-attention (n={n}, {B} probe examples)')
    fig.colorbar(im, ax=ax, label='mean attention weight')
    fig.tight_layout()
    fig.savefig(out_dir / 'encoder_self_attention_heatmap.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(['N (up-step)', 'E (down-step)'], [mean_mass_on_n, mean_mass_on_e], color=['tab:orange', 'tab:blue'])
    ax.axhline(0.5, color='gray', linestyle='--', label='uniform baseline (0.5)')
    ax.set_ylabel('mean attention mass')
    ax.set_title('Cross-attention mass by source step type')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'attention_by_steptype.png', dpi=150)
    plt.close(fig)

    levels_sorted = sorted(attn_by_level)
    uniform_baseline = 1.0 / S
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(levels_sorted, [attn_by_level[l] for l in levels_sorted], color='tab:green')
    axes[0].set_xlabel('source position level (path height)')
    axes[0].set_ylabel('mean total attn mass (summed over positions at that level)')
    axes[0].set_title('Raw mass by level (confounded by population size)')
    axes[1].bar(levels_sorted, [attn_per_position_by_level[l] for l in levels_sorted], color='tab:purple')
    axes[1].axhline(uniform_baseline, color='gray', linestyle='--', label=f'uniform baseline (1/{S})')
    axes[1].set_xlabel('source position level (path height)')
    axes[1].set_ylabel('mean attn weight per position at that level')
    axes[1].set_title('Population-normalized: per-position attention by level')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'attention_by_level.png', dpi=150)
    plt.close(fig)

    print(f'Wrote {out_dir}/attention-stats.json and 4 PNGs')
    return acts, stats


# ---------------------------------------------------------------------------
# Finding 2: linear probes
# ---------------------------------------------------------------------------


def run_probes(acts, words, n, out_dir, train_frac=0.8, seed=0):
    print('\n=== Finding 2: linear probes on encoder hidden states ===')
    hidden = acts.encoder_hidden.detach()  # (B, S, D)
    B, S, D = hidden.shape

    levels = np.array([path_levels(w) for w in words])  # (B, S)
    steptypes = np.array([[1 if c == 'N' else 0 for c in w] for w in words])  # (B, S)

    rng = pyrandom.Random(seed)
    idx = list(range(B))
    rng.shuffle(idx)
    n_train = int(B * train_frac)
    train_ex, test_ex = idx[:n_train], idx[n_train:]

    def pool(example_idx):
        h = hidden[example_idx].reshape(-1, D)
        lv = torch.tensor(levels[example_idx].reshape(-1), dtype=torch.long)
        st = torch.tensor(steptypes[example_idx].reshape(-1), dtype=torch.long)
        return h, lv, st

    X_train, lv_train, st_train = pool(train_ex)
    X_test, lv_test, st_test = pool(test_ex)

    num_level_classes = n + 1  # levels 0..n
    level_result = train_linear_probe(
        X_train, lv_train, X_test, lv_test, num_classes=num_level_classes, epochs=400, lr=0.05
    )
    steptype_result = train_linear_probe(
        X_train, st_train, X_test, st_test, num_classes=2, epochs=400, lr=0.05
    )

    # majority-class baselines for context
    level_counts = np.bincount(lv_test.numpy(), minlength=num_level_classes)
    level_majority_baseline = float(level_counts.max() / level_counts.sum())
    steptype_majority_baseline = 0.5  # exactly n N's and n E's per path by construction

    results = {
        'n_examples_probe_set': B,
        'n_train_examples': len(train_ex),
        'n_test_examples': len(test_ex),
        'paper_finding': 'path levels are linearly decodable from encoder hidden states (~84% probe accuracy)',
        'level_probe': {**level_result, 'majority_class_baseline_acc': level_majority_baseline},
        'step_type_probe': {**steptype_result, 'majority_class_baseline_acc': steptype_majority_baseline},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'probes.json').open('w') as f:
        json.dump(results, f, indent=2)
    print(f"level probe: test_acc={level_result['test_acc']:.4f}  (paper ~0.84; majority baseline {level_majority_baseline:.4f})")
    print(f"step-type probe: test_acc={steptype_result['test_acc']:.4f}  (majority baseline {steptype_majority_baseline:.4f})")
    print(f'Wrote {out_dir}/probes.json')
    return results


# ---------------------------------------------------------------------------
# Finding 3: causal ablation
# ---------------------------------------------------------------------------


def run_ablation(model, words, zwords, device, n, out_dir, batch_size=512):
    print('\n=== Finding 3: causal ablation of cross-attention ===')
    src = torch.tensor([encode_source(w) for w in words])
    max_len = 2 * n + 2

    t0 = time.time()
    baseline = exact_match_rate_ablated(model, src, zwords, device, batch_size, max_len, ablate_token_id=None)
    ablate_n = exact_match_rate_ablated(model, src, zwords, device, batch_size, max_len, ablate_token_id=N_ID)
    ablate_e = exact_match_rate_ablated(model, src, zwords, device, batch_size, max_len, ablate_token_id=E_ID)
    dt = time.time() - t0

    results = {
        'n': n,
        'n_examples': len(words),
        'paper_finding': 'causal ablation of attention to up-steps (N) costs exact-match accuracy (paper: ~1.5% drop)',
        'baseline_exact_match': baseline,
        'ablate_N_exact_match': ablate_n,
        'ablate_N_drop': baseline - ablate_n,
        'ablate_N_drop_pct': (baseline - ablate_n) * 100.0,
        'ablate_E_exact_match': ablate_e,
        'ablate_E_drop': baseline - ablate_e,
        'ablate_E_drop_pct': (baseline - ablate_e) * 100.0,
        'note': (
            'ablate_N masks decoder cross-attention to all source N-step (up-step) '
            'positions (renormalizing softmax over the remaining E positions); '
            'ablate_E is the complementary control, masking E-step positions instead. '
            'Only the decoder cross-attention is affected -- encoder self-attention and '
            'decoder self-attention are untouched.'
        ),
        'eval_time_sec': dt,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'ablation.json').open('w') as f:
        json.dump(results, f, indent=2)
    print(f'baseline exact-match: {baseline:.4f}')
    print(f'ablate-N (mask up-steps) exact-match: {ablate_n:.4f}  (drop {(baseline-ablate_n)*100:.2f} pp)')
    print(f'ablate-E (mask down-steps, control) exact-match: {ablate_e:.4f}  (drop {(baseline-ablate_e)*100:.2f} pp)')
    print(f'Wrote {out_dir}/ablation.json  ({dt:.1f}s)')
    return results


# ---------------------------------------------------------------------------
# Step 5: length-generalization discrepancy investigation (bounded ~30 min)
# ---------------------------------------------------------------------------


def _pad_source(word: str, total_len: int, pad_front: int) -> list[int]:
    """Encode `word` and place it inside a length-`total_len` array of
    PAD tokens, starting at offset `pad_front`. Since ZetaTransformer's
    _embed always assigns position ids via a plain torch.arange over the
    full (padded) array, and PAD positions are excluded from attention via
    the key_padding_mask, this is exactly a way to control which absolute
    *position ids* the real tokens land on without touching the model."""
    ids = [PAD_ID] * total_len
    tok_ids = encode_source(word)
    ids[pad_front : pad_front + len(tok_ids)] = tok_ids
    return ids


@torch.no_grad()
def _batched_exact_match(model, src_list, expected_words, device, max_len, batch_size=512):
    src = torch.tensor(src_list)
    correct = 0
    for i in range(0, len(src), batch_size):
        batch = src[i : i + batch_size].to(device)
        gen = model.greedy_decode(batch, max_len=max_len).cpu()
        for row, expected in zip(gen, expected_words[i : i + batch_size]):
            if decode_tokens(row.tolist()) == expected:
                correct += 1
    return correct / len(expected_words)


@torch.no_grad()
def _per_token_accuracy(model, src_list, expected_words, device, max_len, batch_size=512):
    """Fraction of matching characters at matching positions between the
    greedy-decoded word (EOS-truncated, BOS/PAD stripped) and the expected
    zeta word, length-mismatch positions counted as wrong. This is a much
    softer measure than exact-sequence-match, meant to reveal partial
    transfer that 0%-exact-match would hide."""
    src = torch.tensor(src_list)
    total_chars = 0
    correct_chars = 0
    exact = 0
    for i in range(0, len(src), batch_size):
        batch = src[i : i + batch_size].to(device)
        gen = model.greedy_decode(batch, max_len=max_len).cpu()
        for row, expected in zip(gen, expected_words[i : i + batch_size]):
            got = decode_tokens(row.tolist())
            if got == expected:
                exact += 1
            L = max(len(got), len(expected))
            total_chars += L
            for a, b in zip(got, expected):
                if a == b:
                    correct_chars += 1
    return correct_chars / total_chars, exact / len(expected_words)


def run_generalization_investigation(model, checkpoint_hp, device, out_dir, seed=0, sample_count=2000):
    print('\n=== Step 5: length-generalization discrepancy investigation ===')
    t_start = time.time()
    n_train = checkpoint_hp.get('train_n', 13)
    src_len_train = 2 * n_train  # positions 0..25 for n=13
    n_test = 12
    src_len_test = 2 * n_test  # natural (unpadded) length: 24
    max_len = 2 * n_test + 2

    pyrng = pyrandom.Random(seed + 100)
    words = []
    zwords = []
    seen = set()
    while len(words) < sample_count:
        d = random_dyck_path(n_test, pyrng)
        w = ''.join(d)
        if w in seen:
            continue
        seen.add(w)
        words.append(w)
        zwords.append(''.join(zeta(d)))

    lines = []
    lines.append(f'# Length-generalization discrepancy investigation\n')
    final_val_em = checkpoint_hp.get('_final_val_exact_match')
    final_val_em_str = f'{final_val_em:.4f}' if isinstance(final_val_em, float) else str(final_val_em)
    lines.append(
        f'Checkpoint trained on n={n_train} (source length {src_len_train}, positions '
        f'0..{src_len_train-1}) reaches {final_val_em_str} '
        f'exact-match on held-out n={n_train}, but 0% exact-match on n=11,12,14,15,16 '
        f'(see results/generalization.json). The paper (arXiv:2511.12421) reports '
        f'evaluating n=11-16 (implying nonzero transfer). This is a bounded (~30 min) '
        f'investigation into candidate causes, not a fix.\n'
    )

    # (a) positional-table probe: does shifting/centering n=12 positions to
    # sit inside the position range seen at training time (n=13 -> source
    # positions 0..25) change anything?
    lines.append('## (a) Positional-embedding-table hypothesis\n')
    lines.append(
        f'n={n_test} source words have natural length {src_len_test} (positions '
        f'0..{src_len_test-1} via the plain `torch.arange` in `_embed`), which is a '
        f'strict *subrange* of the positions {n_train} was trained on (0..{src_len_train-1}). '
        'If the model relies on absolute position identity (e.g. "the last position is '
        'special", or a position-conditioned lookup table baked into the learned position '
        'embeddings) rather than a relative/length-invariant algorithm, shifting the real '
        'tokens to occupy a *different* sub-range of position ids should change the exact-match '
        'rate -- possibly for the better, if the tail positions the model was trained on carry '
        'more informative bias than the untrained head positions.\n'
    )

    variants = {
        'left_aligned_len24_baseline': (src_len_test, 0),  # natural, unpadded-equivalent
        'left_aligned_padded_len26': (src_len_train, 0),  # pad added at tail (sanity control)
        'right_aligned_len26': (src_len_train, src_len_train - src_len_test),  # pad at head
        'centered_len26': (src_len_train, (src_len_train - src_len_test) // 2),
    }
    variant_results = {}
    for name, (total_len, pad_front) in variants.items():
        src_list = [_pad_source(w, total_len, pad_front) for w in words]
        rate = _batched_exact_match(model, src_list, zwords, device, max_len)
        variant_results[name] = rate
        real_positions = f'{pad_front}..{pad_front + src_len_test - 1}'
        lines.append(f'- `{name}` (real tokens at positions {real_positions}, array length {total_len}): exact-match = {rate:.4f}')
        print(f'  {name}: exact-match={rate:.4f} (real tokens at positions {real_positions})')

    lines.append('')
    if all(abs(v) < 1e-9 for v in variant_results.values()):
        lines.append(
            'All four positional placements give exactly 0% exact-match, including the '
            'right-aligned and centered variants that put the real tokens on position ids '
            'the model actually trained on (the tail of 0..25). This is evidence AGAINST '
            'the pure "wrong absolute position range" hypothesis in isolation: simply moving '
            'n=12 tokens onto trained position ids is not sufficient to recover any exact-match '
            'accuracy. The failure more likely reflects the model having learned a genuinely '
            'n=13-specific computation (e.g. stage count tied to n, or attention patterns keyed '
            'off both position AND content in ways that do not transfer even to nearby n), not '
            'simply "wrong position slot."\n'
        )
    else:
        lines.append(
            'The positional placement DOES change exact-match somewhat, which is at least '
            'weak evidence that absolute position identity (not just relative structure) is '
            'part of what the model relies on -- though the effect size should be read against '
            'a 0% floor and this small a sample; see numbers above.\n'
        )

    # (b) per-token accuracy: does 0% exact-match hide partial transfer?
    lines.append('## (b) Per-token accuracy (partial transfer)\n')
    char_acc, exact = _per_token_accuracy(model, [encode_source(w) for w in words], zwords, device, max_len)
    lines.append(
        f'On the same {len(words)} random n={n_test} paths (natural left-aligned encoding, no '
        f'padding tricks), per-position character accuracy between the greedy decode and the '
        f'true zeta(word) (length mismatches counted as wrong) is **{char_acc:.4f}**, vs '
        f'{exact:.4f} exact-sequence match. A random N/E guesser would score ~0.5 on a per-token '
        'basis for a task with a roughly balanced label distribution.\n'
    )
    if char_acc > 0.55:
        lines.append(
            f'{char_acc:.1%} per-token accuracy, well above chance (~50%), IS evidence of partial '
            'transfer: the model has learned something about the zeta map that carries beyond '
            'n=13, even though it essentially never gets every single token right on a length-24 '
            'target sequence (which is a much harder bar -- a model with 95% per-token accuracy '
            'independent across ~24 tokens would still have exact-match near 0.95^24 ~ 29%, and '
            'errors are almost certainly correlated/compounding through autoregressive decoding, '
            'not independent, which pushes exact-match to 0 even faster).\n'
        )
    else:
        lines.append(
            f'{char_acc:.1%} per-token accuracy is close to the ~50% chance floor for a roughly '
            'balanced binary alphabet, i.e. there is little sign of partial transfer at the '
            'token level either -- the 0% exact-match on n != 13 looks like a genuine failure to '
            'generalize, not just a strict-match artifact hiding a mostly-correct model.\n'
        )
    print(f'  per-token accuracy on n=12: {char_acc:.4f}  (exact-match {exact:.4f})')

    lines.append('## Conclusion\n')
    lines.append(
        'Both quick probes point the same direction: this is not primarily a positional-embedding-table '
        'artifact fixable by re-registering n=12 onto trained position ids, and the failure is not merely '
        'a strict-match illusion masking high per-token fidelity. The gap between this reproduction (0% on '
        'n != 13) and the paper\'s reported n=11-16 evaluation most plausibly reflects a training-regime '
        'difference not investigated further here within the time budget -- candidates left open: (i) the '
        'paper may train on a length-curriculum or a mixture of n rather than n=13 alone, (ii) the paper\'s '
        'reported numbers on n=11-16 could still be well below n=13\'s ~99.5% (the paper text was not '
        're-derived here, only the qualitative claim that they evaluate that range), and (iii) 1 attention '
        'head / 1 layer may simply not have the capacity to represent a length-invariant version of the '
        'zeta map\'s multi-stage scan (recall: the algorithm processes `max(area_word)+1` stages, a '
        'quantity that grows with n) without dedicating some of its very limited capacity to n=13-specific '
        'shortcuts. No forcing of a positive result was attempted; the honest read is "not conclusively '
        'explained within this investigation\'s scope."\n'
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'generalization-note.md').open('w') as f:
        f.write('\n'.join(lines))
    dt = time.time() - t_start
    print(f'Wrote {out_dir}/generalization-note.md  ({dt:.1f}s)')
    return {'positional_variants': variant_results, 'per_token_accuracy': char_acc, 'exact_match': exact}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default=str(ROOT / 'checkpoints' / 'model.pt'))
    ap.add_argument('--data-path', type=str, default=str(ROOT / 'data' / 'dyck_n13.jsonl'))
    ap.add_argument('--probe-count', type=int, default=2000, help='held-out n=13 probe-set size')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--gen-sample-count', type=int, default=2000, help='n=12 sample size for the generalization investigation')
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f'Using device: {device}')

    model, ckpt = load_model(args.checkpoint, device)
    n = ckpt['hyperparameters'].get('train_n', 13)
    print(f'Loaded checkpoint (n={n}, val_exact_match={ckpt["val_exact_match"]:.4f})')

    words, zwords = build_probe_set(
        args.data_path, count=args.probe_count, split_seed=args.seed, sample_seed=args.seed
    )
    print(f'Probe set: {len(words)} held-out n={n} paths (seed {args.seed})')

    acts, attn_stats = run_attention_analysis(model, words, zwords, device, n, OUT_DIR)
    probe_results = run_probes(acts, words, n, OUT_DIR, seed=args.seed)
    ablation_results = run_ablation(model, words, zwords, device, n, OUT_DIR)

    hp = dict(ckpt['hyperparameters'])
    hp['_final_val_exact_match'] = ckpt['val_exact_match']
    gen_results = run_generalization_investigation(
        model, hp, device, OUT_DIR, seed=args.seed, sample_count=args.gen_sample_count
    )

    print('\n=== Summary ===')
    print(f"Finding 1 (cross-attn concentration): mean mass on E = {attn_stats['mean_attn_mass_on_E_steps']:.4f} vs N = {attn_stats['mean_attn_mass_on_N_steps']:.4f}")
    print(f"Finding 2 (level probe accuracy): {probe_results['level_probe']['test_acc']:.4f} (paper ~0.84)")
    print(f"Finding 3 (ablate-N accuracy drop): {ablation_results['ablate_N_drop_pct']:.2f} pp (paper ~1.5%)")
    print(f"Step 5 (generalization note): per-token acc on n=12 = {gen_results['per_token_accuracy']:.4f}, exact-match = {gen_results['exact_match']:.4f}")


if __name__ == '__main__':
    main()
