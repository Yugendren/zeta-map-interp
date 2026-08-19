"""Interpretability harness for the ZetaTransformer.

Provides the mechanics needed to reproduce the three qualitative findings of
Huang-Jackson-Lee (arXiv:2511.12421) on the learned zeta map:

  1. decoder cross-attention concentrates on interpretable path positions
     ("highest-level selection" / avoidance of up-steps)
  2. path LEVELS (running height of the Dyck path) are linearly decodable
     from encoder hidden states
  3. causally ablating decoder cross-attention to up-steps (N-steps) costs
     exact-match accuracy

This module holds the reusable mechanics (activation capture via hooks,
causal ablation via cross-attention masking, linear probe training, path
level computation); tools/run_interp.py wires these into the end-to-end
pipeline that writes results/interp/*.

A note on *why* hooks alone don't work for attention capture: PyTorch's
nn.TransformerEncoderLayer / nn.TransformerDecoderLayer internally call
their self_attn / multihead_attn sub-modules with `need_weights=False`
(see `_sa_block` / `_mha_block` in torch's source), so that the fused
scaled-dot-product-attention fast path can be used. A plain forward hook
on those sub-modules would therefore only ever observe `None` for the
attention weights. `capture_activations` below works around this by
monkeypatching the *block* methods (not the attention module itself) on
the single encoder/decoder layer instance, for the duration of one
`with` block, to force `need_weights=True` and stash the result.
"""

from __future__ import annotations

import contextlib
import json
import random as pyrandom
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zetamap.model import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    STOI,
    ZetaTransformer,
    decode_tokens,
)

N_ID = STOI['N']
E_ID = STOI['E']


# ---------------------------------------------------------------------------
# Path levels (running height at each step)
# ---------------------------------------------------------------------------


def path_levels(word: str) -> list[int]:
    """Height of the path just before each step is taken (0-indexed): h
    starts at 0, and each step (N or E) is recorded at the height it is
    taken from, before h is updated. Same convention as
    zetamap.dyck.area_word, but defined at every position (not just
    N-step positions), since here we want a LEVEL label for every source
    token, N or E. For a Dyck path of order n, levels range over
    0, ..., n (n+1 possible values: an N step is always taken from a
    height in 0..n-1, an E step from a height in 1..n).
    """
    levels = []
    h = 0
    for c in word:
        levels.append(h)
        h += 1 if c == 'N' else -1
    return levels


# ---------------------------------------------------------------------------
# Held-out probe set: reproduces tools/train.py's own val split so the
# probe set is genuinely held out from training, not just a fresh sample
# from the same exhaustive n=13 file that could overlap the 300K training
# subset.
# ---------------------------------------------------------------------------


def load_val_split(
    data_path: str | Path, val_frac: float = 0.05, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Reproduce exactly the train/val index split tools/train.py uses
    (np.random.RandomState(seed).permutation, val_frac held out as the
    first slice), so the returned (words, zeta_words) are the same
    examples the trained checkpoint never trained on."""
    words, zeta_words = [], []
    with open(data_path) as f:
        for line in f:
            rec = json.loads(line)
            words.append(rec['word'])
            zeta_words.append(rec['zeta_word'])
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(words))
    n_val = int(len(words) * val_frac)
    val_idx = idx[:n_val]
    return [words[i] for i in val_idx], [zeta_words[i] for i in val_idx]


def build_probe_set(
    data_path: str | Path,
    count: int = 2000,
    val_frac: float = 0.05,
    split_seed: int = 0,
    sample_seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Sample `count` examples (without replacement) from the held-out val
    split, seeded for reproducibility."""
    val_words, val_zeta = load_val_split(data_path, val_frac=val_frac, seed=split_seed)
    rng = pyrandom.Random(sample_seed)
    idx = list(range(len(val_words)))
    rng.shuffle(idx)
    idx = idx[:count]
    return [val_words[i] for i in idx], [val_zeta[i] for i in idx]


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------


@dataclass
class Activations:
    encoder_hidden: torch.Tensor  # (B, S, D) -- post encoder-layer hidden states
    encoder_self_attn: torch.Tensor  # (B, S, S) -- encoder self-attention
    decoder_cross_attn: torch.Tensor  # (B, T, S) -- decoder cross-attention
    logits: torch.Tensor  # (B, T, V)


class _AttnRecorder:
    """Tiny mutable box an attn-weight-recording closure writes into."""

    def __init__(self):
        self.weights: torch.Tensor | None = None


def _make_patched_sa_block(layer: nn.TransformerEncoderLayer, recorder: _AttnRecorder):
    def sa_block(x, attn_mask, key_padding_mask, is_causal=False):
        out, weights = layer.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
            is_causal=is_causal,
        )
        recorder.weights = weights
        return layer.dropout1(out)

    return sa_block


def _make_patched_mha_block(layer: nn.TransformerDecoderLayer, recorder: _AttnRecorder):
    def mha_block(x, mem, attn_mask, key_padding_mask, is_causal=False):
        out, weights = layer.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
            is_causal=is_causal,
        )
        recorder.weights = weights
        return layer.dropout2(out)

    return mha_block


@contextlib.contextmanager
def capture_activations(model: ZetaTransformer):
    """Context manager that patches the model's (single) encoder layer and
    decoder layer to record self-/cross-attention weights, and hooks the
    encoder's output to record hidden states, for the duration of the
    `with` block. Use it around exactly one forward pass, e.g.:

        with capture_activations(model) as rec:
            logits = model(src_ids, tgt_in_ids)
        attn = rec['decoder_cross_attn'].weights  # (B, T, S)

    Assumes 1 encoder layer / 1 decoder layer (the paper's architecture,
    and this repo's checkpoint); asserts this to fail loudly otherwise.
    """
    assert len(model.encoder.layers) == 1, 'capture_activations assumes 1 encoder layer'
    assert len(model.decoder.layers) == 1, 'capture_activations assumes 1 decoder layer'
    enc_layer = model.encoder.layers[0]
    dec_layer = model.decoder.layers[0]

    self_attn_recorder = _AttnRecorder()
    cross_attn_recorder = _AttnRecorder()
    hidden_box = {'value': None}

    def hidden_hook(module, inputs, output):
        hidden_box['value'] = output

    enc_layer._sa_block = _make_patched_sa_block(enc_layer, self_attn_recorder)
    dec_layer._mha_block = _make_patched_mha_block(dec_layer, cross_attn_recorder)
    handle = model.encoder.register_forward_hook(hidden_hook)

    try:
        yield {
            'encoder_self_attn': self_attn_recorder,
            'decoder_cross_attn': cross_attn_recorder,
            'encoder_hidden': hidden_box,
        }
    finally:
        del enc_layer._sa_block
        del dec_layer._mha_block
        handle.remove()


def forward_capture(
    model: ZetaTransformer, src_ids: torch.Tensor, tgt_in_ids: torch.Tensor
) -> Activations:
    """Teacher-forced forward pass with full activation capture."""
    with capture_activations(model) as rec:
        logits = model(src_ids, tgt_in_ids)
    return Activations(
        encoder_hidden=rec['encoder_hidden']['value'],
        encoder_self_attn=rec['encoder_self_attn'].weights,
        decoder_cross_attn=rec['decoder_cross_attn'].weights,
        logits=logits,
    )


# ---------------------------------------------------------------------------
# Causal ablation of decoder cross-attention
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_decode_ablated(
    model: ZetaTransformer,
    src_ids: torch.Tensor,
    ablate_token_id: int | None,
    max_len: int | None = None,
) -> torch.Tensor:
    """Greedy decode identical to ZetaTransformer.greedy_decode, except the
    decoder's cross-attention is additionally forbidden from attending to
    any source position whose token id is `ablate_token_id` (achieved by
    folding that condition into memory_key_padding_mask, which only
    affects the decoder's cross-attention -- it leaves encoder
    self-attention and decoder self-attention untouched). If
    `ablate_token_id` is None, behaves exactly like the unablated model.
    """
    model.eval()
    max_len = max_len or model.max_len
    device = src_ids.device
    batch_size = src_ids.size(0)
    memory, base_padding_mask = model.encode(src_ids)
    if ablate_token_id is None:
        combined_mask = base_padding_mask
    else:
        combined_mask = base_padding_mask | (src_ids == ablate_token_id)

    generated = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        logits = model.decode_step(generated, memory, combined_mask)
        next_ids = logits[:, -1, :].argmax(dim=-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, PAD_ID), next_ids)
        generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)
        finished = finished | (next_ids == EOS_ID)
        if bool(finished.all()):
            break
    return generated


@torch.no_grad()
def exact_match_rate_ablated(
    model: ZetaTransformer,
    src: torch.Tensor,
    expected_words: list[str],
    device: torch.device,
    batch_size: int,
    max_len: int,
    ablate_token_id: int | None,
) -> float:
    model.eval()
    correct = 0
    total = len(expected_words)
    for i in range(0, total, batch_size):
        batch_src = src[i : i + batch_size].to(device)
        gen = greedy_decode_ablated(model, batch_src, ablate_token_id, max_len=max_len)
        gen = gen.cpu()
        for row, expected in zip(gen, expected_words[i : i + batch_size]):
            if decode_tokens(row.tolist()) == expected:
                correct += 1
    return correct / total


# ---------------------------------------------------------------------------
# Linear probes
# ---------------------------------------------------------------------------


def train_linear_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    num_classes: int,
    epochs: int = 300,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> dict:
    """Train a single linear layer (d_model -> num_classes) by full-batch
    Adam + cross-entropy + L2 weight decay (the "ridge" regularization) to
    predict a categorical target (level or step-type) from encoder hidden
    states. Returns train/test accuracy plus bookkeeping. No sklearn
    dependency is available in this environment, so this is a from-scratch
    torch equivalent of sklearn's LogisticRegression(penalty='l2')."""
    torch.manual_seed(seed)
    X_train = X_train.detach()
    X_test = X_test.detach()
    d = X_train.shape[1]
    probe = nn.Linear(d, num_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        logits = probe(X_train)
        loss = nn.functional.cross_entropy(logits, y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_acc = (probe(X_train).argmax(-1) == y_train).float().mean().item()
        test_acc = (probe(X_test).argmax(-1) == y_test).float().mean().item()
        final_loss = nn.functional.cross_entropy(probe(X_train), y_train).item()
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'n_train': int(len(y_train)),
        'n_test': int(len(y_test)),
        'num_classes': num_classes,
        'final_train_loss': final_loss,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[ZetaTransformer, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt['hyperparameters']
    model = ZetaTransformer(
        max_len=hp.get('max_len', 40),
        num_encoder_layers=hp.get('encoder_layers', 1),
        num_decoder_layers=hp.get('decoder_layers', 1),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt
