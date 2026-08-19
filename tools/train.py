"""Train the ZetaTransformer on n=13 Dyck paths (word -> zeta(word)),
reproducing the training setup of Huang-Jackson-Lee (arXiv:2511.12421):
AdamW with linear warmup, batch 512, teacher forcing + cross-entropy,
greedy autoregressive decoding for eval.

Usage:
    .venv/bin/python tools/train.py [--lr 3e-4] [--epochs 40] [--encoder-layers 1] ...

Saves:
    checkpoints/model.pt          -- final model state_dict + hyperparameters
    results/train-metrics.json    -- per-epoch train loss / val exact-match + all hyperparameters
    results/generalization.json   -- exact-match on n=11,12 (exhaustive) and
                                      n=14,15,16 (10,000 random samples each)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
from torch import nn

from zetamap.dyck import all_dyck_paths, random_dyck_path, zeta
from zetamap.model import (
    PAD_ID,
    ZetaTransformer,
    count_parameters,
    decode_tokens,
    encode_source,
    encode_target,
)

import random as pyrandom


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    words, zeta_words = [], []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            words.append(rec['word'])
            zeta_words.append(rec['zeta_word'])
    return words, zeta_words


def build_tensors(words: list[str], zeta_words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.tensor([encode_source(w) for w in words], dtype=torch.long)
    tgt = torch.tensor([encode_target(z) for z in zeta_words], dtype=torch.long)
    return src, tgt


def exhaustive_tensors(n: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    words, zeta_words = [], []
    for d in all_dyck_paths(n):
        words.append(''.join(d))
        zeta_words.append(''.join(zeta(d)))
    src, tgt = build_tensors(words, zeta_words)
    return src, tgt, zeta_words


def random_sample_tensors(
    n: int, count: int, rng: pyrandom.Random
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    words, zeta_words = [], []
    for _ in range(count):
        d = random_dyck_path(n, rng)
        words.append(''.join(d))
        zeta_words.append(''.join(zeta(d)))
    src, tgt = build_tensors(words, zeta_words)
    return src, tgt, zeta_words


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def exact_match_rate(
    model: ZetaTransformer,
    src: torch.Tensor,
    expected_words: list[str],
    device: torch.device,
    batch_size: int,
    max_len: int,
) -> float:
    model.eval()
    correct = 0
    total = len(expected_words)
    for i in range(0, total, batch_size):
        batch_src = src[i : i + batch_size].to(device)
        gen = model.greedy_decode(batch_src, max_len=max_len)
        gen = gen.cpu()
        for row, expected in zip(gen, expected_words[i : i + batch_size]):
            if decode_tokens(row.tolist()) == expected:
                correct += 1
    return correct / total


# ---------------------------------------------------------------------------
# LR schedule: linear warmup then constant
# ---------------------------------------------------------------------------


def make_lr_lambda(warmup_steps: int, total_steps: int | None = None, schedule: str = 'constant'):
    """Linear warmup for `warmup_steps`, then either held constant or
    cosine-decayed to 0 over the remaining `total_steps`.

    `schedule='constant'` is what the task spec literally asks for
    ("AdamW lr 3e-4 with warmup"); `schedule='cosine'` is a deviation used
    as a fallback when constant-lr training plateaued well below target
    exact-match despite steadily falling loss (see tools/train.py usage
    notes / results/train-metrics.json 'note' field for when it was used).
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == 'constant' or total_steps is None:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-n', type=int, default=13)
    ap.add_argument('--data-path', type=str, default=str(ROOT / 'data' / 'dyck_n13.jsonl'))
    ap.add_argument('--val-frac', type=float, default=0.05)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup-steps', type=int, default=1000)
    ap.add_argument('--lr-schedule', type=str, default='constant', choices=['constant', 'cosine'])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--encoder-layers', type=int, default=1)
    ap.add_argument('--decoder-layers', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=str, default='auto')
    ap.add_argument('--max-len', type=int, default=40)
    ap.add_argument('--subset-size', type=int, default=0, help='if >0, subsample training set to this many examples (time-budget fallback)')
    ap.add_argument('--early-stop-exact-match', type=float, default=0.99)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--eval-batch-size', type=int, default=1024)
    ap.add_argument('--gen-eval-samples', type=int, default=10000, help='random samples per n for n=14,15,16 generalization eval')
    ap.add_argument('--run-generalization', action='store_true', default=True)
    ap.add_argument('--tag', type=str, default='', help='free-text note appended to the metrics json, e.g. describing a fallback that was used')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')

    # ---- data ----
    t0 = time.time()
    words, zeta_words = load_jsonl(Path(args.data_path))
    print(f'Loaded {len(words)} examples for n={args.train_n} in {time.time()-t0:.1f}s')

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(words))
    n_val = int(len(words) * args.val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    if args.subset_size and args.subset_size < len(train_idx):
        train_idx = train_idx[: args.subset_size]
        print(f'Subsampled training set to {len(train_idx)} examples (time-budget fallback)')

    train_words = [words[i] for i in train_idx]
    train_zeta = [zeta_words[i] for i in train_idx]
    val_words = [words[i] for i in val_idx]
    val_zeta = [zeta_words[i] for i in val_idx]

    train_src, train_tgt = build_tensors(train_words, train_zeta)
    val_src, _ = build_tensors(val_words, val_zeta)

    print(f'Train: {len(train_words)}  Val: {len(val_words)}')

    seq_len_src = 2 * args.train_n
    seq_len_tgt = 2 * args.train_n + 2
    assert args.max_len >= seq_len_tgt, 'max_len too small for training n'

    # ---- model ----
    model = ZetaTransformer(
        max_len=args.max_len,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
    ).to(device)
    n_params = count_parameters(model)
    print(f'Model parameters: {n_params:,} (paper reports 339,716)')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps_per_epoch = (len(train_words) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(args.warmup_steps, total_steps, args.lr_schedule)
    )

    metrics = {
        'hyperparameters': vars(args),
        'device': str(device),
        'param_count': n_params,
        'train_size': len(train_words),
        'val_size': len(val_words),
        'seq_len_src': seq_len_src,
        'seq_len_tgt': seq_len_tgt,
        'epochs': [],
    }

    train_start = time.time()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        perm = torch.randperm(len(train_words))
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_words), args.batch_size):
            batch_idx = perm[i : i + args.batch_size]
            src = train_src[batch_idx].to(device)
            tgt = train_tgt[batch_idx].to(device)
            tgt_in = tgt[:, :-1]
            labels = tgt[:, 1:]

            logits = model(src, tgt_in)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=PAD_ID
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        val_exact_match = exact_match_rate(
            model, val_src, val_zeta, device, args.eval_batch_size, args.max_len
        )
        epoch_time = time.time() - epoch_start
        lr_now = scheduler.get_last_lr()[0]
        print(
            f'Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  '
            f'val_exact_match={val_exact_match:.4f}  lr={lr_now:.2e}  time={epoch_time:.1f}s'
        )
        metrics['epochs'].append(
            {
                'epoch': epoch,
                'train_loss': avg_loss,
                'val_exact_match': val_exact_match,
                'lr': lr_now,
                'epoch_time_sec': epoch_time,
            }
        )

        # checkpoint every epoch (cheap, tiny model) so we always have the latest
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'hyperparameters': vars(args),
                'epoch': epoch,
                'val_exact_match': val_exact_match,
            },
            ROOT / 'checkpoints' / 'model.pt',
        )

        if val_exact_match >= args.early_stop_exact_match:
            print(f'Early stopping: val exact-match {val_exact_match:.4f} >= target {args.early_stop_exact_match}')
            break

    total_train_time = time.time() - train_start
    metrics['total_train_time_sec'] = total_train_time
    metrics['final_val_exact_match'] = metrics['epochs'][-1]['val_exact_match']
    metrics['note'] = args.tag

    (ROOT / 'results').mkdir(exist_ok=True)
    with (ROOT / 'results' / 'train-metrics.json').open('w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Wrote results/train-metrics.json (total train time {total_train_time/60:.1f} min)')

    # ---- generalization eval ----
    if args.run_generalization:
        gen_results = {}
        pyrng = pyrandom.Random(args.seed + 1)

        for n in (11, 12):
            src, _, zwords = exhaustive_tensors(n)
            max_len_needed = 2 * n + 2
            rate = exact_match_rate(
                model, src, zwords, device, args.eval_batch_size, max(max_len_needed, args.max_len)
            )
            gen_results[f'n={n}'] = {'exact_match': rate, 'n_examples': len(zwords), 'mode': 'exhaustive'}
            print(f'Generalization n={n} (exhaustive, {len(zwords)} examples): exact_match={rate:.4f}')

        gen_results['n=13'] = {
            'exact_match': metrics['final_val_exact_match'],
            'n_examples': len(val_words),
            'mode': 'held-out validation split (same distribution as training)',
        }

        for n in (14, 15, 16):
            src, _, zwords = random_sample_tensors(n, args.gen_eval_samples, pyrng)
            max_len_needed = 2 * n + 2
            rate = exact_match_rate(
                model, src, zwords, device, args.eval_batch_size, max(max_len_needed, args.max_len)
            )
            gen_results[f'n={n}'] = {
                'exact_match': rate,
                'n_examples': len(zwords),
                'mode': f'{args.gen_eval_samples} random samples (cycle-lemma sampler)',
            }
            print(f'Generalization n={n} (random {args.gen_eval_samples}): exact_match={rate:.4f}')

        gen_results['hyperparameters'] = vars(args)
        gen_results['param_count'] = n_params
        with (ROOT / 'results' / 'generalization.json').open('w') as f:
            json.dump(gen_results, f, indent=2)
        print('Wrote results/generalization.json')


if __name__ == '__main__':
    main()
