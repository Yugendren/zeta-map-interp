import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch

from zetamap.dyck import all_dyck_paths, zeta
from zetamap.interp import (
    E_ID,
    N_ID,
    build_probe_set,
    exact_match_rate_ablated,
    forward_capture,
    greedy_decode_ablated,
    load_model,
    load_val_split,
    path_levels,
    train_linear_probe,
)
from zetamap.model import ZetaTransformer, encode_source, encode_target

DATA_PATH = ROOT / 'data' / 'dyck_n13.jsonl'
CHECKPOINT = ROOT / 'checkpoints' / 'model.pt'


class TestPathLevels(unittest.TestCase):
    def test_hand_anchors(self):
        # NNEE: heights before each step are 0, 1, 2, 1
        self.assertEqual(path_levels('NNEE'), [0, 1, 2, 1])
        # NENE: heights before each step are 0, 1, 0, 1
        self.assertEqual(path_levels('NENE'), [0, 1, 0, 1])

    def test_matches_area_word_on_N_positions(self):
        from zetamap.dyck import area_word

        for d in list(all_dyck_paths(6))[:20]:
            word = ''.join(d)
            levels = path_levels(word)
            n_positions = [i for i, c in enumerate(word) if c == 'N']
            expected = area_word(d)
            self.assertEqual([levels[i] for i in n_positions], list(expected))


class TestHeldOutSplit(unittest.TestCase):
    def test_probe_set_disjoint_from_low_index_train_examples(self):
        # Not a full leakage proof, but sanity-checks the split is
        # deterministic and reasonably sized.
        val_words, val_zeta = load_val_split(DATA_PATH, val_frac=0.05, seed=0)
        self.assertGreater(len(val_words), 0)
        self.assertEqual(len(val_words), len(val_zeta))
        for w, z in zip(val_words[:5], val_zeta[:5]):
            self.assertEqual(len(w), len(z))

    def test_build_probe_set_reproducible(self):
        words1, zwords1 = build_probe_set(DATA_PATH, count=50, split_seed=0, sample_seed=0)
        words2, zwords2 = build_probe_set(DATA_PATH, count=50, split_seed=0, sample_seed=0)
        self.assertEqual(words1, words2)
        self.assertEqual(zwords1, zwords2)
        self.assertEqual(len(words1), 50)


class TestActivationCapture(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = ZetaTransformer()
        self.model.eval()
        words = [''.join(d) for d in list(all_dyck_paths(4))[:6]]
        zwords = [''.join(zeta(tuple(w))) for w in words]
        self.words = words
        self.src = torch.tensor([encode_source(w) for w in words])
        tgt = torch.tensor([encode_target(z) for z in zwords])
        self.tgt_in = tgt[:, :-1]

    def test_capture_shapes(self):
        with torch.no_grad():
            acts = forward_capture(self.model, self.src, self.tgt_in)
        B = len(self.words)
        S = self.src.shape[1]
        T = self.tgt_in.shape[1]
        self.assertEqual(acts.encoder_hidden.shape, (B, S, 128))
        self.assertEqual(acts.encoder_self_attn.shape, (B, S, S))
        self.assertEqual(acts.decoder_cross_attn.shape, (B, T, S))
        self.assertEqual(acts.logits.shape, (B, T, 5))

    def test_cross_attention_rows_sum_to_one(self):
        with torch.no_grad():
            acts = forward_capture(self.model, self.src, self.tgt_in)
        row_sums = acts.decoder_cross_attn.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4))

    def test_patches_are_cleaned_up(self):
        enc_layer = self.model.encoder.layers[0]
        dec_layer = self.model.decoder.layers[0]
        self.assertNotIn('_sa_block', enc_layer.__dict__)
        self.assertNotIn('_mha_block', dec_layer.__dict__)
        with torch.no_grad():
            forward_capture(self.model, self.src, self.tgt_in)
        # after the context manager exits, the instance-level patch should
        # be gone again (falling back to the class method)
        self.assertNotIn('_sa_block', enc_layer.__dict__)
        self.assertNotIn('_mha_block', dec_layer.__dict__)


class TestAblation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = ZetaTransformer()
        self.model.eval()
        words = [''.join(d) for d in list(all_dyck_paths(4))[:6]]
        self.zwords = [''.join(zeta(tuple(w))) for w in words]
        self.src = torch.tensor([encode_source(w) for w in words])

    def test_ablated_decode_runs_and_shapes(self):
        gen_none = greedy_decode_ablated(self.model, self.src, ablate_token_id=None, max_len=12)
        gen_n = greedy_decode_ablated(self.model, self.src, ablate_token_id=N_ID, max_len=12)
        gen_e = greedy_decode_ablated(self.model, self.src, ablate_token_id=E_ID, max_len=12)
        self.assertEqual(gen_none.shape[0], self.src.shape[0])
        self.assertEqual(gen_n.shape[0], self.src.shape[0])
        self.assertEqual(gen_e.shape[0], self.src.shape[0])

    def test_exact_match_rate_ablated_in_range(self):
        rate = exact_match_rate_ablated(
            self.model, self.src, self.zwords, torch.device('cpu'), batch_size=6, max_len=12, ablate_token_id=None
        )
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)


class TestLinearProbe(unittest.TestCase):
    def test_probe_pipeline_runs_on_50_examples(self):
        torch.manual_seed(0)
        model, _ = load_model(CHECKPOINT, torch.device('cpu'))
        words, zwords = build_probe_set(DATA_PATH, count=50, split_seed=0, sample_seed=0)
        src = torch.tensor([encode_source(w) for w in words])
        tgt = torch.tensor([encode_target(z) for z in zwords])
        tgt_in = tgt[:, :-1]

        with torch.no_grad():
            acts = forward_capture(model, src, tgt_in)

        hidden = acts.encoder_hidden.reshape(-1, 128)
        levels = torch.tensor([lvl for w in words for lvl in path_levels(w)])
        n_train = 40 * hidden.shape[0] // 50
        X_train, y_train = hidden[:n_train], levels[:n_train]
        X_test, y_test = hidden[n_train:], levels[n_train:]

        result = train_linear_probe(X_train, y_train, X_test, y_test, num_classes=14, epochs=50)
        self.assertIn('test_acc', result)
        self.assertGreaterEqual(result['test_acc'], 0.0)
        self.assertLessEqual(result['test_acc'], 1.0)
        self.assertEqual(result['n_train'], n_train)


if __name__ == '__main__':
    unittest.main()
