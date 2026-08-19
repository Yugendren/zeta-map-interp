import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch

from zetamap.model import (
    PAD_ID,
    ZetaTransformer,
    count_parameters,
    decode_tokens,
    encode_source,
    encode_target,
)
from zetamap.dyck import all_dyck_paths, zeta


class TestEncoding(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        for d in list(all_dyck_paths(4))[:5]:
            word = ''.join(d)
            tgt = encode_target(word)
            self.assertEqual(decode_tokens(tgt), word)

    def test_encode_source_length(self):
        self.assertEqual(len(encode_source('NNEE')), 4)


class TestModelShapes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = ZetaTransformer()

    def test_param_count_near_paper(self):
        n_params = count_parameters(self.model)
        # Paper (arXiv:2511.12421) reports 339,716; we allow a few % slack
        # since embedding tying/vocab choices are not pinned by the spec.
        self.assertLess(abs(n_params - 339716) / 339716, 0.02)

    def test_forward_and_greedy_decode_shapes(self):
        words = [''.join(d) for d in list(all_dyck_paths(4))[:3]]
        src = torch.tensor([encode_source(w) for w in words])
        tgt_full = torch.tensor([encode_target(''.join(zeta(tuple(w)))) for w in words])
        tgt_in = tgt_full[:, :-1]
        logits = self.model(src, tgt_in)
        self.assertEqual(logits.shape, (len(words), tgt_in.shape[1], 5))

        gen = self.model.greedy_decode(src, max_len=16)
        self.assertEqual(gen.shape[0], len(words))
        self.assertTrue((gen[:, 0] != PAD_ID).all())


if __name__ == '__main__':
    unittest.main()
