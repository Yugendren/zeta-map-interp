"""Search harness used to find the winning zeta-map convention.

Enumerates the 8-variant "diagonal reading" family (stage order in
{ascending, descending} x scan direction within a stage in
{left-to-right, right-to-left} x symbol assignment in {N-on-k, E-on-k})
and tests each variant against the three acceptance properties:
  (P1) zeta: Dyck_n -> Dyck_n is a bijection
  (P2) area(zeta(D)) == dinv(D)
  (P3) bounce(zeta(D)) == area(D)
exhaustively for n = 1..6 (the winner was subsequently re-checked up to
n = 1..9 in tests/test_dyck.py). Kept here for provenance; not part of the
package or test suite. Run with:

    .venv/bin/python scratch/zeta_search.py
"""

import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from zetamap.dyck import all_dyck_paths, area_word, area, dinv, bounce


def make_variant(stage_order, scan_dir, symbol_assign):
    def z(path):
        n = len(path) // 2
        if n == 0:
            return ()
        a = area_word(path)
        m = max(a)
        ks = list(range(0, m + 2))
        if stage_order == 'desc':
            ks = list(reversed(ks))
        out = []
        for k in ks:
            idxs = list(range(n))
            if scan_dir == 'rtl':
                idxs = list(reversed(idxs))
            for i in idxs:
                if symbol_assign == 'N_on_k':
                    if a[i] == k:
                        out.append('N')
                    elif a[i] == k - 1:
                        out.append('E')
                else:  # E_on_k
                    if a[i] == k:
                        out.append('E')
                    elif a[i] == k - 1:
                        out.append('N')
        return tuple(out)
    return z


def main():
    results = {}
    for stage_order, scan_dir, symbol_assign in product(
        ['asc', 'desc'], ['ltr', 'rtl'], ['N_on_k', 'E_on_k']
    ):
        name = f"{stage_order}/{scan_dir}/{symbol_assign}"
        z = make_variant(stage_order, scan_dir, symbol_assign)
        ok = True
        fail_reason = None
        for n in range(1, 7):
            paths = list(all_dyck_paths(n))
            images = []
            pathset = set(paths)
            for d in paths:
                zd = z(d)
                images.append(zd)
                if zd not in pathset:
                    ok = False
                    fail_reason = f"n={n} zeta({''.join(d)}) = {''.join(zd)} not a valid Dyck path"
                    break
                if area(zd) != dinv(d):
                    ok = False
                    fail_reason = f"n={n} P2 fail on {''.join(d)}: area(zeta)={area(zd)} dinv={dinv(d)}"
                    break
                if bounce(zd) != area(d):
                    ok = False
                    fail_reason = f"n={n} P3 fail on {''.join(d)}: bounce(zeta)={bounce(zd)} area={area(d)}"
                    break
            if not ok:
                break
            if len(set(images)) != len(images):
                ok = False
                fail_reason = f"n={n} not injective"
                break
        results[name] = (ok, fail_reason)
        print(name, ok, fail_reason)

    print()
    print("Winners:", [k for k, v in results.items() if v[0]])


if __name__ == '__main__':
    main()
