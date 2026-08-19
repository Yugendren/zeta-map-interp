"""Dyck path combinatorics: statistics (area, dinv, bounce) and the Haglund
zeta map, sending (area, dinv) statistics on Dyck_n to (bounce, area).

A Dyck path of order n is represented as a tuple of n 'N' and n 'E' steps
walking from (0,0) to (n,n) that stays weakly above the diagonal y = x
(equivalently, reading 'N' as an up-step and 'E' as a down-step, a ballot
sequence: every prefix has at least as many N's as E's).

The *area word* a_1, ..., a_n of a path is defined by walking the path while
tracking height h (h starts at 0; each N step is recorded as a_i = h *before*
incrementing h; each E step decrements h). This gives a_1 = 0 and
a_{i+1} <= a_i + 1 for all i, which is the standard characterization of area
words of Dyck paths.

This module is pure Python / stdlib only.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

Path_ = tuple  # a Dyck path is a tuple[str, ...] of 'N'/'E'


def all_dyck_paths(n: int) -> Iterator[tuple[str, ...]]:
    """Exhaustively yield all Dyck paths of order n as tuples of 'N'/'E'.

    A path is a sequence of n 'N' and n 'E' steps such that every prefix has
    at least as many 'N' as 'E' (weakly above the diagonal).
    """
    if n == 0:
        yield ()
        return

    def rec(path: list[str], n_count: int, e_count: int):
        if n_count == n and e_count == n:
            yield tuple(path)
            return
        if n_count < n:
            path.append('N')
            yield from rec(path, n_count + 1, e_count)
            path.pop()
        if e_count < n_count:
            path.append('E')
            yield from rec(path, n_count, e_count + 1)
            path.pop()

    yield from rec([], 0, 0)


def random_dyck_path(n: int, rng: random.Random) -> tuple[str, ...]:
    """Sample a Dyck path of order n uniformly at random via the cycle lemma.

    Method (Dvoretzky-Motzkin cycle lemma, the standard "add one and rotate"
    construction for uniform random ballot sequences / Dyck paths):

    1. Form the multiset of n 'N' and (n + 1) 'E' steps (one extra 'E') and
       draw a uniformly random linear arrangement of it (a random shuffle).
       This arrangement has length 2n + 1 and, reading N as +1 and E as -1,
       its steps sum to -1.
    2. The cycle lemma guarantees that among the 2n + 1 cyclic rotations of
       this arrangement, EXACTLY ONE has the property that every proper
       prefix sum is >= 0 (equivalently: the running N-count never drops
       below the running E-count until the very last step, which must be
       the unique step that takes the cumulative sum to -1). Concretely,
       that rotation is the one starting immediately after the position
       achieving the (first) global minimum of the prefix-sum sequence.
    3. Dropping the trailing 'E' of that rotation yields a sequence of n 'N'
       and n 'E' steps whose every prefix has #N >= #E, i.e. a Dyck path.

    Why this gives a UNIFORM Dyck path: extending a Dyck path of order n by
    one trailing 'E' and taking all 2n + 1 of its cyclic rotations produces
    2n + 1 *distinct* arrangements (no periodicity is possible since the
    counts n and n + 1 are coprime), and by the cycle lemma exactly one of
    those rotations is itself a valid preimage in step 2 above. So the map
    "arrangement -> Dyck path" from step 2 is exactly (2n + 1)-to-1, and
    since C(2n + 1, n) = (2n + 1) * Catalan(n), every Dyck path has exactly
    the same number (2n + 1) of preimages among the uniformly-random
    arrangements drawn in step 1 -- hence a uniform arrangement yields a
    uniform Dyck path. (Verified empirically: see tests/test_dyck.py, which
    checks validity for many n and near-uniform frequencies for n = 4 by
    exhaustive comparison against all_dyck_paths.)
    """
    if n == 0:
        return ()
    steps = ['N'] * n + ['E'] * (n + 1)
    rng.shuffle(steps)
    prefix = 0
    best = 0
    best_idx = 0
    for i, s in enumerate(steps):
        prefix += 1 if s == 'N' else -1
        if prefix < best:
            best = prefix
            best_idx = i + 1
    rotated = steps[best_idx:] + steps[:best_idx]
    assert rotated[-1] == 'E'
    return tuple(rotated[:-1])


def area_word(path: tuple[str, ...]) -> tuple[int, ...]:
    """Return the area word a_1, ..., a_n of a Dyck path.

    a_i is the height h just before the i-th N step is taken (h starts at 0,
    increments on N, decrements on E).
    """
    word = []
    h = 0
    for step in path:
        if step == 'N':
            word.append(h)
            h += 1
        else:
            h -= 1
    return tuple(word)


def area(path: tuple[str, ...]) -> int:
    """area(D) = sum of the area word.

    Hand-verified anchors: area(('N','N','E','E')) == 1 (area word (0, 1));
    area(('N','E','N','E')) == 0 (area word (0, 0)).
    """
    return sum(area_word(path))


def dinv(path: tuple[str, ...]) -> int:
    """dinv(D) = #{(i, j) : i < j, a_i - a_j in {0, 1}} over the area word.

    Hand-verified anchors: dinv of the path with area word (0,0,0)
    [word NENENE... i.e. UDUDUD] is 3; dinv of the path with area word
    (0,1,2) [UUUDDD] is 0.
    """
    a = area_word(path)
    n = len(a)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] - a[j] in (0, 1):
                count += 1
    return count


def bounce(path: tuple[str, ...]) -> int:
    """bounce(D), computed via the bounce path construction.

    xh[y] = x-coordinate (number of E steps taken so far) at which the path
    makes its y-th N step, i.e. the N step going from height y-1 to height y
    (y = 1, ..., n). Then: c = n; total = 0; while c > 0: c = xh[c];
    total += c.

    Hand-verified anchors: bounce(NENENE) == 3; bounce(NNNEEE) == 0.
    """
    n = len(path) // 2
    xh = [0] * (n + 1)  # xh[y] for y = 1..n
    y = 0
    e_count = 0
    for step in path:
        if step == 'N':
            y += 1
            xh[y] = e_count
        else:
            e_count += 1

    c = n
    total = 0
    while c > 0:
        c = xh[c]
        total += c
    return total


# ---------------------------------------------------------------------------
# The zeta map
# ---------------------------------------------------------------------------
#
# Winning convention (found by exhaustive brute-force search over an 8-way
# "diagonal reading" family of constructions: stage order in {ascending,
# descending} x scan direction within a stage in {left-to-right,
# right-to-left} x symbol assignment in {N-on-k, E-on-k}; see
# scratch/zeta_search.py, preserved below in this docstring for provenance):
#
#   Let a_1, ..., a_n be the area word of D and m = max(a_i) (for n = 0,
#   zeta is the empty path).
#   For k = m+1, m, m-1, ..., 1, 0 (DESCENDING stage order):
#       scan positions i = n, n-1, ..., 1 (RIGHT-TO-LEFT within each stage)
#       for each i in that scan order:
#           if a_i == k:      emit 'E'
#           elif a_i == k-1:  emit 'N'
#           else: skip (position not part of this stage)
#   Concatenate the stages in the order k = m+1, m, ..., 0.
#
# This "descending stages / right-to-left scan / E-on-k,N-on-(k-1)" variant
# is the UNIQUE winner among all 8 variants tried; it is the only one that
# satisfies all three acceptance properties
#   (P1) zeta: Dyck_n -> Dyck_n is a bijection
#   (P2) area(zeta(D)) == dinv(D)
#   (P3) bounce(zeta(D)) == area(D)
# exhaustively for n = 1..9 (checked further than the required 1..6 during
# the search). The other 7 variants all failed, most on very small n:
#   asc/ltr/N_on_k  - fails P3 at n=3 on D=NNENEE (bounce(zeta)=1, need 2)
#                     [this was the initially-tried convention]
#   asc/ltr/E_on_k  - fails P1 at n=1: zeta(NE) = EN, not a Dyck path
#   asc/rtl/N_on_k  - fails P2 at n=2 on D=NNEE (area(zeta)=1, dinv=0)
#   asc/rtl/E_on_k  - fails P1 at n=1: zeta(NE) = EN, not a Dyck path
#   desc/ltr/N_on_k - fails P1 at n=1: zeta(NE) = EN, not a Dyck path
#   desc/ltr/E_on_k - fails P2 at n=2 on D=NNEE (area(zeta)=1, dinv=0)
#   desc/rtl/N_on_k - fails P1 at n=1: zeta(NE) = EN, not a Dyck path
# No widening to the sweep-map formulation was needed: the winner was found
# within the original 8-variant "diagonal reading" family.


def zeta(path: tuple[str, ...]) -> tuple[str, ...]:
    """The Haglund zeta map: Dyck_n -> Dyck_n.

    Satisfies area(zeta(D)) == dinv(D) and bounce(zeta(D)) == area(D) for
    every Dyck path D (verified exhaustively for n = 1..9 in tests).
    """
    n = len(path) // 2
    if n == 0:
        return ()
    a = area_word(path)
    m = max(a)
    out: list[str] = []
    for k in range(m + 1, -1, -1):  # descending: m+1, m, ..., 0
        for i in range(n - 1, -1, -1):  # right-to-left scan
            if a[i] == k:
                out.append('E')
            elif a[i] == k - 1:
                out.append('N')
    return tuple(out)


def zeta_inverse(path: tuple[str, ...]) -> tuple[str, ...]:
    """Inverse of the zeta map, via a lookup table built from all paths of
    the appropriate order n (fine for n <= 13; a direct linear-time inverse
    algorithm exists, see Thomas-Williams "Sweep maps: A continuous family
    of sorting algorithms" for the construction, but is not implemented
    here).
    """
    n = len(path) // 2
    table = {zeta(d): d for d in all_dyck_paths(n)}
    return table[path]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_dataset(n: int, out_path: str | Path) -> None:
    """Write a JSONL dataset of all Dyck paths of order n to out_path.

    Each line has fields: word, area_word, area, dinv, bounce, zeta_word.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        for d in all_dyck_paths(n):
            record = {
                'word': ''.join(d),
                'area_word': list(area_word(d)),
                'area': area(d),
                'dinv': dinv(d),
                'bounce': bounce(d),
                'zeta_word': ''.join(zeta(d)),
            }
            f.write(json.dumps(record) + '\n')
