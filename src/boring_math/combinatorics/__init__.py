# Copyright 2024-2026 Geoffrey R. Scheller
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Combinatorics Library
---------------------

.. admonition:: Combinations and permutations

    Computation efficient written in pure Python.

"""

from pythonic_fp.circulararray.auto import CA
from pythonic_fp.iterables.folding import fold_left
from boring_math.number_theory import coprime

__all__ = ['comb', 'perm']

def comb(n: int, m: int, /, target_top: int = 700, target_bot: int = 5) -> int:
    """
    .. admonition:: C(n, m) - combinations

        Number of ways m items can be taken from n items. Geared to
        works efficiently for Python's arbitrary length integers.

        :param n: Total number of distinct items to choose from.
        :param m: Number of items to choose, order does not matter.
        :returns: Number of ways to choose m items from n items.
        :raises ValueError: If either n < 0 or m < 0.

        .. note::

            - slower but comparable to math.comb
            - default parameters geared to large values of n and m
            - defaults work reasonably well for smaller (human size) values

            .. tip::

                For inner loops with smaller values, use
                ``target_top = target_bot = 1``, - or just
                use ``math.comb(n, m)`` instead

    """
    # edge cases
    if n < 0 or m < 0:
        raise ValueError('for C(n, m) n and m must be non-negative ints')

    if m in {0, n}:
        return 1

    if m > n:
        return 0

    # Using C(n, m) = C(n, n-m) to reduce number of factors to calculate.
    if m > (n // 2):
        m = n - m

    # Prepare data structures
    tops: CA[int] = CA(range(n - m + 1, n + 1))
    bots: CA[int] = CA(range(1, m + 1))

    # Compacting data structures makes algorithm
    # work better for larger arguments.
    size = len(tops)
    while size > target_top:
        size -= 1
        top, bot = coprime(tops.popl() * tops.popl(), bots.popl() * bots.popl())
        tops.pushr(top)
        bots.pushr(bot)

    while size > target_bot:
        size -= 1
        bots.pushr(bots.popl() * bots.popl())

    # Cancel all factors in denominator before multiplying
    # the remaining factors in the numerator.
    for bot in bots:
        for _ in range(len(tops)):
            top, bot = coprime(tops.popl(), bot)
            if top > 1:
                tops.pushr(top)
            if bot == 1:
                break

    return tops.foldl(lambda x, y: x * y, 1)


def perm(n: int, m: int, /) -> int:
    """
    .. admonition:: P(n, m) - permutations

        Number of orderings of m items taken from n items.

        :param n: Total number of distinct items to order.
        :param m: Number of items in each ordering.
        :returns: Number of arrangements of m items from n items.
        :raises ValueError: If either n < 0 or m < 0.

        .. warning::

            About 5 times slower than the math.perm C code.

    """
    # edge cases
    if n < 0 or m < 0:
        raise ValueError('for P(n, m) n and m must be non-negative ints')

    if m > n:
        return 0
    if n == 0:
        return 1
    return fold_left(range(n - m + 1, n + 1), lambda j, k: j * k, 1)
