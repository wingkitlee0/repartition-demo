from __future__ import annotations

import itertools
from collections.abc import Iterator


def split(block: list[int]) -> Iterator[tuple[int, list[int]]]:
    for k, g in itertools.groupby(block, lambda x: x):
        yield k, list(g)


def split_blocks(blocks: list[list[int]]) -> Iterator[tuple[int, list[int]]]:
    for block in blocks:
        yield from split(block)


def merge_blocks(items: Iterator[tuple[int, list[int]]]) -> Iterator[tuple[int, list[int]]]:
    for key, group in itertools.groupby(items, lambda x: x[0]):
        yield key, [x for _, b in group for x in b]
