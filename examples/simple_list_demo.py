import itertools
from typing import Iterator


def split(block: list[int]):
    for k, g in itertools.groupby(block, lambda x: x):
        yield k, list(g)


def split_blocks(blocks: list[list[int]]):
    for block in blocks:
        for k, g in split(block):
            yield k, g


def merge_blocks(items: Iterator[tuple[int, list[int]]]):
    for key, group in itertools.groupby(items, lambda x: x[0]):
        yield key, [x for _, b in group for x in b]


"""
Expected output:
    ```
    1 [1, 1, 1]
    2 [2, 2, 2, 2]
    3 [3, 3, 3, 3, 3]
    4 [4, 4, 4, 4, 4]
    5 [5, 5, 5, 5]
    6 [6, 6, 6]
    7 [7, 7, 7, 7]
    8 [8, 8, 8, 8]
    ```
"""


def main():
    blocks = [
        [1, 1, 1, 2, 2],
        [2, 2, 3, 3, 3],
        [3, 3, 4, 4, 4],
        [4, 4, 5, 5, 5],
        [5, 6, 6, 6, 7],
        [7, 7, 7, 8, 8],
        [8, 8],
    ]

    for key, block in merge_blocks(split_blocks(blocks)):
        print(key, block)


if __name__ == "__main__":
    main()
