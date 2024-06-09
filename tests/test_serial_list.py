from __future__ import annotations

import pytest

from repartition.serial.list_impl import merge_blocks, split_blocks


@pytest.fixture
def dummy_data() -> list[list[int]]:
    return [
        [1, 1, 1, 2, 2],
        [2, 2, 3, 3, 3],
        [3, 3, 4, 4, 4],
        [4, 4, 5, 5, 5],
        [5, 6, 6, 6, 7],
        [7, 7, 7, 8, 8],
        [8, 8],
    ]


def test_split_blocks(dummy_data):
    expected = [
        (1, [1, 1, 1]),
        (2, [2, 2, 2, 2]),
        (3, [3, 3, 3, 3, 3]),
        (4, [4, 4, 4, 4, 4]),
        (5, [5, 5, 5, 5]),
        (6, [6, 6, 6]),
        (7, [7, 7, 7, 7]),
        (8, [8, 8, 8, 8]),
    ]

    result = list(merge_blocks(split_blocks(dummy_data)))

    assert result == expected
