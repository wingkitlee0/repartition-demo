from typing import Iterator, Tuple, Union

import numpy as np
from ray.data.block import Block, BlockAccessor


def split_block(block, keys) -> Iterator[Tuple[Union[str, int], Block]]:
    if isinstance(keys, list) and len(keys) == 1:
        keys = keys[0]

    accessor = BlockAccessor.for_block(block)

    if accessor.num_rows() == 0:
        return []

    arr = accessor.to_numpy(keys)

    if isinstance(arr, np.ndarray):
        indices = np.hstack([[0], np.where(arr[1:] != arr[:-1])[0] + 1, [len(arr)]])
        arr_ = arr
    else:
        arr_ = np.rec.fromarrays(arr.values())
        indices = np.hstack([[0], np.where(arr_[1:] != arr_[:-1])[0] + 1, [len(arr)]])

    for start, end in zip(indices[:-1], indices[1:]):
        key = arr_[start]
        key = tuple(key) if isinstance(key, np.record) else key
        yield key, accessor.slice(start, end)
