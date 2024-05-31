import numpy as np


# OLD
class _MultiColumnSortedKey:
    """Represents a tuple of group keys with a ``__lt__`` method

    This is a simple implementation to support multi-column groupby.
    While a 1D array of tuples suffices to maintain the lexicographical
    sorted order, a comparison method is also needed in ``np.searchsorted``
    (for computing the group key boundaries).
    """

    __slots__ = ("data",)

    def __init__(self, *args):
        self.data = tuple(args)

    def __lt__(self, obj: "_MultiColumnSortedKey") -> bool:
        return self.data < obj.data

    def __repr__(self) -> str:
        """Print as T(1, 2)"""
        return "T" + self.data.__repr__()


def convert_into_recarray(data: dict) -> np.recarray:
    """Convert a dictionary of arrays into a record array"""

    dtype = [(k, v.dtype) for k, v in data.items()]
    return np.array(list(zip(*data.values(), strict=True)), dtype=dtype)


def get_key_boundaries_v0(keys: np.ndarray | dict[str, np.ndarray], append_first: bool = False) -> np.ndarray:
    """Get the boundaries of the key column"""

    if isinstance(keys, dict):
        # For multiple keys, we generate a separate tuple column
        convert_to_multi_column_sorted_key = np.vectorize(_MultiColumnSortedKey)
        keys: np.ndarray = convert_to_multi_column_sorted_key(*keys.values())

    boundaries = [0] if append_first else []
    start = 0
    while start < keys.size:
        end = start + np.searchsorted(keys[start:], keys[start], side="right")
        boundaries.append(end)
        start = end
    return boundaries


def get_key_boundaries_v1(keys: np.ndarray | dict[str, np.ndarray], append_first: bool = False) -> np.ndarray:
    """Get the boundaries of the key column"""

    if isinstance(keys, dict):
        # For multiple keys, we generate a separate tuple column
        keys = convert_into_recarray(keys)

    boundaries = [0] if append_first else []
    start = 0
    while start < keys.size:
        end = start + np.searchsorted(keys[start:], keys[start], side="right")
        boundaries.append(end)
        start = end
    return boundaries


def get_key_boundaries_v2(
    keys: np.ndarray | dict[str, np.ndarray], append_first: bool = False, return_recarray: bool = False
) -> np.ndarray | tuple[np.ndarray, np.recarray]:
    """Get the boundaries of the key column"""

    first_entry = [0] if append_first else []

    if isinstance(keys, np.ndarray):
        indices = np.hstack([first_entry, np.where(keys[1:] != keys[:-1])[0] + 1, [len(keys)]])
        arr_ = keys
    else:
        arr_ = np.rec.fromarrays(keys.values())
        indices = np.hstack([first_entry, np.where(arr_[1:] != arr_[:-1])[0] + 1, [len(arr_)]])

    if return_recarray:
        return indices, arr_
    else:
        return indices
