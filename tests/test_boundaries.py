from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from repartition.boundaries.impl import (
    _MultiColumnSortedKey,
    convert_into_recarray,
    get_key_boundaries_v0,
    get_key_boundaries_v1,
    get_key_boundaries_v2,
)

DEFAULT_LARGE_SIZE = 100_000


@pytest.fixture
def dummy_data():
    return {
        "x": np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int32),
        "y": np.array([1, 1, 3, 2, 2, 3, 1, 2, 3], dtype=np.int32),
    }


def test_tuple_0001(dummy_data):
    tuple_array = np.empty(len(dummy_data["x"]), dtype=object)
    tuple_array[:] = list(zip(*dummy_data.values(), strict=True))

    with pytest.raises(TypeError):
        np.searchsorted(tuple_array, [(1, 1), (1, 2)])


def generate_data(n: int, nchoice: int, seed=None, dtype=None):
    if dtype is None:
        dtype = np.int64

    rng = np.random.default_rng(seed)

    data = {
        "x": rng.integers(0, nchoice, size=n).astype(dtype),
        "y": rng.integers(0, nchoice, size=n).astype(dtype),
    }

    for v in data.values():
        v.sort()

    return data


@pytest.fixture
def dummy_large_data():
    return generate_data(n=DEFAULT_LARGE_SIZE, nchoice=10, seed=1234)


@pytest.mark.parametrize(
    "n, nchoice, seed, expected", [(100, 10, 1234, [8, 12]), (100, 10, 1, [9, 9]), (100, 10, 2, [7, 13])]
)
def test_rec_array_0001(n, nchoice, seed, expected):
    data = generate_data(n, nchoice, seed=seed)

    record_array = convert_into_recarray(data)

    indices = np.searchsorted(record_array, record_array[[1, 8]], side="right")

    assert list(indices) == expected


def test_rec_array_0002(dummy_data):
    record_array = convert_into_recarray(dummy_data)

    indices = np.searchsorted(record_array, record_array[[3, 4]], side="right")

    print(type(indices))


def func_recarray(data, target: list[tuple[int, int]]):
    record_array = convert_into_recarray(data)

    indices = np.searchsorted(record_array, record_array[[3, 4]], side="right")

    return indices


def func_multicolumn(data, target: list[tuple[int, int]]):
    multi_column_array = np.array([_MultiColumnSortedKey(*k) for k in zip(*data.values(), strict=True)])

    indices = np.searchsorted(multi_column_array, [_MultiColumnSortedKey(*k) for k in target])

    return indices


@pytest.mark.parametrize("func", [func_recarray, func_multicolumn])
def test_benchmark(benchmark, func, dummy_large_data):
    f = partial(func, dummy_large_data, target=[(1, 1), (1, 3)])

    indices = benchmark(f)
    print(indices)
    assert len(indices) == 2


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_size(dtype):
    data = generate_data(n=100_000, nchoice=20, seed=1234, dtype=dtype)

    record_array = convert_into_recarray(data)
    bytes_recarray = record_array.nbytes

    multi_column_array = np.array([_MultiColumnSortedKey(*k) for k in zip(*data.values(), strict=True)])

    # get the number of bytes of the array
    bytes_multi = multi_column_array.nbytes

    print(bytes_recarray)
    print(bytes_multi)
    print(f"{dtype=}:", bytes_recarray / bytes_multi)


@pytest.mark.parametrize(
    "get_key_boundaries",
    [
        get_key_boundaries_v0,
        get_key_boundaries_v1,
        get_key_boundaries_v2,
    ],
)
def test_get_key_boundaries_benchmark(benchmark, get_key_boundaries, dummy_large_data):
    func = partial(get_key_boundaries, dummy_large_data)

    benchmark(func)


def test_get_key_boundaries_consistent(dummy_data):
    results = []

    for func in [get_key_boundaries_v0, get_key_boundaries_v1, get_key_boundaries_v2]:
        results.append(func(dummy_data))

    assert np.allclose(results[0], results[1])
    assert np.allclose(results[0], results[2])
