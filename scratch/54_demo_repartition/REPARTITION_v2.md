# A streaming-compatible implementation of repartition_by_column

I would like to propose a streaming version of the repartition_by_column function in Ray Data.

## Use Case

Large datasets are often partitioned by some column values to allow efficient reading of files. In this case, the column values are bounded in each file (i.e., each group is fully contained).

There are also applications in time series, in which multiple independent time series are bundled into one file. In this case, a column of group id is added to distinguish the content.

Currently, to process individual group or time-series in these datasets, a `groupby` operation is needed, which is an all-to-all operation and has a large memory footprint.


## Implementation

Let's take an example of a list of list of integers. This represents the blocks obtained from a read task with arbitrary block size.

Step 0: Each of the initial blocks contains multiple group keys. They are general blocks within each file. The group are not necessarily sorted.
```python
[[1, 1, 1, 2, 2, 2], [2, 2, 3, 3, 4, 4, 6, ], [6, 7, 7, 7, 7], ... ]
```

Step 1: Map / Split. Each block is split into multiple blocks depending on the group values. This operation is stream-able because it is 1-to-k block mapping.
```python
[[1, 1, 1], [2, 2, 2], [2, 2], [3, 3], [4, 4], [6], [6], [7, 7, 7, 7], ...]
```

Step 2: Reduce / Merge. Neighboring blocks belonging to the same group is merged into a single block. This operation is essentially the same as `itertools.groupby`. In this case, when the group of the next group changes, the current block is ready to be passed to the next stage.
```python
[[1, 1, 1], [2, 2, 2, 2, 2], [3, 3], [4, 4], [6, 6], [7, 7, 7, 7], ...]
```

### Serial Case

To see how it can be implemented as a streaming operation, we can write all the steps as iterators:
```python
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
```

The `repartition_by_column` function would simply be a composition of these iterators:
```python
def repartition_by_column(blocks: list[list[int]]) -> Iterator[tuple[int, list[int]]]:
    for key, block in merge_blocks(split_blocks(blocks)):
        yield key, block
```

Each block output by the merge step is ready for the next stage (e.g., `map_batches`)

### Parallel Case

To perform `repartition_by_column` in parallel, we can split the data stream into `p` processes and handle the boundaries. Using the previous example, we assume one process handles one block:

```python
process  |--------0-------|  |----------1--------|  |------2------| ...
data:   [[1, 1, 1, 2, 2, 2], [2, 2, 3, 3, 4, 4, 6], [6, 7, 7, 7, 7], ... ]
```

Step 1 is essentially the same. The splitting can be done within each process:

```python
process  |--------0---------|  |----------1--------------|  |--------2------| ...
data:   [[1, 1, 1], [2, 2, 2], [2, 2], [3, 3], [4, 4], [6], [6], [7, 7, 7, 7], ... ]
```

Step 2A: we need to handle boundaries before the merge step. For `i`-th process, it passes the left-most block to the `i-1`-th process. This ensures the new left-most block is always the beginning of the group.
```python
process  |--------0-----------------|  |----------1-----------|  |--------2------| ...
data:   [[1, 1, 1], [2, 2, 2], [2, 2], [3, 3], [4, 4], [6], [6], [7, 7, 7, 7], ... ]
```

### Time Complexity

Assume we have N data points with M groups and P processes. The time complexity of `repartition_by_column` is O(N/P) assuming `N>>P` and so the boundary data shuffling is negligible. This is because each split requires the reading the group columns (linear time or faster depending on whether they are sorted or not). Each merge also linear times w.r.t. the number of rows.