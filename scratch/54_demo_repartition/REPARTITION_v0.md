# Repartition By Column

In short, this is about a new function to align block boundaries in each partition without global shuffle. For applying user-defined functions in a partitioned dataset, it allows us to parallelize the computation by partitions and avoid applying a global sort (during `map_groups`).

Instead of overloading the `repartition` function, there are also a few alternative names to consider:
1. `repartition_by` or `repartition_by_column`
2. `repartition_sorted`
3. `align_by_column`
4. `coalesce` (see note)

Note: In pyspark, `coalesce` does not involve shuffle and only reduce the number of partitions.

## Use cases

Large datasets are often partitioned into different folder and files, so that the predicate filtering can be used to avoid reading all data. The partition columns can be either computed from the columns in the data (e.g., `part_x = x % 2`) or containing extra information that is not in the data (e.g., `/company=a/`, `/company=b/`, ...).

In addition, each file may contain multiple chunks to avoid small-file problems. For example, a file can bundle multiple time series. These time series are supposed to be processed separately.

### Example of partitioned and bundled dataset
Let's consider a scenario where we have many machines in a factory, and we aim to track their usage. Each machine records sensor data when in use. Over time, we accumulate a substantial amount of data, and we partition it into the following format:

```
/machine=a/date=2024-01-01/part0.parquet
/machine=a/date=2024-01-01/part1.parquet
/machine=a/date=2024-01-02/part0.parquet
...
/machine=c/date=2024-01-03/part9.parquet
```
In each parquet file, we have a few columns: `data_id`, `time`, `sensor_0`, `sensor_1`, etc, where `data_id` is a unique ID for each data segment. We may assume the data are bundled in a way so that each `data_id` chunk is continuous.

To process the data, the standard way is to use `groupby("id").apply(func)`. Currently, Ray Data would perform a global sort on `data_id`, then apply the UDF using `map_batches`. The global sort ensures the block boundaries are aligned with the boundary of `data_id`. However, it also an all-to-all operation that materialize the data all at once.

Since the data is partitioned, the "sort" or alignment of boundaries can be done per partition. For example, the groups `data_id` found in one file will not appear in other files. Therefore, it is possible to align boundaries in each file or partition individually.

## API

Let's assume we have a new API `repartition_by`. To process the dataset in the last section for each `data_id`, we call
```
ds.repartition_by("data_id").map_batches(func, batch_size=None)
```
instead of
```
ds.groupby("data_id").map_groups(func, batch_size=None)
```
The function `repartition_by` will ensure that each block fully contains the data with the same group key `data_id`. We set `batch_size` to `None` so that the `func` is applied to the full block.

Ideally, using `groupby` afterwards will not trigger a global sort:
```
ds.repartition_by("data_id").groupby("data_id").map_groups(func, batch_size=1234)
```
In this case, the internal blocks are aligned by `data_id`, and the `map_groups` will pass the data of given batch_size to `func`.


Note:
* Alternatively, we can add an option in `map_groups` after `groupby`. However, having it as a separate stage is easier to understand and to add more potential options as discussed below.
* A note or warning should be given if we detect the data is not truely partitioned. For example, the same `data_id` are found in two partitions. The `repartition_by` function should not try to do the shuffle or sort.

## Implementation

Suppose we have the following dataset, which consists of multiple time series. Conceptually, the `data_id` column look like
```
[   # partition-level
    [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],  # file 0
    [4, 4, 5, 5, 5, 5, 6, 6],           # file 1
    [7, 7, 7, 7, 7, 7, 8, 8, 8],        # file 2
]
```

The levels of blocks look like
* partition/file level: determined by the number of files and/or pyarrow partition filters
* batch level: determined by the batch size for reading
* sub-block level: determined by the number of rows of the same group keys in a single batch

### Basic Implementation

Consider the following map and reduce stages.
* Map-stage: split each block according to the group key
* Reduce-stage: merge the blocks with the same group key

Assuming we are reading the files in batches (batch size = 4):
```
[   # partition-level / file-level
    [   # batch-level
        [1, 1, 1, 2],
        [2, 2, 2, 3],
        [3, 3, 3],
    ],
    [
        [4, 4, 5, 5],
        [5, 5, 6, 6],
    ],
    [
        [7, 7, 7, 7],
        [7, 7, 8, 8],
        [8],
    ],
]
```
During the map-stage, we split the batch by the group key. The block is materialized to allow access to the actual data.
```
[   # partition-level / file-level
    [   # batch-level
        [1, 1, 1],
        [2],
        [2, 2, 2],
        [3],
        [3, 3, 3],
    ],
    [
        [4, 4],
        [5, 5],
        [5, 5],
        [6, 6],
    ],
    [
        [7, 7],
        [7, 7],
        [7, 7],
        [8, 8],
        [8],
    ],
]
```

For the reduce-stage, we concatenate the blocks with the same group key. Each reducer collects the blocks and merge them:
```python
reducer_mapping = {
    1: [[1, 1, 1]],
    2: [[2], [2, 2, 2]],
    3: [[3], [3, 3, 3]],
    ...
}
```
Since we assume each group key does not live in multiple partitions/files, we can assign each reducer to a partition, e.g., set the number of reducers equal to the number of partitions. For higher number of reducers, we can apply a final `reduce` on the output of the reducers belonging to the same partition.

In this basic implementation, each reducer is a blocking op until all the map-stage output is fetched. Then all the repartitioned blocks in a partition becomes available.

Finally, we have
```
[
    [1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [4, 4],
    [5, 5, 5, 5],
    [6, 6],
    [7, 7, 7, 7, 7, 7],
    [8, 8, 8],
]
```

### Streaming (WIP)

#### Map-stage
The map-stage can be performed in a streaming way by splitting each batch (from a read task) one by one. Each batch results in one or more sub-blocks (single group key). Roughly, in each partition,
```python
for block, _ in ref_bundle.blocks:
    for batch in block.to_batches():
        for b in split_single_block(batch)
            yield b
```

#### Single stream
Suppose the output from above map task is a single stream and we put them into a queue like below. The blocks are not ordered by partition or by batch. However, we may consume the queue in a FIFO manner.
```
[
    [1, 1, 1],  # partition 0 starts
    [2],
    [2, 2, 2],
    [4, 4],     # partition 1 starts
    [7, 7],     # partition 2 starts
    [7, 7],
    [5, 5],
    [3],
    [3, 3, 3],  # partition 0 ends
    [5, 5],
    [6, 6],     # partition 1 ends
    [7, 7],
    [8, 8],
    [8],        # partition 2 ends
]
```

Unlike the reduce-stage of `sort` and `repartition` (by number), we do not know how many groups per partition beforehand. Therefore we cannot assign the map output to a reducer during map-stage.

To let reduce-stage process the map outputs in parallel, we can make use of a coordinator (Ray actor). For this, we should let the map task yield a list of sub-blocks each time:
```
[   # batch-level
    [               # partition 0 starts
        [1, 1, 1],
        [2],
    ],
    [
        [2, 2, 2],
        [3],
    ],
    [               # partition 1 starts
        [4, 4],
        [5, 5],
    ],
    [               # partition 2 starts
        [7, 7],
        [7, 7],
    ],
    [               # partition 0 ends
        [3, 3, 3],
    ],
    [               # partition 1 ends
        [5, 5],
        [6, 6],
    ],
    [               # partition 2 ends
        [7, 7],
        [8, 8],
        [8],
    ],
]
```
The coordinator continuously monitors the input queue (from map-stage) and gets the entry one by one. Each entry consists of a partition id and the data (a list of sub-blocks). We use the partition id to identify which reducer the data should be sent to. It also keeps track of the partitions being processed. For example, an entry with a new partition will trigger the creation of a reducer dynamically. The map task could also put a marker in the stream to indicate the partition has been processed. In short, the coordinator handles K reducers at each moment, where K is the number of partitions involved. When an earlier partition ends (coordinator reads a END marker), the coordinator may label that reducer done and retrieve the output for downstream.


### Configurations and Options

Some configurations and options to consider:

* `preserve_order`: Preserve the ordering of the input blocks. This means a reducer cannot merge two sub-blocks unless it is sure that they are consecutive. A naive implementation is to keep track of the batch and sub-block indices (file-level index may be possible but more difficult).

* `target_max_block_size`: Possibly for downstream `groupby.map_groups` or `map_batches` that is not processing the full block.

* `output_num_blocks`: Potentially the same thing as
    ```python
    ds.repartition_by(10, "data_id")
    ```
    which results in 10 blocks instead of the number of unique values in `data_id`, i.e., one block with multiple `data_id`. On the other hand, if output_num_blocks is large, one group may be split into multiple blocks. User should have control to this behavior

* `sort_by`: Allow sorting within the block (based on a column other than the group key). See Spark RDD's repartitionAndSortWithinPartitions. The usage may look like this:
    ```python
    ds.repartition_by(key="data_id", sort_by="time")
    ```


## Other stuff

### How does pyspark do this?

In `pyspark`, this can be done by using `repartition`:

```
spark_df.repartition("id").applyInPandas(func)
```
This will repartition the dataframe so that all data belong to the same `id` live in the same partition. Note that pyspark's repartition



