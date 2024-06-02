"""
Example:

    ```
    time python scripts/examples/repart_demo1.py -i data/unordered/ -n 10 -b 20_000 --debug
    ```

"""
import argparse
import logging

import polars as pl
import pyarrow.parquet as pq
import ray
import ray.data
from repartition.ray.dummy_ref_bundles import get_ref_bundles_from_pyarrow_dataset
from repartition.ray_data._internal.repartition_by_column import repartition_runner

logger = logging.getLogger(__name__)


DEFAULT_BATCH_SIZE = 100_000


def mapping_func(table):
    return (
        pl.from_arrow(table)
        .select(
            pl.first("g").alias("g"),
            pl.n_unique("g").alias("group_count"),
            pl.len().alias("row_count"),
        )
        .to_arrow()
    )


def process(paths, batch_size, num_actors: int):
    pq_ds = pq.ParquetDataset(paths)

    ref_bundles = get_ref_bundles_from_pyarrow_dataset(pq_ds, batch_size, None)

    print(f"number of ref_bundles: {len(ref_bundles)}")

    all_results = []
    for i, rb in enumerate(ref_bundles):
        print(f"{i=}, {len(rb.blocks)=}")

        block_refs = [b for b, _ in rb.blocks]

        refs = list(repartition_runner(i, block_refs, ["g", num_actors]))

        table_refs = refs[: -(num_actors * 2)]

        ds = ray.data.from_arrow_refs(table_refs)

        print(ds.schema)

        results = ds.map_batches(mapping_func, batch_size=None, batch_format="pyarrow").to_arrow_refs()
        all_results.extend(results)

    print("Processing output")
    df = pl.from_arrow(ray.get(all_results))
    df.write_csv(f"result_repartition_n{num_actors}.csv")


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="input directory or file")
    parser.add_argument("-n", "--num-actors", type=int, default=3, help="number of actors")
    parser.add_argument("-b", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="batch size per block")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()

    if args.debug:
        root_logger = logging.getLogger("repartition")
        root_logger.setLevel(logging.DEBUG)
        logging.getLogger("ray.data").setLevel(logging.DEBUG)

    process(
        paths=args.input,
        batch_size=args.batch_size,
        num_actors=args.num_actors,
    )


if __name__ == "__main__":
    main()
