import argparse
import logging

import pandas as pd
import pyarrow.parquet as pq
import ray

from repartition.ray.dummy_ref_bundles import get_ref_bundles_from_pyarrow_dataset
from repartition.ray.ray_data_impl import repartition_by_column

logger = logging.getLogger(__name__)


def get_num_rows_per_block(repartitioned_ref_bundles: list[list]) -> list[int]:
    results = []
    for ref_bundle in repartitioned_ref_bundles:
        print(len(ref_bundle.blocks))
        for block_ref, block_metadata in ref_bundle.blocks:
            num_rows = block_metadata.num_rows
            # block = ray.get(block_ref)
            print(num_rows)
            results.append(num_rows)
    return results


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="input directory or file"
    )
    parser.add_argument(
        "-n", "--num-actors", type=int, default=3, help="number of actors"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=1000, help="batch size per block"
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()

    if args.debug:
        root_logger = logging.getLogger("repartition")
        root_logger.setLevel(logging.DEBUG)

    pq_ds = pq.ParquetDataset(args.input)

    ray.init()

    ref_bundles = get_ref_bundles_from_pyarrow_dataset(pq_ds, args.batch_size, None)

    repartitioned_ref_bundles, _ = repartition_by_column(
        ref_bundles, "g", args.num_actors
    )

    print(f"number of ref_bundles: {len(repartitioned_ref_bundles)}")

    results = get_num_rows_per_block(repartitioned_ref_bundles)

    print(f"number of blocks: {len(results)}")
    print(f"total number of rows: {sum(results)}")

    df = pd.Series(results, name="num_rows")
    df.sort_values(inplace=True)
    df.to_csv("num_rows.csv", index_label="id")


if __name__ == "__main__":
    main()
