import argparse

import pyarrow as pa
import pyarrow.parquet as pq
import ray
from repartition.ray.actor_impl import (
    get_ref_bundles_from_pyarrow_dataset,
    repartition_by_column,
)
import logging

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="input directory or file")
    parser.add_argument("-n", "--num-actors", type=int, default=3, help="number of actors")
    parser.add_argument("-b", "--batch-size", type=int, default=1000, help="batch size per block")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()

    if args.debug:
        root_logger = logging.getLogger("repartition")
        root_logger.setLevel(logging.DEBUG)

    pq_ds = pq.ParquetDataset(args.input)

    ray.init()

    ref_bundles = get_ref_bundles_from_pyarrow_dataset(pq_ds, args.batch_size, None)

    repartitioned_ref_bundles, _ = repartition_by_column(ref_bundles, "g", args.num_actors)

    print(f"number of ref_bundles: {len(repartitioned_ref_bundles)}")

    results = []
    for ref_bundle in repartitioned_ref_bundles:
        print(len(ref_bundle.blocks))
        for block_ref, block_metadata in ref_bundle.blocks:
            num_rows = block_metadata.num_rows
            # block = ray.get(block_ref)
            print(num_rows)
            results.append(num_rows)

    print(f"number of blocks: {len(results)}")
    print(f"total number of rows: {sum(results)}")


if __name__ == "__main__":
    main()
