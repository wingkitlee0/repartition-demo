import argparse

import polars as pl
import ray
import ray.data


def agg_fn(d):
    result = (
        pl.from_arrow(d)
        .group_by("g")
        .agg(
            pl.min("g").alias("min_g"),
            pl.max("g").alias("max_g"),
            pl.count().alias("row_count"),
        )
        .to_arrow()
    )
    print("len(result) = ", len(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--num-actors", type=int, default=1)
    parser.add_argument(
        "-u",
        "--use-batching",
        action="store_true",
        help="Enable batching by processing each file one by one",
    )
    parser.add_argument("--large", action="store_true", help="use large dataset")
    parser.add_argument("--full", action="store_true", help="use all files (default 1 file)")
    parser.add_argument("--output", "-o", type=str, default=None, help="output directory")
    parser.add_argument(
        "--mode",
        type=str,
        default="rbc",
        help="mode: rbc or groupby",
        choices=["rbc", "groupby"],
    )
    parser.add_argument("--materialize", "-m", action="store_true", help="materialize")
    parser.add_argument("--input", "-i", type=str, default=None, help="input directory")
    args = parser.parse_args()

    ray.init(num_cpus=args.num_actors)

    ray.data.DataContext.get_current().execution_options.preserve_order = True
    # ray.data.DataContext.get_current().target_min_block_size = 128 * 1024**3
    # ray.data.DataContext.get_current().target_max_block_size = 1024**5

    if args.input is not None:
        ray_ds = ray.data.read_parquet(args.input)
    elif args.large:
        if args.full:
            ray_ds = ray.data.read_parquet("../43_repartition_barebone/data/")
        else:
            ray_ds = ray.data.read_parquet("../43_repartition_barebone/data/data-0.parquet")
    elif args.full:
        ray_ds = ray.data.read_parquet("../43_repartition_barebone/data_small")
    else:
        ray_ds = ray.data.read_parquet("../43_repartition_barebone/data_small/data-0.parquet")

    if args.mode == "rbc":
        result = (
            ray_ds.repartition_by_column(
                "g",
                concurrency=args.num_actors,
            ).map_batches(
                fn=agg_fn,
                batch_format="pyarrow",
                # batch_size=1024**3,
                batch_size=None,
                zero_copy_batch=True,
            )
            # .sort("g")
        )
    elif args.mode == "groupby":
        result = ray_ds.groupby("g").map_groups(agg_fn, batch_format="pyarrow")
    else:
        raise NotImplementedError(f"mode {args.mode} not supported")

    if args.materialize:
        result.materialize()

    print("writing output")
    if args.output is not None:
        result.repartition(50, shuffle=False).write_parquet(args.output)
    else:
        print(result.show())
    # print(result.count())


if __name__ == "__main__":
    main()
