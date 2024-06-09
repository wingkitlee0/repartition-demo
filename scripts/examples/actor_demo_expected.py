from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import polars as pl


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="input directory or file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="../data/expected.csv",
        help="output filename",
    )
    args = parser.parse_args()

    (
        pl.read_parquet(args.input)
        .group_by("g")
        .agg(pl.len().alias("num_rows"))
        .sort("num_rows")
        .write_csv("expected.csv")
    )


if __name__ == "__main__":
    main()
