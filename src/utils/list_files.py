import argparse

import polars as pl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, help="directory", required=True)
    args = parser.parse_args()

    df = pl.read_parquet(f"{args.dir}/*.parquet")
    print(df)
