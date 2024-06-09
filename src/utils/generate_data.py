import argparse
import os
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class RandomDataCreator:
    npartitions: int = 5
    ngroups: int = 5
    nmin_per_group: int = 2
    nmax_per_group: int = 4
    ordered: bool = True

    def generate_date_list(self, seed: int | None = None) -> list[np.ndarray]:
        rng = np.random.RandomState(seed)

        all_data = []
        for i in range(self.npartitions):
            groups = i * self.ngroups + np.arange(
                self.ngroups
            )  # groups in current partition

            if not self.ordered:
                rng.shuffle(groups)

            repeats = rng.randint(
                self.nmin_per_group, self.nmax_per_group, len(groups)
            )  # lengths of each group
            data = np.repeat(groups, repeats)
            all_data.append(data)

        return all_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", default=1234, type=int, help="random seed")
    parser.add_argument("--out", "-o", required=True, help="output dir")
    parser.add_argument(
        "--npartitions", "-n", default=5, type=int, help="number of partitions"
    )
    parser.add_argument(
        "--ngroups", "-g", default=5, type=int, help="number of groups per partition"
    )
    parser.add_argument(
        "--nmin", default=5, type=int, help="max number of rows per group"
    )
    parser.add_argument(
        "--nmax", default=10, type=int, help="max number of rows per group"
    )
    parser.add_argument(
        "--unordered",
        action="store_true",
        help="do not order groups (within partition)",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    creator = RandomDataCreator(
        npartitions=args.npartitions,
        ngroups=args.ngroups,
        nmin_per_group=args.nmin,
        nmax_per_group=args.nmax,
        ordered=not args.unordered,
    )
    data_list = creator.generate_date_list(args.seed)

    tables = []
    for arr in data_list:
        data = {
            "g": arr,
            "x": np.random.rand(len(arr)),
        }
        table = pa.Table.from_pydict(data)
        tables.append(table)

    for i, t in enumerate(tables):
        pq.write_table(t, f"{args.out}/data-{i}.parquet")


if __name__ == "__main__":
    main()
