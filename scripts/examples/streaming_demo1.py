import argparse
import pyarrow.parquet as pq

DEFAULT_BATCH_SIZE = 100_000

def read_batches(fragments, batch_size):
    for i, fragment in enumerate(fragments):
        scanner = fragment.scanner(batch_size=batch_size)
        for j, batch in enumerate(scanner.to_batches()):
            yield batch, i, j

def main(paths, batch_size, num_actors):
    pq_ds = pq.ParquetDataset(paths)

    print(f"number of fragments: {len(pq_ds.fragments)}")

    batches = list(read_batches(pq_ds.fragments, batch_size))

    print(f"number of batches: {len(batches)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="input directory or file"
    )
    parser.add_argument(
        "-n", "--num-actors", type=int, default=3, help="number of actors"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="batch size per block",
    )
    args = parser.parse_args()

    main(args.input, args.batch_size, args.num_actors)

