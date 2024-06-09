import argparse
import asyncio
import pyarrow.parquet as pq


async def main(args):
    # def main(args):

    paths = args.input
    num_files = args.num_files
    batch_size = args.batch_size

    pq_ds = pq.ParquetDataset(args.input)

    if num_files is None:
        fragments = pq_ds.fragments
    else:
        if len(pq_ds.fragments) < num_files:
            raise ValueError(
                f"Number of files {len(pq_ds.fragments)} is less than {num_files}"
            )
        fragments = pq_ds.fragments[:num_files]

    for i, fragment in enumerate(fragments):
        scanner = fragment.scanner(batch_size=batch_size)
        print(scanner.count_rows(), scanner.dataset_schema)

        print(i, scanner.to_reader())

        reader = scanner.to_reader()

        # for batch in reader.read_next_batch():
        for batch, _ in scanner.scan_batches():
            print(len(batch), type(batch))

        reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="input directory or file"
    )
    parser.add_argument(
        "--num-files", "-nf", type=int, default=3, help="number of files"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=100_000, help="batch size per block"
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()

    asyncio.run(main(args))

    # main(args)
