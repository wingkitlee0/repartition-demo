import pyarrow.parquet as pq


async def get_batches(paths, num_files: int | None = None):
    pq_ds = pq.ParquetDataset(paths)

    if num_files is None:
        fragments = pq_ds.fragments
    else:
        if len(pq_ds.fragments) < num_files:
            raise ValueError(f"Number of files {len(pq_ds.fragments)} is less than {num_files}")
        fragments = pq_ds.fragments[:num_files]

    for i, fragment in enumerate(fragments):
        scanner = fragment.scanner()
        print(scanner.count_rows(), scanner.dataset_schema)

        yield scanner, i
