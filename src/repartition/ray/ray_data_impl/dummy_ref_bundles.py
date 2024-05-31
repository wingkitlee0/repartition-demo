"""Mimic what Ray Data does"""

import logging
from venv import logger

import pyarrow as pa
import pyarrow.dataset
import pyarrow.parquet as pq
import ray
from ray.data._internal.execution.interfaces import RefBundle
from ray.data.block import BlockAccessor, BlockExecStats

logger = logging.getLogger(__name__)


@ray.remote(num_returns="streaming")
def get_blocks_and_metadata_from_fragment(
    fragment: pyarrow.dataset.ParquetFileFragment,
    batch_size: int,
    columns: list[str],
):
    batches = fragment.to_batches(
        batch_size=batch_size,
        columns=columns,
    )
    metadata = []
    for batch in batches:
        stats = BlockExecStats.builder()
        table = pa.Table.from_batches([batch])
        meta = BlockAccessor.for_block(table).get_metadata(
            input_files=None,
            exec_stats=stats.build(),
        )
        metadata.append(meta)
        yield table
    yield metadata


def get_ref_bundles_from_pyarrow_dataset(
    pq_ds: pq.ParquetDataset,
    batch_size: int,
    columns: list[str],
) -> list[RefBundle]:

    logger.debug(f"{len(pq_ds.fragments)} fragments")

    ref_bundles = []
    for fragment in pq_ds.fragments:
        blocks_and_metadata = list(get_blocks_and_metadata_from_fragment.remote(fragment, batch_size, columns))
        block_refs, metadata_ref = blocks_and_metadata[:-1], blocks_and_metadata[-1]
        metadata = ray.get(metadata_ref)  # a list
        blocks_with_metadata = list(zip(block_refs, metadata, strict=True))
        ref_bundles.append(
            RefBundle(
                blocks=blocks_with_metadata,
                owns_blocks=True,
            )
        )

    return ref_bundles
