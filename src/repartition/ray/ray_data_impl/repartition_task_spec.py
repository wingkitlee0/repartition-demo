from __future__ import annotations

from typing import TypeVar

from ray.data._internal.planner.exchange.interfaces import ExchangeTaskSpec
from ray.data.block import Block, BlockMetadata

T = TypeVar("T")


class RepartitionByColumnTaskSpec(ExchangeTaskSpec):
    """Example ExchangeTaskSpec"""

    SPLIT_SUB_PROGRESS_BAR_NAME = "Split blocks by column"
    MERGE_SUB_PROGRESS_BAR_NAME = "Merge blocks by column"

    def __init__(
        self,
        keys: str | list[str],
        concurrency: int | None,
    ):
        super().__init__(
            map_args=[keys, concurrency],
            reduce_args=[keys],
        )

    @staticmethod
    def map(
        idx: int,
        block: Block,
        output_num_blocks: int,
        keys: str | list[str],
    ) -> list[BlockMetadata | Block]:
        pass

    @staticmethod
    def reduce(
        keys: str | list[str],
        *mapper_outputs: list[Block],
        partial_reduce: bool = False,
    ) -> tuple[Block, BlockMetadata]:
        pass
