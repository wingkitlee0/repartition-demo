from __future__ import annotations

import asyncio
import itertools
import logging
import time
import typing
from collections import deque
from contextlib import contextmanager
from math import ceil
from typing import Any, TypeVar, Union

import ray
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block, BlockAccessor, BlockExecStats

from utils.timer import timer_context

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

KeyType = TypeVar("KeyType")


class BoundaryError(Exception):
    pass


def batched(blocks: list[Any], batch_size: int) -> Iterator[list[Any]]:
    """Iterates over the blocks and yields batches of objects.

    Note:
        This function can be replaced by itertools.batched when Python 3.12.
    """
    for i in range(0, len(blocks), batch_size):
        yield blocks[i : i + batch_size]


def split_single_block(
    block: Block, keys: list[str] | str
) -> Iterator[tuple[str | int, Block]]:
    """Split a single block into multiple blocks based on the key column(s).

    Args:
        block: pyarrow table or pandas DataFrame
        keys: The key column(s) to split on.

    Yields:
        Tuples of key and block. If multiple key columns are specified, the
        key is a tuple of values.
    """
    import numpy as np

    if isinstance(keys, list) and len(keys) == 1:
        keys = keys[0]

    accessor = BlockAccessor.for_block(block)

    if accessor.num_rows() == 0:
        return []

    arr = accessor.to_numpy(keys)

    if isinstance(arr, np.ndarray):
        indices = np.hstack([[0], np.where(arr[1:] != arr[:-1])[0] + 1, [len(arr)]])
        arr_ = arr
    else:
        arr_ = np.rec.fromarrays(arr.values())
        indices = np.hstack([[0], np.where(arr_[1:] != arr_[:-1])[0] + 1, [len(arr_)]])

    for start, end in itertools.pairwise(indices):
        key = arr_[start]
        key = tuple(key) if isinstance(key, np.record) else key
        yield key, accessor.slice(start, end, copy=True)


def merge_tables(keys_and_blocks: list[tuple]):
    """Merge pyarrow tables

    Neighboring tables with the same group key are concatenated. Similar to
    `itertools.groupby`, this operation is local and does not give the same
    result as a `groupby` which collects same key globally.

    Yields:
        This function yields a list of merged tables. The last element of
        the output is a list of metadata for each block.
    """
    if len(keys_and_blocks) == 0:
        return

    for key, block_iterator in itertools.groupby(keys_and_blocks, lambda x: x[0]):
        stats = BlockExecStats.builder()
        block_builder = DelegatingBlockBuilder()

        for _, b in block_iterator:
            block_builder.add_block(b)

        block = block_builder.build()

        meta = BlockAccessor.for_block(block).get_metadata(
            input_files=[],
            exec_stats=stats.build(),
        )
        yield block, meta, key


@ray.remote
class Actor:
    def __init__(self, idx: int, world_size: int, keys: str):
        self.idx = idx
        self.world_size = world_size
        self.keys = keys
        self.name = f"Actor-({self.idx})"

        self.split_queue: deque[tuple[int, ray.ObjectRef]] = deque()

        # For exchange boundary
        # if not left most, there is a left bucket
        self.left_bucket = None if self.idx == 0 else asyncio.Queue(1)
        # if not right most, there is a right bucket
        self.right_bucket = (
            None if (self.idx == self.world_size - 1) else asyncio.Queue(1)
        )  # local only

        self.right_actor = None
        self.right_actor_ready = (
            asyncio.Event() if self.right_bucket is not None else None
        )

        # Indicate it's ready to consume
        self.consume_ready = asyncio.Event()

        # For output
        self._num_output_blocks = 0
        self.output_queue = asyncio.Queue()

        # For logging
        self._input_num_rows = 0
        self._split_num_rows = 0
        self._merge_num_rows = 0

    def __repr__(self):
        return self.name

    def clear(self):
        self.split_queue.clear()

        if self.left_bucket is not None:
            self.left_bucket = asyncio.Queue(1)

        if self.right_bucket is not None:
            self.right_bucket = asyncio.Queue(1)

        self.right_actor = None

        self.consume_ready.clear()

        # For output
        self._num_output_blocks = 0
        self.output_queue = asyncio.Queue()
        # For logging
        self._input_num_rows = 0
        self._split_num_rows = 0
        self._merge_num_rows = 0

    def set_right_actor(self, right_actor):
        self.right_actor = right_actor
        self.right_actor_ready.set()
        logger.debug("done set_right_actor(%s)", self.name)

    async def process(self, blocks: list[ray.ObjectRef]):
        """Process the blocks"""

        logger.debug("process: number of blocks: %d", len(blocks))
        logger.debug("process: type(block): %s", type(blocks[0]))

        # split blocks and handle boundary
        await self.split_blocks(blocks)

        # merge blocks
        await self.merge_blocks()

    async def split_blocks(
        self, block_refs: list[ray.ObjectRef]
    ) -> list[ray.ObjectRef]:
        """Split a list of blocks based on the group key.

        This is the map task that generates multiple sub-blocks for each input block,
        depending on the group key.
        """
        blocks = await asyncio.gather(*block_refs)

        for block in blocks:
            self._input_num_rows += len(block)
            for key, blk in split_single_block(block, self.keys):
                self._split_num_rows += len(blk)
                self.split_queue.append((key, blk))

        # handle left boundary: take the first sub-block
        # if it's not the left-most actor
        if self.left_bucket is not None:
            key, blk = self.split_queue.popleft()
            self.left_bucket.put_nowait((key, blk))

        # handle right boundary: take the last sub-block
        if self.right_bucket is not None:
            key, blk = self.split_queue.pop()
            self.right_bucket.put_nowait((key, blk))

        # TODO: this flag may be removed when it is converted into
        # a fully streaming operation
        self.consume_ready.set()

    async def merge_blocks(self):
        await self.consume_ready.wait()

        for block, meta, key in merge_tables(self.split_queue):
            self.output_queue.put_nowait((block, meta, key))
        self._num_output_blocks = self.output_queue.qsize()

        if self.right_bucket is not None:
            await self.merge_right_blocks()

        self.output_queue.put_nowait("done")
        # tend = time.perf_counter()
        # logger.info(f"{self.name}-merge-blocks: time = {(tend-tstart):0.3f}s")

    async def send_to_left(self):
        """Send the left item to the left actor."""
        if self.left_bucket is None:
            raise BoundaryError("Left bucket does not exist")
        return await self.left_bucket.get()

    async def merge_right_blocks(self):
        """Merge the right-most blocks"""

        if (
            self.right_bucket is None
            or self.right_actor is None
            or self.right_actor_ready is None
        ):
            raise ValueError("Right bucket does not exist")

        await self.right_actor_ready.wait()
        logger.info(f"{self.name}-merge_right_blocks: Get value from right actor")

        # current actor's right is left of the right actor's left
        right_key, right_blk = await self.right_actor.send_to_left.remote()
        left_key, left_block = await self.right_bucket.get()

        for blk, meta, key in merge_tables(
            [(left_key, left_block), (right_key, right_blk)]
        ):
            self.output_queue.put_nowait((blk, meta, key))

        # update
        self._num_output_blocks = self.output_queue.qsize()

    async def consume(self):
        """Consume the output queue

        It returns N+2 items, where N is the number of output blocks.
        """
        all_blocks, all_metadata, all_keys = [], [], []
        while True:
            item = await self.output_queue.get()
            if item == "done":
                print(f"{len(all_blocks)=}")
                return *all_blocks, all_metadata, all_keys
            block, meta, key = item
            all_blocks.append(block)
            all_metadata.append(meta)
            all_keys.append(key)

    def get_num_output_blocks(self):
        return self._num_output_blocks

    def get_input_num_rows(self):
        return self._input_num_rows

    def get_split_num_rows(self):
        return self._split_num_rows

    async def consume_stream(self):
        """Consume the output queue"""
        all_metadata, all_keys = [], []
        block_count = 0
        while True:
            item = await self.output_queue.get()
            if item == "done":
                print(f"{block_count=}")
                yield all_metadata
                yield all_keys

            block, meta, key = item
            yield block
            block += 1
            all_metadata.append(meta)
            all_keys.append(key)


def setup_connected_actors(num_actors: int, keys: Union[list[str], str]) -> list[Actor]:
    actors: list[Actor] = [Actor.remote(i, num_actors, keys) for i in range(num_actors)]

    add_right = [
        left.set_right_actor.remote(right) for left, right in itertools.pairwise(actors)
    ]

    ray.get(add_right)
    return actors


def retreive_results(actors):
    """Retreive the results from the actors.

    Since we do not know the number of blocks in advance, we need
    to call `get_num_output_blocks` to get the number of blocks.
    The rest of the function is simply rearranging the output
    without materializing the object references.
    """
    num_ouput_blocks = ray.get(
        [actor.get_num_output_blocks.remote() for actor in actors]
    )

    refs = [
        actor.consume.options(num_returns=num_blocks + 2).remote()
        for num_blocks, actor in zip(num_ouput_blocks, actors, strict=True)
    ]

    output_blocks, output_metadata, output_keys = [], [], []
    for refs_per_actor in refs:
        output_keys.append(refs_per_actor.pop())
        output_metadata.append(refs_per_actor.pop())
        # output_blocks.extend(refs_per_actor)
        yield from refs_per_actor

    # yield each blocks
    # yield from output_blocks
    # yield 2*K lists of metadata and keys
    yield from output_metadata
    yield from output_keys


@ray.remote(num_cpus=0)
class Repartitioner:
    def __init__(self, num_actors: int, keys: Union[list[str], str]):
        self.num_actors = num_actors
        self.keys = keys
        with timer_context("setup actors"):
            self.actors = setup_connected_actors(num_actors, keys)

    def clear(self):
        ray.get([actor.clear.remote() for actor in self.actors])

    def run_repartition(
        self,
        ref_id,
        blocks,
    ):
        """
        Yields:
            Assuming K actors, this function first yields each block's object reference
            individually. Then it yields K refs from each actor for the metadata, i.e.,
            K lists of metadata. Finally, it yields K refs from each actor for the keys,
            i.e., K lists of keys.
        """
        if len(blocks) <= self.num_actors:
            num_actors = 1
        else:
            num_actors = self.num_actors

        num_blocks_per_actor = ceil(len(blocks) / num_actors)
        logger.info(
            "repartition_runner(%d): len(blocks)=%d, num_actors=%d, num_blocks_per_actor=%d",
            ref_id,
            len(blocks),
            num_actors,
            num_blocks_per_actor,
        )

        process_tasks = [
            self.actors[i].process.remote(batch_per_actor)
            for i, batch_per_actor in enumerate(batched(blocks, num_blocks_per_actor))
        ]

        logger.debug("type(process_tasks[0]) = %s", type(process_tasks[0]))

        with timer_context("retreive results from actors: taken "):
            ray.get(process_tasks)
            yield from retreive_results(self.actors)
            # yield from self.retreive_results()

    def retreive_results(self):
        """Retreive the results from the actors.

        Since we do not know the number of blocks in advance, we need
        to call `get_num_output_blocks` to get the number of blocks.
        The rest of the function is simply rearranging the output
        without materializing the object references.
        """

        output_metadata, output_keys = [], []
        for actor in self.actors:
            refs_per_actor = ray.get(list(actor.consume_stream.remote()))

            output_keys.append(refs_per_actor.pop())
            output_metadata.append(refs_per_actor.pop())
            # yield the remainging items (blocks)
            yield from refs_per_actor

        # yield 2*K lists of metadata and keys
        yield from output_metadata
        yield from output_keys
