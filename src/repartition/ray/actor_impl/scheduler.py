import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskScheduler
from ray.data._internal.stats import StatsDict

from repartition.ray.actor_impl.core import apply_repartition
from repartition.ray.actor_impl.repartition_task_spec import RepartitionByColumnTaskSpec


def repartition_by_column(
    refs: List[RefBundle],
    keys: Union[str, List[str]],
    num_actors: int,
    ray_remote_args: Optional[Dict[str, Any]] = None,
):
    split_spec = RepartitionByColumnTaskSpec(keys=keys, concurrency=num_actors)
    scheduler = RepartitionByColumnTaskScheduler(split_spec)

    return scheduler.execute(
        refs=refs,
        output_num_blocks=-1,
        ctx=None,
        map_ray_remote_args=ray_remote_args,
        reduce_ray_remote_args=ray_remote_args,
    )


class RepartitionByColumnTaskScheduler(ExchangeTaskScheduler):
    """Split-by-column experiment"""

    def execute(
        self,
        refs: List[RefBundle],
        output_num_blocks: int,
        ctx: TaskContext,
        map_ray_remote_args: Optional[Dict[str, Any]] = None,
        reduce_ray_remote_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[RefBundle], StatsDict]:
        print("len(refs)", len(refs))
        print("len(refs[0].blocks)", len(refs[0].blocks))

        owns_blocks = all(rb.owns_blocks for rb in refs)
        repartitioned_blocks = []

        # Looping over N files
        for ref_id, ref in enumerate(refs):
            repartitioned_refs = asyncio.run(apply_repartition(ref_id, ref.blocks, *self._exchange_spec._map_args))

            # Looping over num_actors
            for i, blocks_or_metadata in enumerate(repartitioned_refs):
                metadata = ray.get(blocks_or_metadata.pop())
                print(f"repartition-{i}: {len(blocks_or_metadata)=}, {len(metadata)=}")

                repartitioned_blocks.append(
                    RefBundle(
                        blocks=list(zip(blocks_or_metadata, metadata)),  # list of tuples of (block, metadata)
                        owns_blocks=owns_blocks,
                    )
                )

        return repartitioned_blocks, {}
