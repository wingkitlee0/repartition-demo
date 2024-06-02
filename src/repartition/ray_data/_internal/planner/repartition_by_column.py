from __future__ import annotations

import logging
import typing
from functools import partial
from typing import Any

from ray.data._internal.execution.interfaces import RefBundle, TaskContext

# from ray.data._internal.planner.exchange.repartition_task_scheduler import (
#     RepartitionByColumnTaskScheduler,
# )
# from ray.data._internal.planner.exchange.repartition_task_spec import (
#     RepartitionByColumnTaskSpec,
# )
from ray.data._internal.stats import StatsDict
from repartition.ray_data._internal.planner.exchange.repartition_task_scheduler import RepartitionByColumnTaskScheduler
from repartition.ray_data._internal.planner.exchange.repartition_task_spec import RepartitionByColumnTaskSpec

if typing.TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import AllToAllTransformFn



logger = logging.getLogger(__name__)


def generate_repartition_by_column_fn(
    keys: str | list[str],
    concurrency: int | None,
    ray_remote_args: dict[str, Any] | None,
) -> AllToAllTransformFn:
    """Generate function to split blocks by the specified key column"""

    def fn(
        refs: list[RefBundle],
        ctx: TaskContext,
        keys: str | list[str],
        concurrency: int | None,
        ray_remote_args: dict[str, Any] | None = None,
    ) -> tuple[list[RefBundle], StatsDict]:
        repartition_task_spec = RepartitionByColumnTaskSpec(
            keys=keys,
            concurrency=concurrency,
        )
        scheduler = RepartitionByColumnTaskScheduler(repartition_task_spec)

        return scheduler.execute(
            refs=refs,
            output_num_blocks=-1,
            ctx=ctx,
            map_ray_remote_args=ray_remote_args,
            reduce_ray_remote_args=ray_remote_args,
        )

    return partial(
        fn,
        keys=keys,
        concurrency=concurrency,
        ray_remote_args=ray_remote_args,
    )
