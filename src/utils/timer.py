from contextlib import contextmanager
import time
import logging

logger = logging.getLogger(__name__)


@contextmanager
def timer_context(msg: str):
    start_time = time.perf_counter()
    yield start_time
    end_time = time.perf_counter()
    logger.info("%s: %0.4f seconds", msg, end_time - start_time)
