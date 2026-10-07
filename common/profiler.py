"""Efficiency logging: latency and token/cost per query.

Wraps an inference call to record wall-clock latency and token counts so the
accuracy / cost / latency axes all come from one evaluation pass.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass
class ProfileRecord:
    """Efficiency measurements for a single query.

    Attributes:
        latency_s: Wall-clock seconds for the inference call.
        input_tokens: Number of prompt tokens (if available).
        output_tokens: Number of generated tokens (if available).
    """

    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None


def timed(fn: Callable[[], T]) -> tuple[T, float]:
    """Run ``fn`` and return its result alongside wall-clock latency.

    Uses ``time.perf_counter`` (monotonic, high-resolution) as appropriate for
    duration measurement.

    Args:
        fn: A zero-argument callable performing the inference.

    Returns:
        A ``(result, latency_seconds)`` tuple.
    """
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start
