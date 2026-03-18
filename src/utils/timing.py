from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class TimerResult:
    name: str
    start_time: float
    end_time: Optional[float] = None
    elapsed_s: Optional[float] = None


@contextmanager
def timer(name: str) -> Iterator[TimerResult]:
    print(f"[timer] Starting timer for: {name}")
    result = TimerResult(name=name, start_time=time.perf_counter())
    try:
        yield result
    finally:
        result.end_time = time.perf_counter()
        result.elapsed_s = result.end_time - result.start_time
        print(f"[timer] Finished timer for: {name}, elapsed_s={result.elapsed_s:.6f}")


def measure_once(name: str, fn, *args, **kwargs):
    print(f"[measure_once] Measuring function: {name}")
    with timer(name) as t:
        output = fn(*args, **kwargs)
    print(f"[measure_once] Done measuring function: {name}")
    return output, t.elapsed_s
