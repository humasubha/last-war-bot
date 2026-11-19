from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Metrics:
    started: float = field(default_factory=time.perf_counter)
    _cpu_start: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            self._cpu_start = sum(proc.cpu_times()[:2])
        except ImportError:
            pass
    capture_ms: list[float] = field(default_factory=list)
    detection_ms: list[float] = field(default_factory=list)
    frames: int = 0
    skipped: int = 0

    def report(self) -> dict[str, object]:
        elapsed = max(1e-9, time.perf_counter() - self.started)
        def summary(values: list[float]) -> dict[str, float]:
            if not values:
                return {"average": 0.0, "p50": 0.0, "p95": 0.0}
            a = np.asarray(values)
            return {"average": float(a.mean()), "p50": float(np.percentile(a, 50)),
                    "p95": float(np.percentile(a, 95))}
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            memory_mb = proc.memory_info().rss / (1024 * 1024)
            cpu_seconds = sum(proc.cpu_times()[:2])
            if self._cpu_start is None:
                self._cpu_start = cpu_seconds
            cpu = 100 * max(0.0, cpu_seconds - self._cpu_start) / elapsed
        except ImportError:
            memory_mb = None
            cpu = None
        return {"duration_seconds": elapsed, "frames_captured": self.frames,
                "frames_per_second": self.frames / elapsed,
                "unchanged_skip_percent": 100 * self.skipped / max(1, self.frames),
                "capture_latency_ms": summary(self.capture_ms),
                "change_detection_ms": summary(self.detection_ms),
                "process_cpu_percent_sample": cpu, "process_rss_mb": memory_mb,
                "resource_note": "CPU is process CPU seconds divided by elapsed wall time; RSS is process resident memory."}
