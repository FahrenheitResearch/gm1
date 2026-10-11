"""The quick-look worker's environment: CPU only, nice 19, one thread per pool.

Kept free of numerical imports, so :func:`worker_environment` runs before numpy, scipy or matplotlib is imported
and their thread pools start single-threaded.
"""

from __future__ import annotations

import os
import sys
from typing import Any

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def worker_environment() -> dict[str, Any]:
    """Make this process a sidecar: no GPU visible, nice 19, single-threaded pools; returns what it runs with."""

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in THREAD_VARIABLES:
        os.environ.setdefault(name, "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        current = os.nice(0)
        if current < 19:
            os.nice(19 - current)
    except OSError:
        pass
    return worker_facts()


def worker_facts() -> dict[str, Any]:
    try:
        nice: int | None = os.nice(0)
    except OSError:
        nice = None
    return {"nice": nice, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cupy_imported": "cupy" in sys.modules, "pid": os.getpid()}


__all__ = ["worker_environment", "worker_facts"]
