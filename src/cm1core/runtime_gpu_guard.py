"""Coordinate GM1 with the UUID lock shared by the ArWen ecosystem."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import time
import uuid


def require_exclusive(receipt: object, gpu_uuid: str) -> None:
    """Refuse a card another CUDA process is using.

    gpuwm 2.7.4's ``preflight_exclusive_gpu`` raised on a CUDA co-tenant.
    From 2.8.0 it prices the co-tenant against the device's free memory and
    returns an admission receipt instead, admitting the run when GM1 passes no
    reservation.  GM1's runtime guard is exclusive, so any verdict other than
    ``"exclusive"`` is refused here, with the receipt's own words.  ``None``
    is the 2.7.4 return value, which only arrives when there was no contender.
    Every GM1 caller of ``preflight_exclusive_gpu`` passes its return value
    through here.
    """
    if receipt is None:
        return
    verdict = receipt.get("verdict") if isinstance(receipt, dict) else None
    if verdict == "exclusive":
        return
    from gpuwm.supervisor import GPUPreflightError

    cotenants = receipt.get("cotenants") if isinstance(receipt, dict) else None
    raise GPUPreflightError(
        f"GPU {gpu_uuid} is in use by another CUDA process ({cotenants or 'unnamed'}); "
        f"GM1 runs only on a card it has to itself (preflight verdict {verdict!r}). "
        "Stop the other process or choose another GPU."
    )


@contextmanager
def exclusive_runtime_gpu(gpu_uuid: str, *, directory_lock: Path | None = None):
    from gpuwm.supervisor import GPUFileLock, preflight_exclusive_gpu

    def preflight() -> None:
        require_exclusive(
            preflight_exclusive_gpu(gpu_uuid, approved_pids={os.getpid()}), gpu_uuid)

    token = f"gm1-runtime|pid={os.getpid()}|{uuid.uuid4()}"
    owner = None
    with GPUFileLock(gpu_uuid, run_id=token):
        if directory_lock is not None:
            directory_lock = directory_lock.resolve()
            directory_lock.parent.mkdir(parents=True, exist_ok=True)
            directory_lock.mkdir(exist_ok=False)
            owner = directory_lock / "owner"
            owner.write_text(token, encoding="utf-8")
        try:
            preflight()
            time.sleep(2)
            preflight()
            yield token
            preflight()
        finally:
            if owner is not None:
                if owner.read_text(encoding="utf-8") != token:
                    raise RuntimeError("GPU mutex ownership changed; refusing to release another owner")
                owner.unlink()
                directory_lock.rmdir()

