"""Stand-ins for ``python -m cm1core cartesian`` and ``nvidia-smi`` (tests of tools/nest_chain.py).

Run as scripts.  The fake runtime writes the published layout the real one
writes (``frame-NNNNNN`` with the carriers, the CM1FHS1 envelope and the
location record; ``nest-child/frame-NNNNNN``; ``restart-NNNNNN`` with its
publication record) at the deck's history and restart intervals, and fails
the way ``FAKE_RUNTIME_PLAN`` says for this call number.  The fake
``nvidia-smi`` answers the two queries the chain makes from a JSON state file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import struct
import sys
import time
import tomllib

ARENA_BYTES = 4096


def _write_frame(directory: Path, index: int, t: float, *, truncate: bool = False) -> None:
    staged = directory.parent / ".incomplete" / directory.name
    staged.mkdir(parents=True, exist_ok=True)
    payload = bytes(range(256)) * (ARENA_BYTES // 256)
    arena = payload[: ARENA_BYTES - 100] if truncate else payload
    (staged / "history-carriers.f32").write_bytes(arena)
    header = json.dumps({"fake": True, "t": t}).encode()
    (staged / "history.cm1fhs").write_bytes(b"CM1FHS1\0" + struct.pack("<Q", len(header)) + header + payload)
    (staged / "history-field-table.tsv").write_text("fake\n")
    (staged / "history-publication.json").write_text(json.dumps({"model_time_s": t}))
    (staged / "history-frame-location.json").write_text(json.dumps({
        "arena_nbytes": ARENA_BYTES, "index": index, "model_time_s": t,
        "arena_sha256": __import__("hashlib").sha256(payload).hexdigest(),
    }))
    directory.parent.mkdir(parents=True, exist_ok=True)
    staged.rename(directory)


def _write_restart(directory: Path, number: int, t: float, nested: bool) -> None:
    directory.mkdir(parents=True)
    (directory / "restart.cm1fcp").write_bytes(b"R" * 2048)
    if nested:
        (directory / "nest-child").mkdir()
        (directory / "nest-child" / "restart.cm1fcp").write_bytes(b"C" * 1024)
    (directory / "restart-publication.json").write_text(json.dumps(
        {"model_time_s": t, "restart_number": number}))


def runtime(argv: list[str]) -> int:
    deck_path = Path(argv[0])
    args = dict(zip(argv[1::1], argv[2::1]))
    deck = tomllib.loads(deck_path.read_text())
    frames = Path(args["--history-dir"])
    restarts = Path(args["--restart-dir"])
    plan_path = Path(os.environ["FAKE_RUNTIME_PLAN"])
    plan = json.loads(plan_path.read_text())
    call = plan.setdefault("count", 0)
    plan["count"] = call + 1
    plan.setdefault("calls", []).append({
        "argv": argv, "deck": deck, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid()})
    plan_path.write_text(json.dumps(plan))
    behaviour = (plan.get("legs") or [{}])[min(call, len(plan.get("legs") or [{}]) - 1)]
    run = deck["run"]
    nested = "nest" in deck
    dt = float(run["dt_s"])
    t_end = float(run["duration_s"])
    start = 0.0
    number = 0
    if int(run.get("irst", 0)) == 1:
        rst = int(run["rstnum"])
        record = Path(args["--restart-from"]) / f"restart-{rst:06d}" / "restart-publication.json"
        start = float(json.loads(record.read_text())["model_time_s"])
        number = rst
    terminated = []
    signal.signal(signal.SIGTERM, lambda *_: terminated.append(1))
    hist = float(run["history_interval_s"])
    rsti = float(run["restart_interval_s"])
    frame_index = int(round(start / hist))
    fail_at = behaviour.get("fail_at_s")
    t = start
    step = 0
    while t < t_end - 1e-9:
        t = round(t + dt, 9)
        step += 1
        time.sleep(float(behaviour.get("step_sleep_s", 0.0)))
        if terminated:
            print("gm1: SIGTERM: stopping", flush=True)
            Path(os.environ["FAKE_RUNTIME_PLAN"] + ".sigterm").write_text(str(os.getpid()))
            return 143
        if abs(t / hist - round(t / hist)) < 1e-9:
            frame_index += 1
            _write_frame(frames / f"frame-{frame_index:06d}", frame_index, t,
                         truncate=bool(behaviour.get("truncate_f32")))
            if nested:
                _write_frame(frames / "nest-child" / f"frame-{frame_index:06d}", frame_index, t)
            print(f"frame {frame_index} step {step} t={t}s -> {frames}/frame-{frame_index:06d}", flush=True)
        if abs(t / rsti - round(t / rsti)) < 1e-9:
            number += 1
            _write_restart(restarts / f"restart-{number:06d}", number, t, nested)
        print(f"gm1 progress: step {step} t={t:g}s | 0.500 s/step recent, 0.{400 + step:03d} mean", flush=True)
        if fail_at is not None and t >= fail_at - 1e-9:
            kind = behaviour.get("fail", "other")
            message = {
                "oom": "cupy.cuda.memory.OutOfMemoryError: Out of memory allocating 1,048,576 bytes",
                # verbatim from the RTX 5090 on 2026-09-22 (a 1500 MiB registry budget)
                "registry_oom": ("gpuwm-cm1: CM1-CARTESIAN-RUN-FAILED [cartesian-runtime]: the "
                                 "Cartesian runtime stopped during execution (MemoryError): declared "
                                 "dynamics phase payload exceeds the explicit registry budget"),
                "nan": "RuntimeError: completed model state contains nonfinite physical words",
                "courant": "gpuwm-cm1: CM1-CARTESIAN-COURANT-STOP: Courant number has exceeded 1.5",
                "other": "gpuwm-cm1: CM1-CARTESIAN-RUN-FAILED: something else",
            }[kind]
            print(message, flush=True)
            return 2
    print(f"integrated to {t_end} model seconds", flush=True)
    return 0


def nvidia_smi(argv: list[str]) -> int:
    state_path = Path(os.environ["FAKE_SMI_STATE"])
    state = json.loads(state_path.read_text())
    query = next(a for a in argv if a.startswith("--query"))
    if query.startswith("--query-gpu"):
        sequence = state.get("free_sequence") or []
        free = sequence.pop(0) if len(sequence) > 1 else (sequence[0] if sequence else state["free"])
        state["free_sequence"] = sequence
        state_path.write_text(json.dumps(state))
        total = state.get("total", 32607)
        print(f"{state['uuid']}, NVIDIA GeForce RTX 5090, {total}, {total - free}, {free}, 7")
        print("GPU-other, NVIDIA GeForce RTX 3080, 10240, 0, 10240, 0")
    else:
        for pid, name, mib in state.get("apps", []):
            print(f"{state['uuid']}, {pid}, {name}, {mib}")
    return 0


if __name__ == "__main__":
    which = sys.argv[1]
    sys.exit(runtime(sys.argv[2:]) if which == "runtime" else nvidia_smi(sys.argv[2:]))
