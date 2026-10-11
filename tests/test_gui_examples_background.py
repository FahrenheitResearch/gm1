"""The New run screen's examples are priced in the background.

GET /api/examples answers at once with what is known and whether a pricing is under way; every example is priced
by one ``gpuwm-cm1 estimate --each`` call, a line per example as it lands; the prices are kept on disk for the
first answer after a restart, keyed by the deck, the package, the measured rates, the GPU and its free memory.
The numbers are the ones ``gpuwm-cm1 estimate DIR --root ROOT --no-write --json`` prints for each example alone
(the real estimator, against one fake nvidia-smi answer).  CPU only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from gui_support import ROOT, start_server, stop_server, wait_for

GPUINFO = ROOT / "tests/data/gpuinfo"


def _gpu_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", os.fspath(GPUINFO / "fake_nvidia_smi.py"))
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", os.fspath(GPUINFO / "rtx3080-wsl2-desktop-4p8.json"))
    monkeypatch.setenv("GPUWM_CM1_WSL2", "1")
    monkeypatch.setenv("GPUWM_CM1_CACHE_DIR", os.fspath(tmp_path / "cache"))


@pytest.fixture()
def slow_fake(tmp_path, monkeypatch):
    """The fake runner, pricing each example 0.2 s apart: a pricing that takes seconds, as a real one does."""

    _gpu_env(monkeypatch, tmp_path)
    monkeypatch.setenv("GM1_FAKE_EACH_S", "0.2")
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    yield server, client, tmp_path / "runs"
    stop_server(server)


def _states(body):
    return [row["price"]["state"] for row in body["examples"]]


def _numbers(row):
    est = row["estimate"] or {}
    return (est.get("gpu_need_bytes"), est.get("smallest_card_gb"), (est.get("wall") or {}).get("value_s"),
            est.get("fits"), row["cells"], row["guided"]["offered"])


def test_the_first_answer_comes_at_once_and_the_prices_fill_in(slow_fake):
    server, client, root = slow_fake
    started = time.monotonic()
    first = client.get("/api/examples")
    took = time.monotonic() - started
    assert first.status == 200 and took < 2.0, took
    body = first.json
    assert body["pricing"] is True and len(body["examples"]) >= 20
    assert set(_states(body)) == {"pricing"}
    assert all(row["estimate"] is None and row["name"] and row["line"] is not None for row in body["examples"])
    # Asked again while it prices, some have landed and some have not: they arrive one by one.
    partial = wait_for(lambda: (lambda b: b if 0 < _states(b).count("priced") < len(b["examples"]) else None)(
        client.get("/api/examples").json), timeout=60)
    assert partial["pricing"] is True
    done = wait_for(lambda: (lambda b: b if not b["pricing"] else None)(client.get("/api/examples").json), timeout=60)
    assert set(_states(done)) == {"priced"}
    supercell = next(row for row in done["examples"] if row["id"] == "cm1-supercell")
    assert supercell["estimate"]["fits"] is True and supercell["price"]["priced_at"]
    # Each example still shows its own command, the one that prints the same numbers for it alone.
    command = supercell["commands"]["estimate"]
    assert command.startswith("gpuwm-cm1 estimate ") and "--each" not in command and command.endswith(
        f"--root {root} --no-write --json")
    again = client.get("/api/examples?wait=1", timeout=300).json
    assert [_numbers(r) for r in again["examples"]] == [_numbers(r) for r in done["examples"]]


def _count_each(monkeypatch, server) -> list[list[str]]:
    """Every ``estimate --each`` call the server makes, recorded as it is made."""

    calls: list[list[str]] = []
    real = server.jobs.host_lines

    def counted(args, *rest, **kwargs):
        calls.append([str(a) for a in args])
        return real(args, *rest, **kwargs)

    monkeypatch.setattr(server.jobs, "host_lines", counted)
    return calls


def _priced_once(tmp_path) -> dict:
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        return client.get("/api/examples?wait=1", timeout=300).json
    finally:
        stop_server(server)


def test_a_restarted_server_keeps_the_prices_made_for_the_same_key(tmp_path, monkeypatch):
    """Same deck, install, rates, GPU and free memory: the kept prices are the answer, nothing is priced again."""

    _gpu_env(monkeypatch, tmp_path)
    priced = _priced_once(tmp_path)
    kept = json.loads((tmp_path / "runs/.gm1-gui/examples/estimates.json").read_text())
    assert kept["schema"] == "gm1.gui-example-estimates/v1" and len(kept["entries"]) == len(priced["examples"])
    # The import reports stay in each example's folder; the kept file holds the estimates and their keys.
    assert all(set(entry) == {"key", "priced_at", "estimate", "estimate_rc", "estimate_command"}
               for entry in kept["entries"].values())
    monkeypatch.setenv("GM1_FAKE_EACH_S", "0.3")
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        calls = _count_each(monkeypatch, server)
        started = time.monotonic()
        first = client.get("/api/examples").json
        assert time.monotonic() - started < 2.0
        assert set(_states(first)) <= {"kept", "priced"}
        assert [_numbers(r) for r in first["examples"]] == [_numbers(r) for r in priced["examples"]]
        done = wait_for(lambda: (lambda b: b if not b["pricing"] else None)(client.get("/api/examples").json),
                        timeout=10)
        assert set(_states(done)) == {"priced"} and calls == []
        assert [_numbers(r) for r in done["examples"]] == [_numbers(r) for r in priced["examples"]]
        assert [r["commands"] for r in done["examples"]] == [r["commands"] for r in priced["examples"]]
    finally:
        stop_server(server)


@pytest.mark.parametrize("change", ["free-memory", "measured-rates"])
def test_kept_prices_are_not_used_for_another_key(tmp_path, monkeypatch, change):
    """Another free-memory bucket, or new measured rates on this machine: every example is priced afresh."""

    _gpu_env(monkeypatch, tmp_path)
    priced = _priced_once(tmp_path)
    if change == "free-memory":
        monkeypatch.setenv("GM1_FAKE_GPU_FREE_BYTES", str(8e9 - 2 * 1024 ** 3))
    else:
        rates = tmp_path / "cache" / "rates.json"
        rates.parent.mkdir(parents=True, exist_ok=True)
        rates.write_text(json.dumps({"schema": "test", "points": [1]}))
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        calls = _count_each(monkeypatch, server)
        done = client.get("/api/examples?wait=1", timeout=300).json
        assert set(_states(done)) == {"priced"} and len(calls) >= 1
        assert sum(len([a for a in call if a == "--each"]) for call in calls) == len(done["examples"])
        free = {(r["estimate"] or {}).get("gpu_free_bytes") for r in done["examples"]}
        was = {(r["estimate"] or {}).get("gpu_free_bytes") for r in priced["examples"]}
        assert (free != was) if change == "free-memory" else (free == was)
    finally:
        stop_server(server)


def test_refresh_prices_again_and_a_stale_answer_is_only_checked(tmp_path, monkeypatch):
    from cm1core.gui import setup as S

    _gpu_env(monkeypatch, tmp_path)
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        client.get("/api/examples?wait=1", timeout=300)
        calls = _count_each(monkeypatch, server)
        # Past the freshness limit the next answer starts a round that asks the GPU again; the key is the same,
        # so the prices are confirmed, not made again.
        monkeypatch.setattr(S, "EXAMPLE_PRICE_TTL_S", 0.0)
        checked = client.get("/api/examples?wait=1", timeout=300).json
        assert set(_states(checked)) == {"priced"} and calls == []
        monkeypatch.setattr(S, "EXAMPLE_PRICE_TTL_S", 30.0)
        refreshed = client.get("/api/examples?refresh=1&wait=1", timeout=300).json
        assert set(_states(refreshed)) == {"priced"}
        assert sum(len([a for a in call if a == "--each"]) for call in calls) == len(refreshed["examples"])
    finally:
        stop_server(server)


def test_the_examples_are_priced_on_two_host_slots_at_once(slow_fake, monkeypatch):
    """Imported examples: two workers, each one ``--each`` call over its share, side by side."""

    server, client, root = slow_fake
    cold = _count_each(monkeypatch, server)
    body = client.get("/api/examples?wait=1", timeout=300).json
    n = len(body["examples"])
    # A fresh server imports each example first, then prices it as soon as it is imported.
    assert len(cold) == n and all(len([a for a in call if a == "--each"]) == 1 for call in cold)
    calls = _count_each(monkeypatch, server)
    started = time.monotonic()
    client.get("/api/examples?refresh=1&wait=1", timeout=300)
    took = time.monotonic() - started
    shares = sorted(len([a for a in call if a == "--each"]) for call in calls)
    assert len(calls) == 2 and sum(shares) == n and shares[1] - shares[0] <= 1
    # 0.2 s per example in each call: two calls side by side take about half of one call over all
    assert took < 0.2 * n * 0.75, took


def test_a_runner_without_each_prices_one_by_one(tmp_path, monkeypatch):
    _gpu_env(monkeypatch, tmp_path)
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        jobs = server.jobs

        def no_each(*_args, **_kwargs):
            raise ValueError("this gpuwm-cm1 has no --each")

        monkeypatch.setattr(jobs, "host_lines", no_each)
        body = client.get("/api/examples?wait=1", timeout=300).json
        assert set(_states(body)) == {"priced"}
        supercell = next(row for row in body["examples"] if row["id"] == "cm1-supercell")
        assert supercell["estimate"]["fits"] is True
    finally:
        stop_server(server)


# ---------------------------------------------------------------- the real estimator's --each


def _cli(*args, env):
    return subprocess.run([sys.executable, "-m", "cm1core", *args], capture_output=True, text=True, env=env,
                          cwd=os.fspath(ROOT), timeout=300)


def test_each_prints_what_each_target_alone_prints(tmp_path):
    """Two packaged decks, CM1's two examples imported as the GUI imports them, and a missing target: each line
    is, word for word, what ``estimate TARGET --json`` prints for it alone (one fake nvidia-smi answer and one
    host fixture, so only the pricing's own seconds differ)."""

    env = {**os.environ, "GPUWM_CM1_NVIDIA_SMI": os.fspath(GPUINFO / "fake_nvidia_smi.py"),
           "FAKE_NVIDIA_SMI_FIXTURE": os.fspath(GPUINFO / "rtx3080-wsl2-desktop-4p8.json"),
           "GPUWM_CM1_HOST_FIXTURE": os.fspath(GPUINFO / "host-64gb-1tb.json"),
           "GPUWM_CM1_WSL2": "1", "GPUWM_CM1_CACHE_DIR": os.fspath(tmp_path / "cache"),
           "PYTHONPATH": os.pathsep.join([os.fspath(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    decks = [ROOT / "examples" / name for name in ("cartesian-initial-condition-only.toml",
                                                   "cartesian-periodic-laterals.toml")]
    assert all(d.is_file() for d in decks), decks
    root = tmp_path / "runs"
    root.mkdir()
    rundirs = []
    for name in ("supercell", "squall_line"):
        rundir = tmp_path / "examples" / name
        imported = _cli("import", os.fspath(ROOT / f"src/cm1core/data/cm1r21.1-{name}-namelist.input"),
                        "--out", os.fspath(rundir), "--json", env=env)
        assert imported.returncode == 0, imported.stderr[-2000:]
        rundirs.append(rundir)
    targets = decks + rundirs + [tmp_path / "missing.toml"]

    def written():
        return [(rundir / name).stat().st_mtime_ns if (rundir / name).exists() else None
                for rundir in rundirs for name in ("estimate.json", "geometry.json")]

    before = written()
    each = _cli("estimate", "--root", str(root), "--no-write", "--json",
                *[a for t in targets for a in ("--each", str(t))], env=env)
    assert each.returncode == 0, each.stderr
    lines = [json.loads(line) for line in each.stdout.splitlines()]
    assert [line["target"] for line in lines] == [str(t) for t in targets]
    assert lines[-1]["rc"] == 2 and lines[-1]["payload"]["status"] == "not-priced"
    for target, line in zip(targets[:-1], lines):
        alone = _cli("estimate", str(target), "--root", str(root), "--no-write", "--json", env=env)
        assert line["rc"] == alone.returncode == 0, alone.stderr[-2000:]
        want = json.loads(alone.stdout)
        got = line["payload"]
        for record in (want, got):
            record.pop("elapsed_s")
            record["gpu_query"].pop("elapsed_s")
        assert got == want, (target.name, [k for k in got if got[k] != want.get(k)])
    assert written() == before  # --no-write


def test_each_refuses_what_it_cannot_do(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([os.fspath(ROOT / "src"), os.environ.get("PYTHONPATH", "")]),
           "GPUWM_CM1_NVIDIA_SMI": "none"}
    no_json = _cli("estimate", "--each", str(tmp_path), env=env)
    assert no_json.returncode == 2 and "--json" in no_json.stderr
    with_target = _cli("estimate", str(tmp_path), "--each", str(tmp_path), "--json", env=env)
    assert with_target.returncode == 2 and "TARGET" in with_target.stderr


def _estimate(gpu_need, gpu_free, disk_need, disk_free, tiled=None):
    rec = {"tile_128_option": {"gpu_memory_bytes": tiled}} if tiled is not None else {}
    return {"gpu": {"free_bytes": gpu_free}, "host": {"disk_free_bytes": disk_free},
            "needs": {"gpu_memory_bytes": {"value": gpu_need}, "disk_bytes": {"value": disk_need}},
            "recommendation": rec}


def test_a_kept_price_whose_verdict_would_flip_inside_its_bucket_is_not_kept():
    """One 256 MiB bucket of free memory (or 1 GiB of disk) can hold both sides of a verdict's
    edge (need <= 0.95 x free; disk share 0.8 and 0.9); such a kept price is priced again."""

    from cm1core.gui.setup import DISK_BUCKET_BYTES, FREE_BUCKET_BYTES, _machine_words, _verdicts_hold

    mib = 1024 ** 2
    base = 40 * FREE_BUCKET_BYTES  # 10 GiB free, the start of a bucket
    need = 0.95 * (base + 100 * mib)  # fits at base + 100 MiB, not at base + 50 MiB
    kept = _estimate(need, base + 100 * mib, 1e9, 500e9)
    now = {"free_bytes": base + 50 * mib, "host": {"disk_free_bytes": 500e9}}
    assert _machine_words({**now, "free_bytes": base + 100 * mib}) == _machine_words(now)  # the same bucket
    assert not _verdicts_hold(kept, now)
    assert _verdicts_hold(kept, {**now, "free_bytes": base + 200 * mib})
    # the tile-128 advice's own need
    kept = _estimate(1e9, base + 100 * mib, 1e9, 500e9, tiled=need)
    assert not _verdicts_hold(kept, now)
    # the disk: 0.8 of free is a warning, 0.9 a stop
    disk_free = 400 * DISK_BUCKET_BYTES
    for fraction in (0.8, 0.9):
        disk_need = fraction * (disk_free + 250 * mib)
        kept = _estimate(1e9, base, disk_need, disk_free + 300 * mib)
        assert _verdicts_hold(kept, {"free_bytes": base, "host": {"disk_free_bytes": disk_free + 400 * mib}})
        assert not _verdicts_hold(kept, {"free_bytes": base, "host": {"disk_free_bytes": disk_free + 100 * mib}})
    # nothing to compare with: never kept
    assert not _verdicts_hold(None, now) and not _verdicts_hold(kept, None)


def test_the_key_holds_what_the_estimate_reads_from_the_gpu_and_the_host():
    """Programs on the GPU (the gpu-busy warning), more than 1 GB in use (gpu-shared), RAM, the disk: each
    changes what the estimate prints, so each is in the key."""

    from cm1core.gui.setup import DISK_BUCKET_BYTES, _machine_words

    gpu = {"uuid": "GPU-1", "name": "NVIDIA GeForce RTX 3080", "total_bytes": 10 * 1024 ** 3, "sm": "sm_86",
           "wsl2": True, "compute_apps": [], "in_use_bytes": 0.5e9, "free_bytes": 9 * 1024 ** 3,
           "host": {"ram_total_bytes": 64e9, "disk_root": "/runs", "wsl2": True,
                    "disk_free_bytes": 900 * DISK_BUCKET_BYTES}}
    words = _machine_words(gpu)
    for change in ({"compute_apps": [{"pid": 7, "name": "python"}]}, {"in_use_bytes": 1.5e9},
                   {"uuid": "GPU-2"}, {"host": {**gpu["host"], "ram_total_bytes": 32e9}},
                   {"host": {**gpu["host"], "disk_free_bytes": 899 * DISK_BUCKET_BYTES}},
                   {"host": {**gpu["host"], "disk_root": "/other"}}):
        assert _machine_words({**gpu, **change}) != words, change
    assert _machine_words({**gpu, "host": {**gpu["host"], "ram_available_bytes": 1}}) == words
