"""The GUI server never imports CuPy.

Two checks, each in a fresh interpreter:

1. A script builds the server as ``gpuwm-cm1 gui`` does (without the import guard, so the check is honest),
   drives a whole session through HTTP (session, import, check, geometry, estimate, start, the event stream to
   the end, results) and prints the CuPy modules in its ``sys.modules``: none.
2. ``gpuwm-cm1 gui --fake-runner`` runs as its own process; after the same session its ``/proc/PID/maps`` holds
   no CuPy or CUDA library.

Both run under this interpreter.  Set ``GM1_TEST_CUPY_PYTHON`` to an interpreter that has CuPy installed to run
them where an import would succeed.  CPU only: no device is opened.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPERCELL = ROOT / "tests" / "data" / "cm1import" / "r21.1-supercell.namelist.input"

SESSION = r'''
import http.client, json, sys, time
from urllib.parse import quote

def session(port, token, namelist):
    def call(method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        h = {"Host": f"127.0.0.1:{port}", "X-GM1-Token": token}
        data = None
        if body is not None:
            data = json.dumps(body); h["Content-Type"] = "application/json"
        c.request(method, path, body=data, headers=h)
        r = c.getresponse(); out = r.read(); c.close()
        assert r.status == 200, (path, r.status, out[:300])
        return json.loads(out)
    call("GET", "/api/session"); call("GET", "/api/runs")
    assert call("POST", "/api/import", {"name": "s", "namelist": {"text": namelist}})["ok"]
    for action in ("check", "geometry", "estimate", "start"):
        assert call("POST", f"/api/runs/s/{action}")["ok"], action
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("GET", "/api/runs/s/events", headers={"Host": f"localhost:{port}", "X-GM1-Token": token})
    r = c.getresponse()
    event = None
    while True:
        line = r.fp.readline().decode()
        assert line, "the stream closed early"
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        if event == "end" and line.startswith("data:"):
            break
    c.close()
    deadline = time.time() + 30
    while call("GET", "/api/runs/s/status")["state"] != "finished" and time.time() < deadline:
        time.sleep(0.1)
    results = call("GET", "/api/runs/s/results")
    assert results["cm1out"], results
'''

IN_PROCESS = SESSION + r'''
from pathlib import Path
from cm1core.gui.jobs import fake_runner
from cm1core.gui.prefs import Prefs
from cm1core.gui.server import build_server, serve_in_thread
import cm1core.cli  # the gui command's own import path
work = Path(sys.argv[1])
server = build_server(work / "runs", port=0, runner=fake_runner(), prefs=Prefs(work / "config"))
serve_in_thread(server)
session(server.port, server.token, Path(sys.argv[2]).read_text())
server.shutdown(); server.server_close()
print(json.dumps({"cupy": sorted(m for m in sys.modules if m.split(".")[0] in ("cupy", "cupyx", "cupy_backends")),
                  "numpy": "numpy" in sys.modules}))
'''


def _interpreters() -> list[str]:
    found = [sys.executable]
    extra = os.environ.get("GM1_TEST_CUPY_PYTHON")
    if extra and Path(extra).exists():
        found.append(extra)
    return found


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), *[p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]])
    env["GM1_FAKE_STEP_S"] = "0.001"
    env["GM1_FAKE_STARTUP_S"] = "0.05"
    return env


@pytest.mark.parametrize("python", _interpreters())
def test_a_whole_session_leaves_no_cupy_in_sys_modules(tmp_path, python):
    script = tmp_path / "session.py"
    script.write_text(IN_PROCESS)
    done = subprocess.run([python, str(script), str(tmp_path), str(SUPERCELL)], capture_output=True, text=True,
                          env=_env(), timeout=180)
    assert done.returncode == 0, done.stderr[-3000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["cupy"] == []
    # nothing on the server's path needs numpy either
    assert report["numpy"] is False


@pytest.mark.parametrize("python", _interpreters())
def test_the_gui_process_maps_no_cupy_or_cuda_library(tmp_path, python):
    env = _env()
    process = subprocess.Popen(
        [python, "-m", "cm1core", "gui", "--root", str(tmp_path / "runs"), "--port", "0", "--fake-runner",
         "--config-dir", str(tmp_path / "config")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
    try:
        lines = []
        while not lines or "Ctrl+C" not in lines[-1]:
            line = process.stdout.readline()
            assert line, "".join(lines) + process.stderr.read()
            lines.append(line)
        banner = "".join(lines)
        url = re.search(r"http://127\.0\.0\.1:(\d+)/\?token=(\S+)", banner)
        assert url is not None, banner
        assert "Fake runner: nothing here runs a model" in banner
        namespace: dict = {}
        exec(SESSION, namespace)  # noqa: S102 - the test's own session driver
        namespace["session"](int(url.group(1)), url.group(2), SUPERCELL.read_text())
        maps = Path(f"/proc/{process.pid}/maps").read_text()
        assert not re.search(r"cupy|libcuda|libcudart|libnvrtc", maps), \
            sorted({line.split()[-1] for line in maps.splitlines() if re.search(r"cupy|cuda|nvrtc", line)})
    finally:
        os.killpg(process.pid, signal.SIGINT)
        try:
            out, _ = process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            out, _ = process.communicate()
    assert "Runs keep going" in out


@pytest.mark.parametrize("python", _interpreters())
def test_the_server_refuses_a_cupy_import_in_its_own_process(tmp_path, python):
    code = (
        "from cm1core.gui.server import block_gpu_imports\n"
        "block_gpu_imports()\n"
        "try:\n    import cupy\nexcept ImportError as e:\n    print('refused:', e)\n"
    )
    done = subprocess.run([python, "-c", code], capture_output=True, text=True, env=_env(), timeout=60)
    assert "refused: the GUI server never imports cupy" in done.stdout, done.stderr


@pytest.mark.parametrize("python", _interpreters())
def test_the_gui_process_with_the_real_runner_maps_no_cupy(tmp_path, python):
    """The same check without --fake-runner: the real runner reads this build's command table in the server
    (cm1core.cli.build_parser) and runs the real check as a subprocess.  Whether that check passes depends
    on the interpreter's packages; the server process must hold no CuPy or CUDA library either way."""

    import http.client

    deck = ROOT / "tests/data/decks/stock-imported.gpuwm-cm1.toml"
    rundir = tmp_path / "runs" / "stock"
    rundir.mkdir(parents=True)
    (rundir / "deck.toml").write_bytes(deck.read_bytes())
    (rundir / "run.json").write_text('{"schema": "gm1.cm1-run/v1", "segments": []}')
    process = subprocess.Popen(
        [python, "-m", "cm1core", "gui", "--root", str(tmp_path / "runs"), "--port", "0",
         "--config-dir", str(tmp_path / "config")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env(), start_new_session=True)
    try:
        lines = []
        while not lines or "Ctrl+C" not in lines[-1]:
            line = process.stdout.readline()
            assert line, "".join(lines) + process.stderr.read()
            lines.append(line)
        url = re.search(r"http://127\.0\.0\.1:(\d+)/\?token=(\S+)", "".join(lines))
        port, token = int(url.group(1)), url.group(2)

        def call(method, path):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
            connection.request(method, path, headers={"Host": f"127.0.0.1:{port}", "X-GM1-Token": token})
            response = connection.getresponse()
            body = json.loads(response.read())
            connection.close()
            return response.status, body

        status, session = call("GET", "/api/session")
        assert status == 200 and session["runner"] == "real" and "check" in session["commands_available"]
        status, checked = call("POST", "/api/runs/stock/check")
        assert status == 200 and checked["command"].startswith("gpuwm-cm1 check ")
        maps = Path(f"/proc/{process.pid}/maps").read_text()
        assert not re.search(r"cupy|libcuda|libcudart|libnvrtc", maps), \
            sorted({line.split()[-1] for line in maps.splitlines() if re.search(r"cupy|cuda|nvrtc", line)})
    finally:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
