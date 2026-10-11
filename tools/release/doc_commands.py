#!/usr/bin/env python3
"""Run a guide's command blocks in order, as a reader would type them (the guides' doc-test).

    python tools/release/doc_commands.py GUIDE.md [GUIDE.md ...] --work DIR --mode fake|real [--json OUT]

The guides (docs/QUICKSTART.md, docs/CM1-USERS.md) mark every fenced block with what it is:

  ```sh           typed by the reader: run here, one line at a time, in DIR (HOME is DIR/home, so ~/gm1-runs
                  stays inside DIR). Blocks run in the order they appear, and each guide starts in a new DIR.
  ```sh install   installing: tools/release/smoke_cm1_user.sh runs these against the built wheel, not this tool
  ```powershell install   the Windows installer line (PowerShell): not run here
  ```sh cluster   for another machine (ssh, sbatch, qsub, git archive): checked for form, not run
  ```sh example   a worked example on a named run the guide does not bring (a real WRF or ArWen history):
                  checked for form, not run
  ```python       run with this Python in DIR (real mode), or compiled (fake mode)
  ```text         what a command prints: not run

Files the reader brings (their own WRF history, ~/wrf/...) are made first; READER_FILES lists them.

Every line must exit 0, except `gpuwm-cm1 doctor` (0 ready, 1 something to fix) and `gpuwm-cm1 preset fit`
(0 fits, 1 not offered here, which must say why). A line with `gpuwm-cm1 gui` starts the GUI on a free port, checks
that it printed its link and answers with its token, and stops it.

--mode real   the `gpuwm-cm1` on PATH (an installed wheel) on this computer's GPU: the release's walkthrough.
--mode fake   no GPU and no model: a `gpuwm-cm1` on PATH that runs this checkout's real host commands
              (examples, import, edit, estimate, preset, jobscript, reproduce, doctor) and sends the GPU and frame
              commands (run, resume, cm1out, quicklook) and the status and control of those runs (status, stop,
              pause) to the GUI's fake runner (cm1core.gui.fake_runner), which writes the same run-folder contract
              with made-up numbers.
              nvidia-smi answers as the WSL2 RTX 3080 fixture (tests/data/gpuinfo). The CPU suite's doc-test.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
BLOCK = re.compile(r"^```([^\n`]*)\n(.*?)^```[ \t]*$", re.M | re.S)
RUN_KINDS = ("sh",)
KINDS = ("sh", "sh install", "sh cluster", "sh example", "powershell install", "python", "text")
#: gpuwm-cm1 subcommands the fake mode sends to the fake runner: the GPU and frame commands, and the status and
#: control of the runs it makes (the GUI reads and controls a fake run the same way).
FAKE_COMMANDS = ("run", "resume", "cm1out", "quicklook", "status", "stop", "pause")
#: Lines that may end 1 as well as 0, and what a 1 must print.
ALSO_ONE = {("doctor",): None, ("preset", "fit"): "Not offered"}


#: Files a guide names that the reader brings (their own WRF or ArWen history): written into DIR/home before the
#: guide runs, so its lines run as typed. tests/wrf_support.py makes a small, physically consistent WRF history
#: (three hourly records, a 3 km Lambert grid near 35 N, 97 W, CM1-hydrostatic pressure up to 20.5 km).
READER_FILES = {"~/wrf/wrfout_d01_2024-05-20_18:00:00": "wrf-history"}


def reader_files(text: str, home: Path) -> list[str]:
    """Write the reader's own files a guide names; return the ones written."""

    written = []
    for name, kind in READER_FILES.items():
        if name not in text:
            continue
        target = home / name[2:]
        target.parent.mkdir(parents=True, exist_ok=True)
        if kind == "wrf-history":
            sys.path.insert(0, str(ROOT / "tests"))
            import wrf_support

            wrf_support.write_wrfout(target, fmt="netCDF-4", producer="WRF", records=3, start="2024-05-20_18:00:00",
                                     interval_s=3600.0, map_proj=1, alpha_deg=10.0, lat0=35.2, lon0=-97.4,
                                     nx=10, ny=9, nz=40, mp_physics=10, title=" OUTPUT FROM WRF V4.6.1 MODEL")
        written.append(name)
    return written


def blocks(text: str) -> Iterator[tuple[str, str, int]]:
    """(kind, body, line number of the opening fence) for every fenced block."""

    for match in BLOCK.finditer(text):
        yield match.group(1).strip(), match.group(2), text.count("\n", 0, match.start()) + 1


def lines(body: str) -> Iterator[str]:
    """The commands of a sh block: continuation lines joined, blank and comment lines dropped."""

    pending = ""
    for raw in body.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        line = (pending + line).strip()
        pending = ""
        if line and not line.startswith("#"):
            yield line


def gpuwm_words(command: str) -> list[str]:
    """The words after `gpuwm-cm1`, or [] when the line is another program."""

    try:
        words = shlex.split(command, comments=True)
    except ValueError:
        return []
    return words[1:] if words and words[0] == "gpuwm-cm1" else []


SHIM = r'''#!{python}
"""gpuwm-cm1 for tools/release/doc_commands.py --mode fake: real host commands, the fake runner for GPU work."""
import os, subprocess, sys, time
from pathlib import Path
FAKE = {fake!r}
args = sys.argv[1:]
if args and args[0] in FAKE:
    if args[0] in ("run", "resume") and "--detach" in args:
        # As the real --detach: its own session, a log in RUNDIR/logs, back once the new segment exists.
        args = [word for word in args if word != "--detach"]
        run = Path(args[1])
        before = set(p.name for p in (run / "segments").glob("*")) if (run / "segments").is_dir() else set()
        (run / "logs").mkdir(parents=True, exist_ok=True)
        with open(run / "logs" / "supervisor-doc.log", "ab") as log:
            process = subprocess.Popen([sys.executable, "-m", "cm1core.gui.fake_runner", *args], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            now = set(p.name for p in (run / "segments").glob("*")) if (run / "segments").is_dir() else set()
            # as the real one waits for the segment's state: here its event log, written once run.json names the pid
            if any((run / "segments" / name / "events.jsonl").is_file() for name in now - before):
                break
            if process.poll() is not None:
                break
            time.sleep(0.05)
        print(f"{{run.name}}: started in the background (pid {{process.pid}})")
        sys.exit(0 if process.poll() in (None, 0) else process.returncode)
    if args[0] == "status" and "--wait" in args:
        # the real --wait, over the fake runner's status: answer once the run is not waiting or running
        import json
        args = [word for word in args if word != "--wait"]
        words = [sys.executable, "-m", "cm1core.gui.fake_runner", *args]
        if "--json" not in words:
            words.append("--json")
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            answer = subprocess.run(words, capture_output=True, text=True)
            try:
                state = json.loads(answer.stdout).get("state")
            except ValueError:
                state = None
            if state not in ("waiting", "running", "stale", "starting"):
                break
            time.sleep(0.2)
        print(answer.stdout, end="")
        sys.exit(answer.returncode)
    os.execv(sys.executable, [sys.executable, "-m", "cm1core.gui.fake_runner", *args])
os.execv(sys.executable, [sys.executable, "-m", "cm1core", *args])
'''


def fake_environment(work: Path) -> dict[str, str]:
    """PATH with the shim first, this checkout on PYTHONPATH, and the WSL2 RTX 3080 fixture as nvidia-smi."""

    bin_dir = work / ".doc-bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "gpuwm-cm1"
    shim.write_text(SHIM.format(python=sys.executable, fake=set(FAKE_COMMANDS)), encoding="utf-8")
    shim.chmod(0o755)
    gpuinfo = ROOT / "tests" / "data" / "gpuinfo"
    return {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
        "PYTHONPATH": str(ROOT / "src"),
        "GPUWM_CM1_NVIDIA_SMI": str(gpuinfo / "fake_nvidia_smi.py"),
        "FAKE_NVIDIA_SMI_FIXTURE": str(gpuinfo / "rtx3080-wsl2-desktop-4p8.json"),
        "GPUWM_CM1_WSL2": "1",
        "GPUWM_CM1_HOST_FIXTURE": str(gpuinfo / "host-64gb-1tb.json"),
        "GM1_FAKE_STEP_S": os.environ.get("GM1_FAKE_STEP_S", "0.004"),
        "GM1_FAKE_STARTUP_S": os.environ.get("GM1_FAKE_STARTUP_S", "0.3"),
    }


def run_gui(command: str, env: dict[str, str], cwd: Path, log: Path) -> tuple[int, str]:
    """Start `gpuwm-cm1 gui` on a free port, check its link and token, stop it."""

    words = shlex.split(command, comments=True) + ["--port", "0"]
    with open(log, "w", encoding="utf-8") as handle:
        server = subprocess.Popen(words, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=handle, text=True)
        try:
            deadline = time.monotonic() + 120
            port = token = None
            while time.monotonic() < deadline and port is None:
                line = server.stdout.readline() if server.stdout else ""
                if not line and server.poll() is not None:
                    break
                handle.write(line)
                match = re.search(r"http://127\.0\.0\.1:(\d+)/\?token=(\S+)", line)
                if match:
                    port, token = int(match.group(1)), match.group(2)
            if port is None:
                return 1, "the GUI printed no link"
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
            connection.request("GET", "/api/session", headers={"Host": f"127.0.0.1:{port}", "X-GM1-Token": token})
            answer = connection.getresponse()
            answer.read()
            return (0 if answer.status == 200 else 1), f"link printed; /api/session {answer.status}"
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()


def run_guide(guide: Path, work: Path, mode: str, *, timeout_s: float = 1800) -> dict[str, Any]:
    text = guide.read_text(encoding="utf-8")
    work.mkdir(parents=True, exist_ok=True)
    (work / "home").mkdir(exist_ok=True)
    env = {**os.environ, "HOME": str(work / "home")}
    for name in ("CUDA_PATH", "CUDA_HOME"):
        env.pop(name, None)
    if mode == "fake":
        env.update(fake_environment(work))
    logs = work / ".doc-logs"
    logs.mkdir(exist_ok=True)
    record: dict[str, Any] = {"guide": guide.name, "mode": mode, "blocks": [], "commands": [], "problems": []}
    record["reader_files"] = reader_files(text, work / "home")
    number = 0
    for kind, body, line_no in blocks(text):
        record["blocks"].append({"kind": kind, "line": line_no})
        if kind not in KINDS:
            record["problems"].append(f"{guide.name}:{line_no}: a block marked {kind!r}; use one of {KINDS}")
            continue
        if kind == "python":
            if mode == "fake":
                try:
                    compile(body, f"{guide.name}:{line_no}", "exec")
                    outcome = {"rc": 0, "note": "compiled (fake mode reads no real output)"}
                except SyntaxError as error:
                    outcome = {"rc": 1, "note": str(error)}
            else:
                number += 1
                started = time.monotonic()
                done = subprocess.run([sys.executable, "-c", body], cwd=work, env=env, capture_output=True,
                                      text=True, timeout=timeout_s)
                (logs / f"{number:02d}-python.out").write_text(done.stdout + done.stderr, encoding="utf-8")
                outcome = {"rc": done.returncode, "seconds": round(time.monotonic() - started, 1),
                           "tail": (done.stdout + done.stderr)[-600:]}
            record["commands"].append({"line": line_no, "kind": "python", **outcome})
            if outcome["rc"] != 0:
                record["problems"].append(f"{guide.name}:{line_no}: the python block failed: {outcome.get('tail', '')}")
            continue
        if kind == "powershell install":
            continue
        if kind in ("sh install", "sh cluster", "sh example"):
            for command in lines(body):
                try:
                    shlex.split(command, comments=True)
                except ValueError as error:
                    record["problems"].append(f"{guide.name}:{line_no}: {command!r} does not parse: {error}")
            continue
        if kind != "sh":
            continue
        for command in lines(body):
            number += 1
            words = gpuwm_words(command)
            started = time.monotonic()
            log = logs / f"{number:02d}.log"
            if words[:1] == ["gui"]:
                rc, note = run_gui(command, env, work, log)
                output = note
            else:
                done = subprocess.run(["bash", "-c", command], cwd=work, env=env, capture_output=True, text=True,
                                      timeout=timeout_s)
                rc, output = done.returncode, done.stdout + done.stderr
                log.write_text(f"$ {command}\n{output}", encoding="utf-8")
            allowed, must_say = {0}, None
            for prefix, say in ALSO_ONE.items():
                if tuple(words[: len(prefix)]) == prefix:
                    allowed, must_say = {0, 1}, say
            ok = rc in allowed and not (rc == 1 and must_say and must_say not in output)
            if "Traceback (most recent call last)" in output:
                ok = False
            record["commands"].append({"line": line_no, "command": command, "rc": rc, "ok": ok,
                                       "seconds": round(time.monotonic() - started, 1),
                                       "tail": output.strip()[-400:]})
            if not ok:
                record["problems"].append(f"{guide.name}:{line_no}: `{command}` exited {rc}: {output.strip()[-600:]}")
    record["passed"] = not record["problems"]
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("guides", nargs="+", type=Path)
    parser.add_argument("--work", type=Path, required=True, help="a new folder; each guide gets one inside it")
    parser.add_argument("--mode", choices=("fake", "real"), required=True)
    parser.add_argument("--json", type=Path, help="write the record here")
    arguments = parser.parse_args(argv)
    if arguments.mode == "real" and shutil.which("gpuwm-cm1") is None:
        print("--mode real needs gpuwm-cm1 on PATH (an installed wheel)", file=sys.stderr)
        return 2
    records = [run_guide(guide, arguments.work / guide.stem, arguments.mode) for guide in arguments.guides]
    summary = {"schema": "gpuwm-cm1.doc-commands/v1", "mode": arguments.mode, "guides": records,
               "passed": all(record["passed"] for record in records)}
    if arguments.json:
        arguments.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for record in records:
        print(f"{record['guide']}: {len(record['commands'])} commands, "
              f"{'passed' if record['passed'] else 'FAILED'}")
        for problem in record["problems"]:
            print(f"  {problem}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
