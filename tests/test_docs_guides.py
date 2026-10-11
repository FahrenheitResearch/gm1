"""The guides are doc-tested.

1. Every command block of docs/QUICKSTART.md and docs/CM1-USERS.md runs, in order, on CM1's stock supercell
   namelist (tools/release/doc_commands.py --mode fake: the real host commands, the GUI's fake runner for the GPU
   work, the WSL2 RTX 3080 fixture as nvidia-smi); and `gpuwm-cm1 preset fit` runs for every preset and budget on
   fake-GPU fixtures. The release is checked with the same guides in --mode real on the RTX 3080.
2. No sentence claims cluster-core equivalence.
3. docs/QUICKSTART.md passes guided mode's copy lint, and docs/COMMANDS.md lists every command the API table can
   return (it is generated from that table).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys

import pytest

from cm1core import __version__
from cm1core.cm1import import namelist as cm1_namelist

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = ROOT / "docs" / "QUICKSTART.md"
CM1_USERS = ROOT / "docs" / "CM1-USERS.md"
COMMANDS = ROOT / "docs" / "COMMANDS.md"
RELEASE_DOCS = (QUICKSTART, CM1_USERS, COMMANDS, ROOT / "docs" / "INSTALL.md", ROOT / "docs" / "RELEASE-NOTES.md",
                ROOT / "docs" / "TROUBLESHOOTING.md", ROOT / "README.md")
DOC_COMMANDS = runpy.run_path(str(ROOT / "tools" / "release" / "doc_commands.py"))
GPUINFO = ROOT / "tests" / "data" / "gpuinfo"


def _parser_available() -> bool:
    try:
        cm1_namelist.locate_tool()
    except Exception:  # noqa: BLE001 - any failure to find the parser means the importer cannot run here
        return False
    return True


needs_parser = pytest.mark.skipif(not _parser_available(),
                                  reason="the cm1-r21-namelist tool is not built (cargo build in rust/cm1-state)")


@needs_parser
@pytest.mark.parametrize("guide", [QUICKSTART, CM1_USERS], ids=lambda path: path.name)
def test_every_command_block_runs_in_order(guide: Path, tmp_path: Path) -> None:
    record = DOC_COMMANDS["run_guide"](guide, tmp_path / guide.stem, "fake", timeout_s=600)
    assert record["passed"], "\n".join(record["problems"])
    ran = [row for row in record["commands"] if "command" in row]
    assert len(ran) >= (6 if guide == QUICKSTART else 20)
    kinds = {block["kind"] for block in record["blocks"]}
    assert kinds <= set(DOC_COMMANDS["KINDS"])


@needs_parser
def test_the_cm1_users_guide_walks_import_run_pause_resume_and_output(tmp_path: Path) -> None:
    """The fake run folders the guide leaves are the ones a reader would have."""

    record = DOC_COMMANDS["run_guide"](CM1_USERS, tmp_path / "g", "fake", timeout_s=600)
    assert record["passed"], "\n".join(record["problems"])
    runs = tmp_path / "g" / "runs"
    supercell = json.loads((runs / "supercell" / "run.json").read_text(encoding="utf-8"))
    assert len(supercell["segments"]) == 2  # run --until 1h, then resume to timax
    paused = runs / "paused"
    ends = [json.loads(line) for segment in sorted((paused / "segments").iterdir())
            for line in (segment / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if '"kind": "end"' in line or '"kind":"end"' in line]
    assert [end.get("reason") for end in ends][:1] == ["paused"]
    assert (runs / "supercell" / "job-slurm.sh").is_file()
    assert (runs / "supercell" / "reproduce").is_dir()


PRESET_FIXTURES = ("rtx3080-wsl2-desktop-4p8.json", "rtx4080-16gb-headless.json", "rtx5090-32gb.json")


@needs_parser
@pytest.mark.parametrize("fixture", PRESET_FIXTURES)
def test_preset_fit_runs_for_every_preset_and_budget(fixture: str) -> None:
    presets = sorted(path.name for path in (ROOT / "src" / "cm1core" / "presets").iterdir()
                     if (path / "preset.json").is_file())
    assert len(presets) == 21  # the five first presets and the sixteen that followed
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
           "GPUWM_CM1_NVIDIA_SMI": str(GPUINFO / "fake_nvidia_smi.py"),
           "FAKE_NVIDIA_SMI_FIXTURE": str(GPUINFO / fixture),
           "GPUWM_CM1_HOST_FIXTURE": str(GPUINFO / "host-64gb-1tb.json")}
    fitted = 0
    for preset in presets:
        for budget in ("10m", "1h", "overnight"):
            done = subprocess.run([sys.executable, "-m", "cm1core", "preset", "fit", preset, "--budget", budget,
                                   "--json"], capture_output=True, text=True, env=env, timeout=300)
            assert "Traceback" not in done.stderr, done.stderr
            assert done.returncode in (0, 1), (preset, budget, done.returncode, done.stderr[-800:])
            answer = json.loads(done.stdout)
            if done.returncode == 0:
                assert answer["status"] == "fits"
                fitted += 1
            else:
                assert answer["status"] != "fits" and answer["reason"], (preset, budget, answer)
    assert fitted >= 3


def test_no_sentence_claims_cluster_core_equivalence() -> None:
    for path in RELEASE_DOCS:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            assert not re.search(r"\bcores\b", line, re.I), f"{path.name}:{number}: {line}"


def test_the_quickstart_passes_guided_modes_copy_lint() -> None:
    from cm1core.gui.copy_lint import lint_markdown

    assert lint_markdown(QUICKSTART) == []


def test_the_commands_page_lists_every_command_the_api_can_return() -> None:
    from cm1core.gui.commands import API

    page = COMMANDS.read_text(encoding="utf-8")
    commands_page = runpy.run_path(str(ROOT / "tools" / "release" / "commands_page.py"))
    assert page == commands_page["render"](), "docs/COMMANDS.md is behind: run tools/release/commands_page.py"
    for endpoint in API:
        assert f"`{endpoint.command.replace('|', chr(92) + '|')}`" in page, endpoint.command
    for sidecar in ("gpuwm-cm1 cm1out RUNDIR --follow", "gpuwm-cm1 quicklook RUNDIR --follow"):
        assert sidecar in page


def test_install_lines_name_this_version_and_the_release_files() -> None:
    """A pip or uv line installs gpuwm-cm1 by name (from PyPI) with no extra, or this version's wheel file; a line
    with the exact pins names this version, since the constraints file is this release's."""

    wheel = f"gpuwm_cm1-{__version__}-py3-none-manylinux_2_28_x86_64.whl"
    install_lines = []
    for path in RELEASE_DOCS:
        for kind, body, _line in DOC_COMMANDS["blocks"](path.read_text(encoding="utf-8")):
            if kind in ("sh install", "powershell install"):
                install_lines += [(path.name, line) for line in DOC_COMMANDS["lines"](body)]
    pip_lines = [(name, line) for name, line in install_lines
                 if re.search(r"\bpip install\b|\buv tool install\b", line) and re.search(r"gpuwm[-_]cm1", line)]
    assert {name for name, _line in pip_lines} >= {"README.md", "INSTALL.md"}, "no install line"
    for name, line in pip_lines:
        assert "[gpu" not in line, (name, line)  # the GPU stack is in the base requirements
        for pinned in re.findall(r"gpuwm-cm1==([^\s\"']+)", line):
            assert pinned == __version__, (name, line)
        if "gpuwm_cm1" in line:
            assert wheel in line, (name, line)
        if "-c constraints-gpu-cu13.txt" in line:
            assert f"gpuwm-cm1=={__version__}" in line or wheel in line, (name, line)
    assert ("INSTALL.md", f'pip install "gpuwm-cm1=={__version__}" -c constraints-gpu-cu13.txt') in pip_lines
    # The one-line installers, in the three guides a new user reads.
    raw = "https://raw.githubusercontent.com/FahrenheitResearch/gm1/main"
    for name in ("README.md", "INSTALL.md", "QUICKSTART.md"):
        assert (name, f"curl -LsSf {raw}/install.sh | sh") in install_lines, name
        assert (name, f"irm {raw}/install.ps1 | iex") in install_lines, name
    smoke = (ROOT / "tools" / "release" / "smoke_cm1_user.sh").read_text(encoding="utf-8")
    # The smoke installs as the guides say (the wheel file, the exact pins), plus the film extra QUICKSTART offers
    # (it makes an MP4 film).
    assert '"$WHEEL[film]" -c "$CONSTRAINTS"' in smoke
    assert 'pip install "gpuwm-cm1[film]"' in (ROOT / "docs" / "QUICKSTART.md").read_text(encoding="utf-8")


def test_the_doc_runner_reads_blocks_and_lines() -> None:
    text = "x\n```sh\ngpuwm-cm1 a \\\n  --b\n# note\n\ngpuwm-cm1 c\n```\n```text\nout\n```\n"
    found = list(DOC_COMMANDS["blocks"](text))
    assert [kind for kind, _body, _line in found] == ["sh", "text"]
    assert list(DOC_COMMANDS["lines"](found[0][1])) == ["gpuwm-cm1 a    --b", "gpuwm-cm1 c"]
    assert DOC_COMMANDS["gpuwm_words"]("gpuwm-cm1 preset fit x --budget 1h") == ["preset", "fit", "x", "--budget", "1h"]
    assert shutil.which("bash")
