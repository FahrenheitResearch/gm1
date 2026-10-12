"""The guides are doc-tested.

1. Every command block of docs/QUICKSTART.md, docs/CM1-USERS.md and docs/GM1-DECKS.md runs, in order, on CM1's
   stock supercell namelist (tools/release/doc_commands.py --mode fake: the real host commands, the GUI's fake runner for the GPU
   work, the WSL2 RTX 3080 fixture as nvidia-smi); and `gpuwm-cm1 preset fit` runs for every preset and budget on
   fake-GPU fixtures. The release is checked with the same guides in --mode real on the RTX 3080.
2. No sentence claims cluster-core equivalence.
3. docs/QUICKSTART.md passes guided mode's copy lint, and docs/COMMANDS.md lists every command the API table can
   return (it is generated from that table).
4. docs/GM1-DECKS.md: its GPU blocks name only options the commands have, its table has a row for every example
   deck, and the README's "Beyond the GUI" commands exist.
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
DECKS = ROOT / "docs" / "GM1-DECKS.md"
README = ROOT / "README.md"
RELEASE_DOCS = (QUICKSTART, CM1_USERS, COMMANDS, DECKS, ROOT / "docs" / "INSTALL.md",
                ROOT / "docs" / "RELEASE-NOTES.md", ROOT / "docs" / "TROUBLESHOOTING.md", README)
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
@pytest.mark.parametrize("guide", [QUICKSTART, CM1_USERS, DECKS], ids=lambda path: path.name)
def test_every_command_block_runs_in_order(guide: Path, tmp_path: Path) -> None:
    record = DOC_COMMANDS["run_guide"](guide, tmp_path / guide.stem, "fake", timeout_s=600)
    assert record["passed"], "\n".join(record["problems"])
    ran = [row for row in record["commands"] if "command" in row]
    assert len(ran) >= (6 if guide == QUICKSTART else 20)
    if guide == DECKS:
        # The GPU blocks are not run here; every one of their lines was checked against the commands' --help.
        checked = [row for row in record["commands"] if "checked" in row]
        assert len(checked) >= 6 and not any(row["unknown_options"] for row in checked)
        assert any("--integrate" in row["checked"] for row in checked)
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


def test_a_gpu_block_is_checked_against_the_commands_help_without_a_gpu(tmp_path: Path) -> None:
    guide = tmp_path / "g.md"
    guide.write_text("```sh gpu\ngpuwm-cm1 cartesian d.toml --integrate --output r.json --history-dir f\n"
                     "gpuwm-cm1 cartesian d.toml --integrate --output r.json --no-such-option\n"
                     "gpuwm-cm1 nosuchcommand x\ntouch control/stop\n```\n", encoding="utf-8")
    record = DOC_COMMANDS["run_guide"](guide, tmp_path / "work", "fake", timeout_s=120)
    rows = [row for row in record["commands"] if "checked" in row]
    assert [row["unknown_options"] for row in rows] == [[], ["--no-such-option"], ["no command `gpuwm-cm1 nosuchcommand`"], []]
    assert len(record["problems"]) == 2 and not record["passed"]
    assert not (tmp_path / "work" / "r.json").exists()  # nothing was run


def test_the_decks_page_has_a_row_for_every_example_deck() -> None:
    text = DECKS.read_text(encoding="utf-8")
    rows = set(re.findall(r"^\| `([a-z0-9-]+)` \|", text, re.M))
    decks = {path.stem for path in (ROOT / "examples").glob("*.toml")}
    assert rows == decks, (sorted(decks - rows), sorted(rows - decks))
    # The page says how many `gpuwm-cm1 examples` lists and how many the repository holds.
    unshipped = DOC_COMMANDS["unshipped_examples"]()
    assert f"`gpuwm-cm1 examples` lists {len(decks) - len(unshipped)} decks" in text
    assert f"all {len(decks)} example decks" in (ROOT / "docs" / "RELEASE-NOTES.md").read_text(encoding="utf-8")
    # The decks the page's own commands use are shipped ones.
    for name in re.findall(r"decks/([a-z0-9-]+)\.toml", text):
        assert name in decks, name


def test_the_readme_names_the_deck_door_and_only_commands_that_exist() -> None:
    from cm1core.cli import build_parser

    text = README.read_text(encoding="utf-8")
    for needle in ("## What GM1 has done", "## Beyond the GUI", "docs/GM1-DECKS.md", "gpuwm-cm1 examples",
                   "gpuwm-cm1 cartesian", "gpuwm-cm1 chain", "gpuwm-cm1 reproduce", "gpuwm-cm1 doctor --report"):
        assert needle in text, needle
    help_text = build_parser().format_help()
    for command in set(re.findall(r"`gpuwm-cm1 ([a-z0-9]+)", text + DECKS.read_text(encoding="utf-8"))):
        assert re.search(rf"^\s+{command}\s", help_text, re.M), command


def test_the_doc_runner_reads_blocks_and_lines() -> None:
    text = "x\n```sh\ngpuwm-cm1 a \\\n  --b\n# note\n\ngpuwm-cm1 c\n```\n```text\nout\n```\n"
    found = list(DOC_COMMANDS["blocks"](text))
    assert [kind for kind, _body, _line in found] == ["sh", "text"]
    assert list(DOC_COMMANDS["lines"](found[0][1])) == ["gpuwm-cm1 a    --b", "gpuwm-cm1 c"]
    assert DOC_COMMANDS["gpuwm_words"]("gpuwm-cm1 preset fit x --budget 1h") == ["preset", "fit", "x", "--budget", "1h"]
    assert shutil.which("bash")
