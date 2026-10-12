"""docs/NESTING.md and examples/tornado-4m are held to the command line and to each other.

1. Every command the two pages show is one the CLI or a published tool accepts: the `gpuwm-cm1` lines parse with
   the real parser, every option of a `python3 tools/...` line is an option of that tool, and every repository path
   a command names exists. Blocks carry the marks tools/release/doc_commands.py reads.
2. The lines a reader can run without a GPU (the `sh` blocks: check, plan, the plan summary, the sounding report,
   the estimate) run, in order, from the repository root, and the page's sample output is what they print.
3. Every deck in examples/tornado-4m loads, the stages of one variant differ only where their headers say, and
   every `[nest]` and window key a shipped deck uses is explained on the page.
4. The folder stays out of the wheel and the sdist, and the tools the pages name are in the public snapshot.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import sys
import tomllib

import pytest

from cm1core.cli import build_parser
from cm1core.config import load_config

ROOT = Path(__file__).resolve().parents[1]
NESTING = ROOT / "docs" / "NESTING.md"
EXAMPLE = ROOT / "examples" / "tornado-4m"
README = EXAMPLE / "README.md"
PAGES = (NESTING, README)
DOC_COMMANDS = runpy.run_path(str(ROOT / "tools" / "release" / "doc_commands.py"))
SHIPPED_SOUNDING = "src/cm1core/data/soundings/koun-2011-05-24-1800z.input_sounding"
VARIANTS = ("", "norman/", "small/")
DECKS = tuple(f"{variant}{stage}.toml" for variant in VARIANTS for stage in ("S1", "P", "S3"))
#: Programs a command line may start with, besides gpuwm-cm1 and python3 tools/NAME.py.
OTHER_PROGRAMS = {"mkdir", "cp", "git", "cd"}
#: What tools/plan_summary.py says the card holds above the plan's listed bytes.
OVERHEAD_GIB = 4.0


def _commands(page: Path, kinds: tuple[str, ...]) -> list[tuple[int, str]]:
    found = []
    for kind, body, line in DOC_COMMANDS["blocks"](page.read_text(encoding="utf-8")):
        if kind in kinds:
            found += [(line, command) for command in DOC_COMMANDS["lines"](body)]
    return found


def _words(command: str) -> list[str]:
    """The command's words up to an output redirection."""

    words = shlex.split(command, comments=True)
    return words[: words.index(">")] if ">" in words else words


def _native_tools_available() -> bool:
    try:
        from cm1core.cartesian_door import resolve_initializer_library

        resolve_initializer_library(None)
    except Exception:  # noqa: BLE001 - any failure to find the library means the plan cannot run here
        return False
    return True


needs_native = pytest.mark.skipif(not _native_tools_available(),
                                  reason="the cm1-state initializer library is not built (the plan reads a sounding with it)")


def test_the_pages_exist_and_mark_every_block() -> None:
    for page in PAGES:
        kinds = {kind for kind, _body, _line in DOC_COMMANDS["blocks"](page.read_text(encoding="utf-8"))}
        assert kinds <= set(DOC_COMMANDS["KINDS"]) | {"toml"}, (page.name, kinds)
    assert len(_commands(NESTING, ("sh", "sh example"))) >= 20
    assert len(_commands(NESTING, ("sh",))) >= 6


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.parent.name + "/" + path.name)
def test_every_command_is_one_the_cli_or_a_published_tool_accepts(page: Path) -> None:
    parser = build_parser()
    stage_tool = ROOT / "tools" / "release" / "stage_snapshot.py"
    allowed = runpy.run_path(str(stage_tool))["allowed"] if stage_tool.is_file() else None
    seen_tools = set()
    for line, command in _commands(page, ("sh", "sh example")):
        words = _words(command)
        where = f"{page.name}:{line}: {command}"
        assert words, where
        if words[0] == "gpuwm-cm1":
            try:
                parser.parse_args(words[1:])
            except SystemExit as error:  # argparse refused the line
                raise AssertionError(f"{where}: the CLI refuses it ({error.code})") from None
        elif words[0] == "python3":
            assert len(words) > 1 and re.fullmatch(r"tools/[a-z_]+\.py", words[1]), where
            tool = ROOT / words[1]
            assert tool.is_file(), where
            if allowed is not None:
                assert allowed(words[1]), f"{where}: {words[1]} is not in the public snapshot's allowlist"
            source = tool.read_text(encoding="utf-8")
            for word in words[2:]:
                if word.startswith("--"):
                    assert f'"{word}"' in source, f"{where}: {words[1]} has no option {word}"
            seen_tools.add(words[1])
        else:
            assert words[0] in OTHER_PROGRAMS, where
        for word in words[1:]:
            if word.startswith(("examples/", "src/", "tools/", "docs/")):
                assert (ROOT / word).exists(), f"{where}: {word} is not in the repository"
    if page == NESTING:
        assert seen_tools >= {"tools/nest_place.py", "tools/plan_summary.py", "tools/nest_window_preview.py",
                              "tools/nest_window_trajectories.py", "tools/tornado_criterion.py",
                              "tools/check_nest_following.py", "tools/report_sounding_parameters.py"}


def test_every_tool_and_file_the_pages_name_in_prose_exists() -> None:
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        for name in set(re.findall(r"`((?:tools|examples|docs|src)/[A-Za-z0-9_./-]+)`", text)):
            assert (ROOT / name).exists(), f"{page.name} names {name}, which is not in the repository"
        for target in set(re.findall(r"\]\((?!https?:)([^)#]+)\)", text)):
            assert (page.parent / target).resolve().exists(), f"{page.name} links to {target}"


@pytest.mark.parametrize("deck", DECKS)
def test_every_deck_loads(deck: str) -> None:
    config = load_config(EXAMPLE / deck)
    assert (config.nest is not None) == deck.endswith("S3.toml")


def _table(deck: str) -> dict:
    return tomllib.loads((EXAMPLE / deck).read_text(encoding="utf-8"))


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_stages_of_one_variant_differ_only_where_their_headers_say(variant: str) -> None:
    first, second, third = (_table(f"{variant}{stage}.toml") for stage in ("S1", "P", "S3"))
    for table in ("initialization", "grid", "dynamics", "boundaries", "closures", "physics", "diagnostics"):
        assert first[table] == second[table] == third[table], (variant, table)
    assert first["runtime"] == second["runtime"]
    assert {key for key in first["runtime"] if first["runtime"][key] != third["runtime"][key]} == {"memory_budget_mib"}
    assert "nest" not in first and "nest" not in second and "nest" in third
    assert (first["run"]["irst"], second["run"]["irst"], third["run"]["irst"]) == (0, 1, 1)
    window = third["lagrangian"]["nest_child_window"]
    assert window["end_s"] <= third["run"]["duration_s"]


def test_the_shipped_sounding_variant_changes_only_the_frame_and_the_bubble() -> None:
    for stage in ("S1", "P", "S3"):
        original, norman = _table(f"{stage}.toml"), _table(f"norman/{stage}.toml")
        changed = {(table, key) for table in original for key in original[table]
                   if not isinstance(original[table][key], dict) and original[table][key] != norman[table][key]}
        assert changed == {("closures", "u_move_m_s"), ("closures", "v_move_m_s"),
                           ("initialization", "bubble_center_x_m"), ("initialization", "bubble_center_y_m")}, stage
        if stage == "S3":
            assert original["nest"] == norman["nest"]
            assert original["lagrangian"] == norman["lagrangian"]


def test_the_four_metre_deck_ends_inside_the_clean_span_and_writes_a_half_second_window() -> None:
    deck = _table("S3.toml")
    assert deck["run"]["duration_s"] <= 3300.0
    nest, window = deck["nest"], deck["lagrangian"]["nest_child_window"]
    child_dt = deck["run"]["dt_s"] / nest["time_refinement_ratio"]
    assert window["stride_child_steps"] * child_dt == 0.5
    assert window["end_s"] == deck["run"]["duration_s"]
    # 227 x 227 x 45 cells: the record size the plan's window door reports.
    per_minute = 137_575_536 * 60.0 / 0.5
    assert per_minute <= window["budget_bytes_per_model_minute"] <= 1.1 * per_minute
    assert (nest["parent_i0"], nest["parent_j0"], deck["run"]["rstnum"]) == (295, 304, 8)


def test_every_nest_and_window_key_a_deck_uses_is_explained() -> None:
    from cm1core import config as config_module
    from cm1core.lagrangian_config import NEST_CHILD_WINDOW_KEYS

    page = NESTING.read_text(encoding="utf-8")
    used_nest, used_window = set(), set()
    for deck in DECKS:
        table = _table(deck)
        used_nest |= set(table.get("nest", {}))
        used_window |= set(table.get("lagrangian", {}).get("nest_child_window", {}))
    assert used_nest <= set(config_module._NEST_KEYS) and used_window <= set(NEST_CHILD_WINDOW_KEYS)
    for key in sorted(used_nest | used_window | set(config_module._NEST_KEYS) | set(NEST_CHILD_WINDOW_KEYS)):
        shown = "[nest.tornado_criterion]" if key == "tornado_criterion" else f"`{key}`"
        assert shown in page, f"docs/NESTING.md does not explain {key}"
    for key in config_module.NEST_TORNADO_CRITERION_KEYS:
        assert f"`{key}`" in page, key
    # The page's own [nest] example is the small deck's table.
    shown = [tomllib.loads(body) for kind, body, _line in DOC_COMMANDS["blocks"](page) if kind == "toml"]
    small = _table("small/S3.toml")["nest"]
    assert {key: value for key, value in small.items() if key != "tornado_criterion"} in [t.get("nest") for t in shown]


def test_the_folder_stays_out_of_the_wheel_and_the_sdist() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "examples/*.toml" in project["tool"]["setuptools"]["package-data"]["cm1core"]
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    assert "prune examples/tornado-4m" in manifest
    assert manifest.index("prune examples/tornado-4m") > manifest.index("recursive-include examples *.toml")
    # The original sounding is not shipped: only decks and the page are here.
    names = sorted(path.relative_to(EXAMPLE).as_posix() for path in EXAMPLE.rglob("*") if path.is_file())
    assert names == sorted(DECKS + ("README.md",))
    assert "eabada6c37b805bc05fd3be39fe2eadca048d86bdc7a3882971d33634d78de05" in README.read_text(encoding="utf-8")


def _environment(work: Path) -> dict[str, str]:
    env = {**os.environ, "HOME": str(work / "home"), **DOC_COMMANDS["fake_environment"](work)}
    (work / "home").mkdir(parents=True, exist_ok=True)
    python3 = work / ".doc-bin" / "python3"
    python3.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    python3.chmod(0o755)
    return env


def _plan(deck: str, env: dict[str, str], work: Path) -> dict:
    out = work / deck.replace("/", "-")
    done = subprocess.run(
        [sys.executable, "-m", "cm1core", "cartesian", str(EXAMPLE / deck), "--plan", "--integrate",
         "--sounding", str(ROOT / SHIPPED_SOUNDING), "--output", f"{out}.receipt.json",
         "--history-dir", f"{out}.frames", "--restart-dir", f"{out}.restarts"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-1500:]
    import json

    return json.loads(done.stdout)


@needs_native
def test_the_lines_that_need_no_gpu_run_and_print_what_the_page_shows(tmp_path: Path) -> None:
    env = _environment(tmp_path)
    printed: dict[str, str] = {}
    for page in PAGES:
        for line, command in _commands(page, ("sh",)):
            done = subprocess.run(["bash", "-c", command], cwd=ROOT, env=env, capture_output=True, text=True,
                                  timeout=900)
            output = done.stdout + done.stderr
            assert done.returncode == 0 and "Traceback (most recent call last)" not in output, (
                f"{page.name}:{line}: `{command}` exited {done.returncode}: {output[-1200:]}")
            printed[f"{page.name}: {command}"] = done.stdout
    summaries = [text for command, text in printed.items() if "tools/plan_summary.py" in command]
    assert len(summaries) == 1
    page = NESTING.read_text(encoding="utf-8")
    samples = [body for kind, body, _line in DOC_COMMANDS["blocks"](page) if kind == "text" and body.startswith("status ")]
    assert samples == [summaries[0]], "docs/NESTING.md's sample of the plan summary is not what the tool prints"
    report = next(text for command, text in printed.items() if "report_sounding_parameters.py" in command)
    assert "CAPE,CIN (J/kg)" in report and "0-1,0-3 km SRH (m2/s2)" in report
    estimate = next(text for command, text in printed.items() if " estimate " in command)
    assert "448 x 448 x 80" in estimate and "Nests are not priced" not in estimate


@needs_native
def test_the_readme_quotes_the_plans_of_the_decks_that_need_no_restart(tmp_path: Path) -> None:
    env = _environment(tmp_path)
    readme = README.read_text(encoding="utf-8")
    summary = runpy.run_path(str(ROOT / "tools" / "plan_summary.py"))
    assert summary["MEASURED_OVERHEAD_GIB"] == OVERHEAD_GIB
    for deck in ("norman/S1.toml", "small/S1.toml"):
        row = summary["summary"](_plan(deck, env, tmp_path))
        cells = [cell.strip() for cell in next(line for line in readme.splitlines()
                                               if line.startswith(f"| `{deck}`")).strip("|").split("|")]
        assert cells[1].startswith(" x ".join(str(n) for n in row["parent_grid"])), (deck, cells)
        assert cells[2] == f"{row['start_s']:g} to {row['end_s']:g} s", (deck, cells)
        assert cells[3:6] == [f"{row['parent_steps']:,}", str(row["frames"]), str(row["restarts"])], (deck, cells)
        assert cells[6] == f"{row['listed_gib']:.2f} GiB", (deck, cells)
        assert row["status"] == "priced-no-device-opened" and row["storm_track"] == "tracked-member-stays-inside"
