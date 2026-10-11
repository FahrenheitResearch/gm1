"""Installing and self-diagnosis for people who are not developers.

1. `pip install gpuwm-cm1` alone is a working GPU install: the CUDA 13 stack is in the base requirements, in ranges
   that hold the validated versions, and the old extras are accepted empty aliases.
2. The installers: install.sh (POSIX sh; run here with stand-in uv, curl and nvidia-smi) and install.ps1 (safe to
   pipe into Invoke-Expression; parsed by PowerShell where it exists).
3. `gpuwm-cm1 gui` opens the browser where there is a screen, never over ssh or without a terminal.
4. `gpuwm-cm1 doctor`: one exact fix per failure (the Windows driver under WSL2, an old card, Python, a damaged
   install), the presets a short computer greys, and `--report`, one pasteable block that names no user or host.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
GPUINFO = ROOT / "tests" / "data" / "gpuinfo"
INSTALL_SH = ROOT / "install.sh"
INSTALL_PS1 = ROOT / "install.ps1"
RAW = "https://raw.githubusercontent.com/FahrenheitResearch/gm1/main"
SH_LINE = f"curl -LsSf {RAW}/install.sh | sh"
PS_LINE = f"irm {RAW}/install.ps1 | iex"
#: A Linux home folder, spelled in parts (a public test names no home path).
HOMES = "/".join(("", "home", ""))


def _parser_available() -> bool:
    try:
        from cm1core.cm1import import namelist

        namelist.locate_tool()
    except Exception:  # noqa: BLE001
        return False
    return True


needs_parser = pytest.mark.skipif(not _parser_available(), reason="the cm1-r21-namelist tool is not built")


# ---------------------------------------------------------------- 1. the plain install

def _requirement(text: str) -> tuple[str, frozenset[str]]:
    match = re.match(r"^([A-Za-z0-9_.-]+(?:\[[^\]]*\])?)(.*)$", text.replace(" ", ""))
    assert match, text
    return match.group(1).lower(), frozenset(part for part in match.group(2).split(",") if part)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text))


def _admits(specifiers: frozenset[str], version: str) -> bool:
    for specifier in specifiers:
        operator, bound = re.match(r"(>=|<=|==|<|>)(.+)", specifier).groups()
        have, limit = _version_tuple(version), _version_tuple(bound)
        width = max(len(have), len(limit))
        have, limit = have + (0,) * (width - len(have)), limit + (0,) * (width - len(limit))
        if not {">=": have >= limit, "<=": have <= limit, "==": have == limit, "<": have < limit,
                ">": have > limit}[operator]:
            return False
    return True


def test_a_plain_install_brings_the_cuda_13_stack_in_ranges_that_hold_the_validated_versions() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    found = {_requirement(text) for text in project["dependencies"]}
    import runpy

    inspector = runpy.run_path(str(ROOT / "tools" / "inspect_artifacts.py"))
    assert found == set(inspector["EXPECTED_BASE_REQUIREMENTS"])
    names = {name for name, _specs in found}
    assert {"cupy-cuda13x[ctk]", "nvidia-cuda-nvrtc", "nvidia-nvjitlink", "gpuwm", "gpuwm-data"} == names
    # Every range is bounded above, and holds the version every GPU check ran with.
    validated = {}
    for line in (ROOT / "constraints" / "gpu-cu13-validated.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==")
            validated[name.strip()] = version.strip()
    for name, specifiers in found:
        assert any(spec.startswith("<") for spec in specifiers), name
        assert _admits(specifiers, validated[name.split("[")[0]]), (name, specifiers)
    # The old install lines keep working and add nothing.
    extras = project["optional-dependencies"]
    assert extras["gpu-cu13"] == [] and extras["gpu"] == []
    # The doctor's own fix lines name the same ranges.
    from cm1core.cm1run import system

    assert {_requirement(text) for text in system.GPU_REQUIREMENTS} == {
        (name, specs) for name, specs in found if name.startswith(("cupy", "nvidia"))}
    assert {_requirement(text) for text in system.ARWEN_REQUIREMENTS} == {
        (name, specs) for name, specs in found if name.startswith("gpuwm")}


def test_the_inspector_checks_the_base_requirements_and_the_alias_extras() -> None:
    import runpy

    inspector = runpy.run_path(str(ROOT / "tools" / "inspect_artifacts.py"))
    key = inspector["requirement_key"]
    assert key("cupy-cuda13x[ctk]<15,>=14.2") == ("cupy-cuda13x[ctk]", frozenset({">=14.2", "<15"}))
    assert key("gpuwm-data (<2.8.1,>=2.8.0)") == ("gpuwm-data", frozenset({">=2.8.0", "<2.8.1"}))
    source = (ROOT / "tools" / "inspect_artifacts.py").read_text(encoding="utf-8")
    assert '"wheel-gpu-stack-in-base-dependencies"' in source
    assert '"wheel-gpu-extras-are-accepted-aliases"' in source


# ---------------------------------------------------------------- 2. install.sh

FAKE_UV = r"""#!/bin/sh
echo "uv $*" >> "$FAKE_LOG"
case "$1 $2" in
  "--version "*) echo "uv ${FAKE_UV_VERSION:-0.13.0}"; exit 0 ;;
  "tool --help") exit 0 ;;
  "tool dir") echo "$HOME/.local/bin"; exit 0 ;;
  "tool list") [ -x "$HOME/.local/bin/gpuwm-cm1" ] && echo "gpuwm-cm1 v0.1.0"; exit 0 ;;
  "tool uninstall") rm -f "$HOME/.local/bin/gpuwm-cm1"; exit 0 ;;
  "cache clean") exit 0 ;;
  "tool install")
    [ "${FAKE_UV_FAIL:-0}" = 1 ] && exit 1
    for word in "$@"; do
      case "$prev" in --constraints) cp "$word" "$FAKE_LOG.constraints" ;; esac
      prev=$word
    done
    mkdir -p "$HOME/.local/bin"
    printf '#!/bin/sh\necho "gpuwm-cm1 $*" >> "%s"\n' "$FAKE_LOG" > "$HOME/.local/bin/gpuwm-cm1"
    chmod +x "$HOME/.local/bin/gpuwm-cm1"
    exit 0 ;;
esac
exit 0
"""

FAKE_CURL = r"""#!/bin/sh
echo "curl $*" >> "$FAKE_LOG"
url="" out=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out=$2; shift 2 ;;
    --retry|--connect-timeout) shift 2 ;;
    -*) shift ;;
    *) url=$1; shift ;;
  esac
done
case "$url" in
  *pypi.org*) body='{"info":{"author":"","name":"gpuwm-cm1","summary":"a, b","version":"0.1.0","yanked":false},"urls":[{"python_version":"py3"}]}' ;;
  *gpu-cu13-validated.txt) body='gpuwm==2.8.0
cupy-cuda13x==14.2.0' ;;
  *) exit 22 ;;
esac
[ "${FAKE_OFFLINE:-0}" = 1 ] && exit 6
printf '%s\n' "$body" > "$out"
"""

FAKE_SMI = """#!/bin/sh
echo "${FAKE_DRIVER:-581.42}, NVIDIA GeForce RTX 3080"
"""

FAKE_UNAME = """#!/bin/sh
case "$1" in
  -s) echo Linux ;;
  -m) echo "${FAKE_ARCH:-x86_64}" ;;
  -r) echo "6.8.0-generic" ;;
  *) echo Linux ;;
esac
"""

needs_posix = pytest.mark.skipif(os.name == "nt" or shutil.which("sh") is None, reason="needs a POSIX sh")


def _tool(folder: Path, name: str, text: str) -> None:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def sandbox(tmp_path: Path):
    """A home folder and a PATH with stand-ins for uv, curl, nvidia-smi and uname (no network, no GPU)."""

    home, bin_dir = tmp_path / "home", tmp_path / "fakebin"
    home.mkdir()
    bin_dir.mkdir()
    # An empty process table: the computer running the tests may run GM1 itself.
    (tmp_path / "proc").mkdir()
    (home / ".bashrc").write_text("# the user's own\n", encoding="utf-8")
    _tool(bin_dir, "uv", FAKE_UV)
    _tool(bin_dir, "curl", FAKE_CURL)
    _tool(bin_dir, "nvidia-smi", FAKE_SMI)
    _tool(bin_dir, "uname", FAKE_UNAME)
    log = tmp_path / "calls.log"
    env = {"HOME": str(home), "PATH": f"{bin_dir}:/usr/bin:/bin", "FAKE_LOG": str(log), "LANG": "C",
           "GM1_PROC_ROOT": str(tmp_path / "proc")}

    def run(*args: str, cwd: Path | None = None, **extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["sh", str(INSTALL_SH), *args], cwd=cwd or tmp_path, env={**env, **extra},
                              capture_output=True, text=True, timeout=120)

    def calls() -> list[str]:
        return log.read_text(encoding="utf-8").splitlines() if log.is_file() else []

    return run, calls, home, tmp_path


@needs_posix
def test_install_sh_is_posix_sh_and_names_its_one_liners() -> None:
    assert subprocess.run(["sh", "-n", str(INSTALL_SH)], capture_output=True).returncode == 0
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n") and "\r" not in text and text.isascii()
    assert SH_LINE in text and f"{SH_LINE} -s -- --uninstall" in text and PS_LINE in text
    assert "main \"$@\"\n" in text.splitlines(keepends=True)[-1]  # a cut-off download runs nothing
    if shutil.which("shellcheck"):
        done = subprocess.run(["shellcheck", "-s", "sh", str(INSTALL_SH)], capture_output=True, text=True)
        assert done.returncode == 0, done.stdout


@needs_posix
def test_install_sh_installs_the_newest_release_with_its_exact_versions_and_reruns_cleanly(sandbox) -> None:
    run, calls, home, _tmp = sandbox
    done = run()
    assert done.returncode == 0, done.stdout + done.stderr
    install = [line for line in calls() if line.startswith("uv tool install")]
    assert len(install) == 1
    words = install[0].split()
    # uv's own Python, never the system's (an apt autoremove or an Ubuntu upgrade must not break GM1).
    assert words[3:11] == ["--python", "3.12", "--python-preference", "only-managed", "--force", "--upgrade",
                           "--reinstall-package", "gpuwm-cm1"]
    assert words[-1] == "gpuwm-cm1==0.1.0" and "--constraints" in words
    # The film add-on, so Save as film makes MP4s for installer users too.
    assert words[-3:-1] == ["--with", "imageio-ffmpeg>=0.5"]
    assert "With the exact versions GM1 was tested with." in done.stdout
    assert "cupy-cuda13x==14.2.0" in (_tmp / "calls.log.constraints").read_text(encoding="utf-8")
    assert any(line == "gpuwm-cm1 doctor --plain" for line in calls())
    assert "gpuwm-cm1 gui" in done.stdout and "gpuwm-cm1 doctor --report" in done.stdout
    assert "NVIDIA driver 581.42" in done.stdout
    # Run again: it updates, and the PATH lines are written once.
    assert run().returncode == 0
    for name in (".profile", ".bashrc"):
        text = (home / name).read_text(encoding="utf-8")
        assert text.count("# Added by the GM1 installer") == 1, name
        assert 'export PATH="$HOME/.local/bin:$PATH"' in text
    assert not (home / ".zshrc").exists()
    assert len([line for line in calls() if line.startswith("uv tool install")]) == 2


@needs_posix
def test_install_sh_takes_a_version_a_wheel_or_a_wheel_folder(sandbox) -> None:
    run, calls, _home, tmp = sandbox
    assert run(GM1_VERSION="0.1.0", GM1_SKIP_DOCTOR="1").returncode == 0
    assert [line for line in calls() if line.startswith("uv tool install")][-1].endswith(" gpuwm-cm1==0.1.0")
    wheels = tmp / "wheels"
    wheels.mkdir()
    (wheels / "gpuwm_cm1-0.1.0-py3-none-manylinux_2_28_x86_64.whl").write_bytes(b"PK")
    done = run(cwd=wheels, GM1_WHEEL="gpuwm_cm1-0.1.0-py3-none-manylinux_2_28_x86_64.whl", GM1_SKIP_DOCTOR="1")
    assert done.returncode == 0, done.stderr
    install = [line for line in calls() if line.startswith("uv tool install")][-1]
    assert install.endswith(f" {wheels}/gpuwm_cm1-0.1.0-py3-none-manylinux_2_28_x86_64.whl")
    assert "--constraints" in install  # the wheel's own release, by its name
    before = len(calls())
    done = run(GM1_FIND_LINKS="wheels", GM1_SKIP_DOCTOR="1", GM1_REINSTALL="1")
    assert done.returncode == 0
    new = calls()[before:]
    install = [line for line in new if line.startswith("uv tool install")][-1]
    assert f"--find-links {tmp}/wheels" in install and install.endswith(" gpuwm-cm1") and " --reinstall " in install
    assert not any("pypi.org" in line for line in new)
    assert run(GM1_WHEEL="missing.whl").returncode == 1


@needs_posix
def test_install_sh_warns_with_the_fix_and_still_installs_without_a_driver_or_offline(sandbox) -> None:
    run, calls, _home, _tmp = sandbox
    done = run(GM1_SKIP_DOCTOR="1", GM1_NVIDIA_SMI="none")
    assert done.returncode == 0
    assert "No NVIDIA driver found" in done.stdout or "No NVIDIA graphics card is visible inside WSL" in done.stdout
    assert "https://www.nvidia.com/Download/index.aspx" in done.stdout
    done = run(GM1_SKIP_DOCTOR="1", FAKE_DRIVER="535.104.05")
    assert done.returncode == 0 and "R580 or newer" in done.stdout
    # Offline PyPI and GitHub: GM1's own ranges, no constraints file.
    done = run(GM1_SKIP_DOCTOR="1", FAKE_OFFLINE="1")
    assert done.returncode == 0
    install = [line for line in calls() if line.startswith("uv tool install")][-1]
    assert "--constraints" not in install and install.endswith(" gpuwm-cm1")
    # ... and says so, so a pasted installer log shows it.
    assert "installing with GM1's own version ranges" in done.stdout
    assert "curl: (" not in done.stdout + done.stderr


@needs_posix
def test_install_sh_replaces_a_uv_older_than_the_tested_one(sandbox) -> None:
    run, calls, _home, _tmp = sandbox
    # The tested uv (0.13.0) is fetched from its own versioned installer; the fake curl has no such page, so the
    # install stops there with the reason.
    done = run(GM1_SKIP_DOCTOR="1", FAKE_UV_VERSION="0.4.30")
    assert done.returncode == 1
    assert "older than the one GM1 is tested with (0.13.0)" in done.stdout
    assert any("https://astral.sh/uv/0.13.0/install.sh" in line for line in calls() if line.startswith("curl"))
    assert not [line for line in calls() if line.startswith("uv tool install")]
    # A newer uv is used as it is.
    assert run(GM1_SKIP_DOCTOR="1", FAKE_UV_VERSION="0.14.2").returncode == 0


@needs_posix
def test_install_sh_will_not_replace_gm1_under_a_running_storm_or_window(sandbox) -> None:
    run, calls, _home, tmp = sandbox
    for words in (["/opt/py/bin/python", "/opt/x/.local/bin/gpuwm-cm1", "gui"],
                  ["python3", "-m", "cm1core", "run", "/runs/storm"],
                  ["python3", "-m", "cm1core", "cartesian", "deck.toml", "--output", "x"]):
        proc = tmp / "proc"
        shutil.rmtree(proc)
        (proc / "4242").mkdir(parents=True)
        (proc / "4242" / "cmdline").write_bytes(b"\0".join(word.encode() for word in words) + b"\0")
        done = run(GM1_SKIP_DOCTOR="1")
        assert done.returncode == 1, words
        assert "a storm (or the GM1 window) is running" in done.stderr and "paste the line again" in done.stderr
        assert run("--uninstall").returncode == 1
    assert not [line for line in calls() if line.startswith(("uv tool install", "uv tool uninstall"))]
    # Other programs, and GM1's other commands, do not stop it.
    (proc / "4242" / "cmdline").write_bytes(b"gpuwm-cm1\0preset\0new\0x\0--name\0run\0")
    (proc / "77").mkdir()
    (proc / "77" / "cmdline").write_bytes(b"vim\0notes.txt\0")
    assert run(GM1_SKIP_DOCTOR="1").returncode == 0


@needs_posix
def test_install_sh_from_windows_leaves_the_closing_words_to_the_windows_installer(sandbox) -> None:
    run, _calls, _home, _tmp = sandbox
    done = run(GM1_SKIP_DOCTOR="1", GM1_FROM_WINDOWS="1")
    assert done.returncode == 0
    assert "GM1 is installed" not in done.stdout and "export PATH" not in done.stdout
    assert "gpuwm-cm1 gui" not in done.stdout


@needs_posix
def test_install_sh_refuses_what_cannot_run_and_says_why(sandbox) -> None:
    run, _calls, _home, _tmp = sandbox
    done = run(FAKE_ARCH="aarch64")
    assert done.returncode == 1 and "x86-64" in done.stderr
    done = run(GM1_SKIP_DOCTOR="1", FAKE_UV_FAIL="1")
    assert done.returncode == 1 and "Rerun the same command" in done.stderr


@needs_posix
def test_install_sh_uninstalls_and_keeps_the_runs(sandbox) -> None:
    run, calls, home, _tmp = sandbox
    assert run(GM1_SKIP_DOCTOR="1").returncode == 0
    (home / "gm1-runs" / "storm").mkdir(parents=True)
    done = run("--uninstall")
    assert done.returncode == 0 and "uv tool uninstall gpuwm-cm1" in calls()
    assert "GM1 is uninstalled" in done.stdout and (home / "gm1-runs" / "storm").is_dir()
    assert not (home / ".local" / "bin" / "gpuwm-cm1").exists()
    # GM1's downloaded packages go too, and only GM1's: no bare "uv cache clean", which would empty every cache.
    clean = [line for line in calls() if line.startswith("uv cache clean")]
    assert len(clean) == 1 and clean[0] != "uv cache clean"
    for package in ("gpuwm-cm1", "gpuwm", "cupy-cuda13x", "nvidia-cuda-nvrtc", "nvidia-nvjitlink", "imageio-ffmpeg"):
        assert f" {package} " in f"{clean[0]} ", package
    assert "Freed the downloaded packages" in done.stdout and "uv cache clean" not in done.stdout
    # A second uninstall has nothing to free.
    again = run("--uninstall")
    assert again.returncode == 0 and "not installed" in again.stdout and "Freed" not in again.stdout


# ---------------------------------------------------------------- 2. install.ps1

def test_install_ps1_is_safe_to_pipe_into_invoke_expression() -> None:
    text = INSTALL_PS1.read_text(encoding="utf-8")
    assert text.isascii()
    # No exit: under irm | iex it would close the person's PowerShell window. The one exit is the shell's,
    # inside the command install.ps1 runs in WSL.
    assert not re.search(r"(?im)^\s*exit\b", text)
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    # The other "exit" is a word Ubuntu's first start asks the person to type, inside a sentence.
    assert re.findall(r"\bexit\b(?!  and press Enter)", code) == ["exit"] and "; exit $rc'" in code
    # Everything runs inside one function, called once at the end.
    assert text.rstrip().splitlines()[-1].strip() == "}"
    assert "\nfunction Invoke-GM1Installer {" in text and "Invoke-GM1Installer -Uninstall" in text
    # It does what the guides say.
    assert PS_LINE in text and "-Uninstall" in text and "[switch]$Uninstall" in text
    assert "'--install', '-d', 'Ubuntu'" in text and "-Verb RunAs" in text
    assert "--exec bash -lc" in text and "gpuwm-cm1 gui" in text and "GM1.lnk" in text
    assert "$MinDriver = 580" in text and "Never install an NVIDIA driver inside WSL" in text
    assert "GM1_INSTALL_SH" in text and "WSLENV" in text and "GM1_WHEEL/p" in text
    assert f"$RepoRaw = '{RAW}'" in text and '"$RepoRaw/install.sh"' in text
    # Settings reach install.sh, and the closing words say the Windows truth about the GM1 window.
    assert "'GM1_SKIP_DOCTOR'" in text and "'GM1_NO_MODIFY_PATH'" in text
    assert "Closing that window never stops a running storm" in text and "Windows stops a storm" in text
    assert "closing it on Windows stops a running storm" not in text and "keep its window open" not in text
    # A help report in one click, and GM1's own icon.
    assert "'GM1 check.lnk'" in text and "doctor --report" in text and "gm1-report.txt" in text
    assert "Set-Clipboard" in text and "Write-GM1Icon" in text
    # Uninstall: the shortcuts first, and the WSL side works when install.sh cannot be downloaded.
    uninstall = text[text.index("    if ($Uninstall) {"):text.index("    Step 'Checking Windows'")]
    assert uninstall.index("Remove-GM1Shortcuts") < uninstall.index("Get-InstallSh")
    assert "try { $script = Get-InstallSh } catch" in uninstall and "tool uninstall gpuwm-cm1" in uninstall
    assert "wsl --unregister" in uninstall
    # Ubuntu's first start happens in this window, then the install goes on.
    assert "Initialize-DistroUser $name" in text and "& wsl.exe -d $Name" in text
    # The literal way to set a setting in PowerShell.
    assert f"$env:GM1_DISTRO = 'Ubuntu-24.04'; irm {RAW}/install.ps1 | iex" in text


@pytest.mark.skipif(not (shutil.which("pwsh") or shutil.which("powershell")), reason="no PowerShell here")
def test_install_ps1_parses_in_powershell() -> None:
    shell = shutil.which("powershell") or shutil.which("pwsh")
    command = ("$e = $null; $t = $null; [void][System.Management.Automation.Language.Parser]::ParseFile("
               f"'{INSTALL_PS1}', [ref]$t, [ref]$e); $e.Count")
    done = subprocess.run([shell, "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=120)
    assert done.stdout.strip().splitlines()[-1] == "0", done.stdout + done.stderr


# ---------------------------------------------------------------- 3. the browser

class _Popen:
    def __init__(self, fail: tuple[str, ...] = ()) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self.fail = fail

    def __call__(self, words, **kwargs):  # noqa: ANN001 - subprocess.Popen's shape
        if Path(words[0]).name in self.fail:
            raise OSError(8, "Exec format error")
        self.calls.append((list(words), kwargs.get("cwd")))
        assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["start_new_session"]
        return object()


def test_the_browser_opens_only_where_there_is_a_screen() -> None:
    from cm1core.gui.browser import why_not

    desktop = {"DISPLAY": ":0"}
    assert why_not(desktop, wsl=False, terminal=True, platform="linux") is None
    assert why_not({"WAYLAND_DISPLAY": "wayland-0"}, wsl=False, terminal=True, platform="linux") is None
    assert why_not({}, wsl=False, terminal=True, platform="linux") == "there is no display"
    assert why_not({**desktop, "SSH_CONNECTION": "a b c d"}, wsl=False, terminal=True, platform="linux")
    assert why_not({"SSH_TTY": "/dev/pts/1"}, wsl=True, terminal=True, platform="linux")
    assert why_not(desktop, wsl=False, terminal=False, platform="linux") == "the output is not a terminal"
    assert why_not({**desktop, "GPUWM_CM1_NO_BROWSER": "1"}, wsl=False, terminal=True, platform="linux")
    assert why_not({**desktop, "GPUWM_CM1_NO_BROWSER": "0"}, wsl=False, terminal=True, platform="linux") is None
    # WSL2 has no DISPLAY of its own to need: the Windows browser opens.
    assert why_not({}, wsl=True, terminal=True, platform="linux") is None


def test_under_wsl_the_windows_browser_opens_through_wslview_explorer_or_cmd() -> None:
    from cm1core.gui.browser import open_link

    url = "http://127.0.0.1:8765/?token=abc"
    found = {"wslview": "/usr/bin/wslview", "explorer.exe": "/mnt/c/Windows/explorer.exe",
             "cmd.exe": "/mnt/c/Windows/System32/cmd.exe"}
    popen = _Popen()
    assert open_link(url, env={}, wsl=True, terminal=True, which=found.get, exists=lambda _p: True,
                     popen=popen) == (True, "wslview")
    assert popen.calls == [(["/usr/bin/wslview", url], None)]
    popen = _Popen(fail=("wslview",))
    assert open_link(url, env={}, wsl=True, terminal=True, which=found.get, exists=lambda _p: True,
                     popen=popen) == (True, "explorer.exe")
    popen = _Popen(fail=("wslview", "explorer.exe"))
    opened = open_link(url, env={}, wsl=True, terminal=True, which=found.get, exists=lambda _p: True, popen=popen)
    assert opened == (True, "cmd.exe")
    assert popen.calls == [(["/mnt/c/Windows/System32/cmd.exe", "/c", "start", "", url], "/mnt/c")]
    # Windows' programs off PATH: their usual place.
    popen = _Popen()
    assert open_link(url, env={}, wsl=True, terminal=True, which=lambda _name: None, exists=lambda _p: True,
                     popen=popen)[0]
    assert popen.calls[0][0] == ["/mnt/c/Windows/explorer.exe", url]
    # Nothing to open it with: the printed link stays, nothing raises.
    assert open_link(url, env={}, wsl=True, terminal=True, which=lambda _name: None, exists=lambda _p: False,
                     popen=_Popen()) == (False, "no Windows program to open it with was found")


def test_on_a_linux_desktop_python_webbrowser_opens_it_and_ssh_never_does() -> None:
    from cm1core.gui.browser import open_link

    opened: list[str] = []
    url = "http://127.0.0.1:8765/?token=abc"
    assert open_link(url, env={"DISPLAY": ":0"}, wsl=False, terminal=True, platform="linux",
                     webbrowser_open=lambda link: opened.append(link) or True) == (True, "webbrowser")
    assert opened == [url]
    assert open_link(url, env={"DISPLAY": ":0", "SSH_CONNECTION": "x"}, wsl=False, terminal=True, platform="linux",
                     webbrowser_open=lambda link: opened.append(link) or True)[0] is False
    assert open_link(url, env={"DISPLAY": ":0"}, wsl=False, terminal=True, platform="linux",
                     webbrowser_open=lambda _link: False) == (False, "no browser answered")
    assert len(opened) == 1


@pytest.mark.parametrize("no_browser", [False, True])
def test_gpuwm_cm1_gui_opens_the_link_unless_told_not_to(tmp_path, monkeypatch, capsys, no_browser) -> None:
    from cm1core.cli import main
    from cm1core.gui import browser, server

    seen: list[str] = []
    monkeypatch.setattr(browser, "open_link", lambda url: (seen.append(url) or True, "test"))

    def closed(self, poll_interval=0.5):  # noqa: ANN001 - serve_forever's shape
        raise KeyboardInterrupt

    monkeypatch.setattr(server.GuiServer, "serve_forever", closed)
    # The GUI refuses CuPy imports in its own process (a meta-path finder); keep that out of this test process.
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    # One case under WSL2, one on Linux: only Linux gets the ssh hint.
    monkeypatch.setenv("GPUWM_CM1_WSL2", "1" if no_browser else "0")
    args = ["gui", "--root", str(tmp_path / "runs"), "--port", "0", "--fake-runner", "--config-dir",
            str(tmp_path / "config")]
    assert main(args + (["--no-browser"] if no_browser else [])) == 0
    out = capsys.readouterr().out
    assert re.search(r"http://127\.0\.0\.1:\d+/\?token=\S+", out)  # the link is always printed
    assert ("ssh -L" in out) == (not no_browser)
    # Under WSL2 a run keeps Ubuntu up, so closing the window never ends a storm; only Windows going down does.
    assert ("Closing this window closes the GUI, never a storm" in out) == no_browser
    assert ("Shutting down or restarting Windows stops a storm" in out) == no_browser
    assert "closing it stops the storm" not in out and "Runs keep going" in out
    assert ("never a run" in out) == (not no_browser)
    if no_browser:
        assert seen == [] and "Opened the link" not in out
    else:
        assert len(seen) == 1 and seen[0] in out and "Opened the link in your browser (test;" in out


# ---------------------------------------------------------------- 4. doctor: one exact fix per failure

def _gpu(**fields):
    from cm1core.cm1run import gpuinfo

    record = {"index": 0, "uuid": "GPU-0", "name": "NVIDIA GeForce RTX 3080", "total_bytes": 10 * 1024**3,
              "free_bytes": 9 * 1024**3, "compute_capability": "8.6", "driver_version": "581.42", "wsl2": False}
    record.update(fields)
    return gpuinfo.GpuInfo.from_record(record)


def test_the_driver_fixes_say_where_the_driver_lives() -> None:
    from cm1core.cm1run import gpuinfo
    from cm1core.cm1run.system import NVIDIA_DOWNLOAD, driver_row

    linux = driver_row(gpuinfo.GpuQuery("no-nvidia-smi", message="none"))
    assert linux.state == "fail" and linux.fix.startswith("Install the NVIDIA driver R580 or newer")
    assert NVIDIA_DOWNLOAD in linux.fix
    wsl = driver_row(gpuinfo.GpuQuery("no-nvidia-smi", message="none", wsl2=True))
    assert wsl.state == "fail" and "inside WSL2" in wsl.text
    for words in ("on Windows", NVIDIA_DOWNLOAD, "wsl --update", "wsl --shutdown",
                  "Never install an NVIDIA driver inside WSL"):
        assert words in wsl.fix, words
    old = driver_row(gpuinfo.GpuQuery("ok", gpus=(_gpu(driver_version="566.36", wsl2=True),), wsl2=True))
    assert old.state == "fail" and "the Windows driver" in old.text and "older than R580" in old.text
    assert old.fix.startswith("Update the NVIDIA driver on Windows to R580") and "inside WSL" in old.fix
    old = driver_row(gpuinfo.GpuQuery("ok", gpus=(_gpu(driver_version="535.104.05"),)))
    assert old.fix.startswith("Update the NVIDIA driver to R580 or newer")
    empty = driver_row(gpuinfo.GpuQuery("ok", gpus=(), wsl2=True))
    assert empty.state == "fail" and "wsl --shutdown" in empty.fix
    broken = driver_row(gpuinfo.GpuQuery("error", message="nvidia-smi failed"))
    assert broken.state == "fail" and "Restart the computer" in broken.fix
    assert driver_row(gpuinfo.GpuQuery("ok", gpus=(_gpu(),))).state == "ok"


def test_a_card_older_than_turing_is_named_with_what_to_do() -> None:
    from cm1core.cm1run import gpuinfo
    from cm1core.cm1run.system import gpu_rows, too_old

    pascal = _gpu(name="NVIDIA GeForce GTX 1080", compute_capability="6.1")
    assert too_old(pascal) and not too_old(_gpu()) and not too_old(_gpu(compute_capability=None))
    assert not too_old(_gpu(compute_capability="7.5"))
    (row,) = gpu_rows(gpuinfo.GpuQuery("ok", gpus=(pascal,)))
    assert row.state == "fail" and "compute capability 7.5" in row.text and "RTX 20 series" in row.fix
    second = _gpu(index=1, uuid="GPU-1")
    rows = {row.id: row for row in gpu_rows(gpuinfo.GpuQuery("ok", gpus=(pascal, second)))}
    assert rows["gpu-0"].state == "note" and rows["gpu-0"].fix.endswith("--gpu 1")


def test_repairs_name_the_command_for_the_way_gm1_was_installed(monkeypatch) -> None:
    from cm1core.cm1run import system

    tool = {"kind": "uv-tool", "installer": "uv", "words": "uv tool (the installer)"}
    assert system.reinstall_fix("package", tool) == system.INSTALLER_LINE
    assert system.reinstall_fix("gpu", tool) == system.INSTALLER_REINSTALL_LINE
    assert "GM1_REINSTALL=1" in system.reinstall_fix("arwen", tool)
    venv = {"kind": "venv", "installer": "pip", "words": "pip in a virtual environment"}
    assert system.reinstall_fix("package", venv).endswith("-m pip install --force-reinstall --no-deps gpuwm-cm1")
    assert '"cupy-cuda13x[ctk]>=14.2,<15" "nvidia-cuda-nvrtc>=13.3,<13.4"' in system.reinstall_fix("gpu", venv)
    assert '"gpuwm>=2.8.0,<2.8.1"' in system.reinstall_fix("arwen", venv)
    uv_venv = {"kind": "venv", "installer": "uv", "words": "uv pip in a virtual environment"}
    assert system.reinstall_fix("package", uv_venv).startswith("uv pip install --python ")
    source = {"kind": "source", "installer": "pip", "words": "a source checkout"}
    assert system.reinstall_fix("package", source).startswith("cargo build")
    # This test runs from a checkout (or the snapshot): the method says so.
    method = system.install_method()
    assert set(method) == {"kind", "installer", "words"} and method["kind"] in ("source", "venv", "system", "uv-tool")
    # A missing CuPy, a damaged ArWen and a too-old Python each carry their command.
    monkeypatch.setattr(system, "_cupy_distribution", lambda: None)
    monkeypatch.setattr(system.importlib.util, "find_spec", lambda _name: None)
    cuda = system.cuda_row(venv)
    assert cuda.state == "fail" and cuda.fix == system.reinstall_fix("gpu", venv)
    monkeypatch.setattr(system.sys, "version_info", (3, 11, 9, "final", 0))
    python = system.python_row()
    assert python.state == "fail" and system.INSTALLER_LINE in python.fix


def test_the_command_line_greets_a_newcomer_and_says_its_version(capsys) -> None:
    from cm1core import __version__
    from cm1core.cli import main

    with pytest.raises(SystemExit) as stop:
        main(["--version"])
    assert stop.value.code == 0 and capsys.readouterr().out.strip() == f"gpuwm-cm1 {__version__}"
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "gpuwm-cm1 gui" in out and "gpuwm-cm1 doctor" in out and "error" not in out
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "port in progress" not in out and "New here? Type:          gpuwm-cm1 gui" in out
    assert out.index("    gui ") < out.index("    doctor ") < out.index("    version ")


def test_a_driver_whose_cuda_library_will_not_load_is_not_ready(monkeypatch) -> None:
    from cm1core.cm1run import gpuinfo, system

    query = gpuinfo.GpuQuery("ok", gpus=(_gpu(),))
    monkeypatch.setenv(system.LIBCUDA_ENV, "none")
    row = system.libcuda_row(query)
    assert row is not None and row.state == "fail" and "libcuda" in row.text
    assert row.fix.startswith("Install the NVIDIA driver R580") and "nvidia:580" in row.fix
    report = system.system_report(".", query=query, host=gpuinfo.HostInfo(
        disk_root="/x", disk_free_bytes=10**12, disk_total_bytes=2 * 10**12, ram_total_bytes=64 * 10**9,
        ram_available_bytes=60 * 10**9, wsl2=False), presets=False)
    assert report["exit"] == 1 and report["rows"][0]["id"] == "driver" and report["rows"][0]["state"] == "fail"
    # No driver at all is the driver row's own failure; a stand-in nvidia-smi alone (the tests') skips the load.
    assert system.libcuda_row(gpuinfo.GpuQuery("no-nvidia-smi")) is None
    monkeypatch.delenv(system.LIBCUDA_ENV)
    monkeypatch.setenv(gpuinfo.NVIDIA_SMI_ENV, "/fake/nvidia-smi")
    assert system.libcuda_loadable() is None and system.libcuda_row(query) is None


def test_a_cuda_no_driver_error_reads_in_plain_words() -> None:
    from cm1core.cm1run.events import NO_DEVICE_REASON, SegmentEvents, plain_record

    raw = {"kind": "refused", "code": "CM1-CARTESIAN-CUDA-DEVICE",
           "reason": "the CUDA runtime cannot enumerate a device: cudaErrorInsufficientDriver: CUDA driver version "
                     "is insufficient for CUDA runtime version",
           "remedy": "run on a host with a working NVIDIA driver and a visible device"}
    plain = plain_record(raw)
    assert plain["reason"] == NO_DEVICE_REASON and plain["remedy"] == "Run gpuwm-cm1 doctor --plain for the fix."
    assert "cudaErrorInsufficientDriver" in plain["technical"] and plain["code"] == raw["code"]
    end = plain_record({"kind": "end", "reason": "refused", "message": "cudaErrorNoDevice: no CUDA-capable device"})
    assert end["reason"] == "refused" and end["message"].startswith(NO_DEVICE_REASON)
    other = {"kind": "refused", "reason": "nz > 80 needs the capacity overlay", "remedy": "x"}
    assert plain_record(other) is other
    events = SegmentEvents().add([raw])
    assert events.refused["reason"] == NO_DEVICE_REASON


def test_short_ram_under_wsl2_says_how_to_give_it_more() -> None:
    from cm1core.cm1run import gpuinfo
    from cm1core.cm1run.system import disk_row, ram_row

    host = gpuinfo.HostInfo(disk_root="/x", disk_free_bytes=5 * 10**9, disk_total_bytes=100 * 10**9,
                            ram_total_bytes=8 * 10**9, ram_available_bytes=6 * 10**9, wsl2=True)
    ram = ram_row(host)
    assert ram.state == "note" and ".wslconfig" in ram.fix and "wsl --shutdown" in ram.fix
    # A line to paste into PowerShell (Notepad would save .wslconfig.txt), and no second [wsl2] line.
    assert 'Add-Content "$env:USERPROFILE' in ram.fix and "do not add a second one" in ram.fix
    disk = disk_row(host)
    assert disk.state == "note" and "gpuwm-cm1 gui --root FOLDER" in disk.fix
    full = gpuinfo.HostInfo(disk_root="/x", disk_free_bytes=10**8, disk_total_bytes=100 * 10**9,
                            ram_total_bytes=64 * 10**9, ram_available_bytes=60 * 10**9, wsl2=False)
    assert disk_row(full).state == "fail" and disk_row(full).fix


@needs_parser
def test_the_presets_row_names_what_a_short_computer_greys_and_why(monkeypatch) -> None:
    from cm1core.cm1run import gpuinfo
    from cm1core.cm1run.system import presets_row

    roomy = gpuinfo.HostInfo(disk_root="/x", disk_free_bytes=10**12, disk_total_bytes=2 * 10**12,
                             ram_total_bytes=64 * 10**9, ram_available_bytes=60 * 10**9, wsl2=False)
    big = gpuinfo.GpuQuery("ok", gpus=(_gpu(name="NVIDIA GeForce RTX 5090", total_bytes=32 * 1024**3,
                                            free_bytes=31 * 1024**3, compute_capability="12.0"),))
    row = presets_row(big, roomy)
    assert row is not None and row.state == "ok", row
    assert re.match(r"\d+ of the \d+ storms fit this computer at their own time\.", row.text)
    assert "greyed in this release on every computer" in row.text
    # A nearly full card under WSL2: the big storms are greyed, with the card's own reason and the fix.
    busy = gpuinfo.GpuQuery("ok", gpus=(_gpu(total_bytes=10 * 1024**3, free_bytes=1024**3, wsl2=True),), wsl2=True)
    row = presets_row(busy, roomy)
    assert row is not None and row.state == "note", row  # the smallest storms still fit
    assert ".;" not in row.text and row.text.endswith(".")
    assert "Greyed here:" in row.text and "GPU memory" in row.text
    assert row.fix.startswith("Close programs that use the graphics card") and "4-5 GB" in row.fix
    # A full disk greys every storm.
    full = gpuinfo.HostInfo(disk_root="/x", disk_free_bytes=2 * 10**8, disk_total_bytes=10**11,
                            ram_total_bytes=64 * 10**9, ram_available_bytes=60 * 10**9, wsl2=False)
    row = presets_row(big, full)
    assert row is not None and row.state == "fail" and "free disk" in row.text
    assert row.fix.startswith("Free some disk space")
    # No usable GPU: the driver and GPU rows already say why.
    assert presets_row(gpuinfo.GpuQuery("no-nvidia-smi"), roomy) is None
    old = gpuinfo.GpuQuery("ok", gpus=(_gpu(compute_capability="6.1"),))
    assert presets_row(old, roomy) is None


def test_doctor_plain_ends_with_how_to_ask_for_help(tmp_path, monkeypatch, capsys) -> None:
    from cm1core.cli import main

    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", "none")
    monkeypatch.setenv("GPUWM_CM1_WSL2", "0")
    assert main(["doctor", "--plain", "--root", str(tmp_path)]) == 1
    text = capsys.readouterr().out
    assert "gpuwm-cm1 doctor --report" in text.splitlines()[-1]
    assert "GM1 is ready when no line starts with FIX." in text and "Exit 0" not in text
    # Plain `gpuwm-cm1 doctor` is the same check (the developer report is --science), and fails without a driver.
    assert main(["doctor", "--root", str(tmp_path)]) == 1
    again = capsys.readouterr().out
    assert again.splitlines()[0] == text.splitlines()[0] and "FIX   Driver" in again
    assert "GM1 is ready when no line starts with FIX." in again
    # Every failure carries its fix line.
    lines = text.splitlines()
    for number, line in enumerate(lines):
        if line.startswith("FIX "):
            assert "fix: " in lines[number + 1], line


# ---------------------------------------------------------------- 4b. doctor --report

def test_scrub_removes_the_home_folder_other_users_and_the_host_name() -> None:
    from cm1core.cm1run.report import scrub

    home = f"{HOMES}alice"
    text = (f"runs in {home}/gm1-runs and {home} | {home}x/y | {HOMES}bob/data | /mnt/c/Users/Alice/Desktop "
            + "C:" + chr(92) + "Users" + chr(92) + "Alice" + chr(92) + "x"
            + " on host blue-box.lan (blue-box) | Ubuntu 24.04 blue-boxes")
    out = scrub(text, home=home, hosts=["blue-box.lan", "blue-box"])
    assert "alice" not in out.lower() and "bob" not in out and "blue-box " not in out
    assert "~/gm1-runs" in out and f"{HOMES}<user>/data" in out and "/mnt/c/Users/<user>/Desktop" in out
    assert "C:" + chr(92) + "Users" + chr(92) + "<user>" in out
    assert "on host <host> (<host>)" in out and "blue-boxes" in out
    # A folder name ends at punctuation, which stays.
    out = scrub(f"{home}: 734 GB free. Runs in {home}. Other: {HOMES}bob: 1 GB, ({HOMES}carol)", home=home,
                hosts=[])
    assert out == f"~: 734 GB free. Runs in ~. Other: {HOMES}<user>: 1 GB, ({HOMES}<user>)"


def _failed_run(root: Path) -> Path:
    rundir = root / "my-first-storm"
    segment = rundir / "segments" / "001"
    segment.mkdir(parents=True)
    (rundir / "run.json").write_text(json.dumps({"schema": "gm1.cm1-run/v1", "segments": []}), encoding="utf-8")
    records = [
        {"kind": "admitted", "pid": 999999, "host": "elsewhere", "duration_s": 7200.0,
         "time_utc": "2026-10-10T00:00:00Z"},
        {"kind": "end", "reason": "error", "pid": 999999, "model_time_s": 600.0,
         "message": f"out of memory while writing {root}/my-first-storm", "time_utc": "2026-10-10T00:10:00Z"},
    ]
    (segment / "events.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records),
                                          encoding="utf-8")
    (segment / "run.log").write_text("".join(f"step {number}\n" for number in range(40)) + "Traceback: boom\n",
                                     encoding="utf-8")
    return rundir


def test_doctor_report_is_one_block_with_versions_checks_and_the_last_run(tmp_path, monkeypatch, capsys) -> None:
    from cm1core.cli import main
    from cm1core.cm1run import report

    home = tmp_path / "home"
    runs = home / "gm1-runs"
    runs.mkdir(parents=True)
    _failed_run(runs)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", str(GPUINFO / "fake_nvidia_smi.py"))
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", str(GPUINFO / "rtx3080-wsl2-desktop-4p8.json"))
    monkeypatch.setenv("GPUWM_CM1_WSL2", "1")
    monkeypatch.setenv("GPUWM_CM1_HOST_FIXTURE", str(GPUINFO / "host-64gb-1tb.json"))
    monkeypatch.setattr(report, "_host_names", lambda: ["blue-box"])
    assert main(["doctor", "--report"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("### GM1 doctor report\n")
    for label in ("| GM1 | gpuwm-cm1 ", "| ArWen | ", "| CuPy | ", "| NVRTC | ", "| numpy | ", "| Python | ",
                  "| OS | ", "| WSL2 | yes |", "| glibc | ", "| NVIDIA driver | ", "| GPU 0 | NVIDIA GeForce RTX 3080, sm_86",
                  "| Installed with | ", "| Runs folder | ~/gm1-runs |"):
        assert label in text, label
    assert "**Doctor:** " in text and "| ok | Driver |" in text and "| Python |" in text
    assert "**Last run:** `my-first-storm`: Failed. out of memory while writing ~/gm1-runs/my-first-storm" in text
    assert "<details><summary>The end of its log</summary>" in text and "Traceback: boom" in text
    assert "step 16\n" in text and "step 15\n" not in text  # the last 25 lines
    assert str(tmp_path) not in text and "blue-box" not in text
    # Without runs, it says so; and it never needs a GPU.
    assert report.last_run(tmp_path / "empty") is None


def test_doctor_report_with_no_driver_still_prints_everything(tmp_path, monkeypatch) -> None:
    from cm1core.cm1run import gpuinfo
    from cm1core.cm1run.report import doctor_report

    monkeypatch.setenv("GPUWM_CM1_HOST_FIXTURE", str(GPUINFO / "host-64gb-1tb.json"))
    text = doctor_report(tmp_path, query=gpuinfo.GpuQuery("no-nvidia-smi", message="none", wsl2=True))
    assert "| NVIDIA driver | none found (no-nvidia-smi) |" in text and "| FIX" not in text
    assert "| fail | Driver |" in text and "wsl --shutdown" in text and "**Last run:** none" in text


# ---------------------------------------------------------------- 5. the support pages

DOCS = ROOT / "docs"


def test_every_guide_gives_the_same_two_one_liners() -> None:
    for path in (ROOT / "README.md", DOCS / "INSTALL.md", DOCS / "QUICKSTART.md"):
        text = path.read_text(encoding="utf-8")
        assert f"```powershell install\n{PS_LINE}\n```" in text, path.name
        assert f"```sh install\n{SH_LINE}\n```" in text, path.name
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    install = readme[readme.index("## Install"):readme.index("## What v0.1 does")]
    # The pip lines are INSTALL.md's three: a new environment, then pip (a system Python refuses pip install).
    assert "```sh install\npython3 -m venv ~/gm1\n. ~/gm1/bin/activate\npip install gpuwm-cm1\n```" in install
    assert "gpuwm-cm1 gui" in install and "not Command Prompt" in install
    assert "pip in Windows PowerShell cannot install GM1" in install
    # The first questions are answered before anyone asks.
    for words in ("**Mac?**", "**AMD or Intel graphics?**", "**A laptop?**", "**Is anything sent over the internet?**"):
        assert words in install, words
    assert "TROUBLESHOOTING.md" in install and "INSTALL.md" in install
    assert "-c constraints" not in install  # the exact pins are in INSTALL.md


def test_troubleshooting_covers_the_top_problems_in_plain_words() -> None:
    from cm1core.gui.copy_lint import lint_markdown

    path = DOCS / "TROUBLESHOOTING.md"
    text = path.read_text(encoding="utf-8")
    for words in ("No NVIDIA driver found", "wsl --update", "wsl --shutdown", "never install an",
                  "https://www.nvidia.com/Download/index.aspx", "Port 8765 is in use", "--port",
                  "out-of-memory", "A storm card is greyed", "The pictures are blank at first",
                  "20 to 30 minutes of storm time", '"Supercell, 100 m" is greyed', "~/gm1-runs",
                  "Update", "Uninstall", "gpuwm-cm1 doctor --report", "gpuwm-cm1 doctor --plain",
                  "issues/new/choose", "0x80370102", "command not found", "externally-managed-environment",
                  "ensurepip is not available", "GM1 check", "gm1-report.txt", "My storm stopped overnight",
                  "Doctor says RAM is low (Windows)", "--set-sparse true", "wsl --unregister",
                  "never a running storm", "stops a running storm", "pip cache purge"):
        assert words in text, words
    assert "shuts Ubuntu down" not in text and "the GM1 window was closed" not in text
    assert lint_markdown(path) == []


def test_the_bug_form_asks_for_the_doctor_report_and_points_to_troubleshooting_first() -> None:
    form = (ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").read_text(encoding="utf-8")
    for field in ("id: report", "id: what-you-did", "id: what-happened"):
        block = form[form.index(field):]
        block = block[:block.index("validations:") + 40]
        assert "required: true" in block, field
    assert "gpuwm-cm1 doctor --report" in form and "TROUBLESHOOTING.md" in form
    config = (ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml").read_text(encoding="utf-8")
    assert "blank_issues_enabled: false" in config and "docs/TROUBLESHOOTING.md" in config
    # Questions (Mac, AMD, privacy) have their answers one click away, before the bug form.
    assert "#common-questions" in config and "GM1 check" in form


def test_the_installers_and_support_pages_ship_in_the_snapshot_not_the_wheel() -> None:
    import runpy

    stage = ROOT / "tools" / "release" / "stage_snapshot.py"
    if stage.is_file():
        allowed = runpy.run_path(str(stage))["allowed"]
        for path in ("install.sh", "install.ps1", "docs/TROUBLESHOOTING.md", ".github/ISSUE_TEMPLATE/bug_report.yml",
                     ".github/ISSUE_TEMPLATE/config.yml", ".gitattributes"):
            assert allowed(path), path
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include docs/TROUBLESHOOTING.md" in manifest and "install.sh" not in manifest
    # A Windows checkout keeps the shell script's Linux line ends.
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in attributes
