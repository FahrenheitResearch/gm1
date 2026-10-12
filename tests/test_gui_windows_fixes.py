"""What a first real run on Windows (WSL2) showed, kept fixed.

1. The browser: the link opens through ``rundll32.exe url.dll,FileProtocolHandler`` first and never through
   ``explorer.exe`` (which opens a File Explorer window for a link with a query); "Opened" is said only when the
   opener exited with success, a hung opener is killed, and otherwise the console says "Open this link in your
   browser:".
2. A storm that is running is what a freshly opened page shows; a second start brings the open page up.
3. The link is kept with the runs folder (mode 0600), so an old tab still works after a restart; a browser
   without the token gets a page in plain words, a program gets JSON; ``--new-token`` makes a new link.
4. Under WSL2 every folder is also named as File Explorer names it, with an Open folder action.
5. The Runs list says each run's size and the total, and deletes a run or its big files (never one that is going).
6. The doctor's disk line says what it measures under WSL2.

CPU only, the fake runner, loopback ports, no Windows program is run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys

import pytest

from gui_support import import_supercell, start_server, stop_server, wait_for

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/cm1core/gui/static"
URL = "http://127.0.0.1:8765/?token=TfOy28lyoKKWvB0fdINdewiaGpjixRCBsljLKIpQE-_"
WINDOWS = {"rundll32.exe": "/mnt/c/Windows/System32/rundll32.exe", "cmd.exe": "/mnt/c/Windows/System32/cmd.exe",
           "powershell.exe": "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
           "explorer.exe": "/mnt/c/Windows/explorer.exe"}
BACK = chr(92)


class _Process:
    def __init__(self, code: int | None, pid: int = 0) -> None:
        self.code, self.pid, self.killed = code, pid, False

    def wait(self, timeout=None):  # noqa: ANN001 - subprocess.Popen's shape
        if self.code is None and not self.killed:
            raise subprocess.TimeoutExpired("opener", timeout)
        return self.code if self.code is not None else -9

    def kill(self) -> None:
        self.killed = True


class _Popen:
    """Stands in for subprocess.Popen: each program exits with the code given for its name (None: never)."""

    def __init__(self, codes: dict[str, int | None] | None = None, missing: tuple[str, ...] = ()) -> None:
        self.codes, self.missing = codes or {}, missing
        self.calls: list[tuple[list[str], str | None]] = []
        self.made: list[_Process] = []

    def __call__(self, words, **kwargs):  # noqa: ANN001
        name = Path(words[0]).name
        if name in self.missing:
            raise OSError(8, "Exec format error")
        # Detached: no stream of this process, its own session, no other descriptor.
        assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
        assert kwargs["start_new_session"] is True and kwargs["close_fds"] is True
        self.calls.append((list(words), kwargs.get("cwd")))
        process = _Process(self.codes.get(name, 0))
        self.made.append(process)
        return process


def _open(popen, url: str = URL, which=WINDOWS.get, exists=lambda _p: True, **more):  # noqa: ANN001
    from cm1core.gui.browser import open_link

    return open_link(url, env={}, wsl=True, terminal=True, which=which, exists=exists, popen=popen, wait_s=0.01,
                     **more)


# ---------------------------------------------------------------- 1. the browser opener

def test_the_link_opens_through_rundll32_first_and_never_through_explorer() -> None:
    from cm1core.gui.browser import wsl_openers

    openers = wsl_openers(URL, WINDOWS.get, lambda _p: True)
    assert [name for name, _words, _cwd in openers] == ["rundll32.exe", "cmd.exe", "powershell.exe"]
    by_name = {name: (words, cwd) for name, words, cwd in openers}
    # The link is one argument, as written: its ?, = and - reach Windows untouched.
    assert by_name["rundll32.exe"] == ([WINDOWS["rundll32.exe"], "url.dll,FileProtocolHandler", URL], "/mnt/c")
    # cmd.exe complains about a Linux working folder, so it starts on the Windows drive; "" is start's title.
    assert by_name["cmd.exe"] == ([WINDOWS["cmd.exe"], "/c", "start", "", URL], "/mnt/c")
    assert by_name["powershell.exe"][0] == [WINDOWS["powershell.exe"], "-NoProfile", "-NonInteractive", "-Command",
                                            f"Start-Process '{URL}'"]
    # explorer.exe opens a File Explorer window for a link with a query: it is never an opener of links.
    assert not any("explorer" in " ".join(words).lower() for _name, words, _cwd in openers)
    # wslview, when it is installed, is the last resort.
    found = {**WINDOWS, "wslview": "/usr/bin/wslview"}
    assert [name for name, _w, _c in wsl_openers(URL, found.get, lambda _p: True)][-1] == "wslview"
    # A link cmd.exe would cut at "&" is not given to cmd.exe; one with a quote is not given to PowerShell.
    assert "cmd.exe" not in [name for name, _w, _c in wsl_openers(URL + "&x=1", WINDOWS.get, lambda _p: True)]
    assert "powershell.exe" not in [name for name, _w, _c in wsl_openers(URL + "'", WINDOWS.get, lambda _p: True)]
    # Windows' programs off PATH are found in their usual place.
    usual = wsl_openers(URL, lambda _name: None, lambda _p: True)
    assert usual[0][1][0] == WINDOWS["rundll32.exe"] and len(usual) == 3
    assert wsl_openers(URL, lambda _name: None, lambda _p: False) == []


def test_opened_is_said_only_when_the_opener_exits_with_success() -> None:
    popen = _Popen()
    assert _open(popen) == (True, "rundll32.exe")
    assert popen.calls == [([WINDOWS["rundll32.exe"], "url.dll,FileProtocolHandler", URL], "/mnt/c")]
    # The first fails (it exits 1): the next is tried, and named.
    popen = _Popen({"rundll32.exe": 1})
    assert _open(popen) == (True, "cmd.exe")
    assert [Path(words[0]).name for words, _cwd in popen.calls] == ["rundll32.exe", "cmd.exe"]
    # One that cannot be started at all (WSL's Windows programs switched off) is passed over the same way.
    popen = _Popen(missing=("rundll32.exe", "cmd.exe"))
    assert _open(popen) == (True, "powershell.exe")
    # Every opener fails: not opened, and the reason names what was tried.
    popen = _Popen({"rundll32.exe": 1, "cmd.exe": 1, "powershell.exe": 1})
    opened, why = _open(popen)
    assert opened is False and "Windows did not open it" in why and "rundll32.exe" in why
    assert _open(_Popen(), which=lambda _n: None, exists=lambda _p: False) == (
        False, "no Windows program to open it with was found")


def test_an_opener_that_does_not_exit_is_killed_and_is_not_success() -> None:
    popen = _Popen({"rundll32.exe": None, "cmd.exe": None, "powershell.exe": None})
    opened, why = _open(popen)
    assert opened is False
    # It is not left behind to keep WSL up.  No other opener follows it: a slow computer may have opened the
    # page before the helper was ended, and each further helper would open one more tab.
    assert len(popen.made) == 1 and popen.made[0].killed
    assert [Path(words[0]).name for words, _cwd in popen.calls] == ["rundll32.exe"]
    assert why == "rundll32.exe did not answer in 0 s; the page may have opened all the same"
    # One that failed outright is still followed by the next; the one after that hangs, and the trying ends there.
    popen = _Popen({"rundll32.exe": 1, "cmd.exe": None})
    opened, why = _open(popen)
    assert opened is False and why.startswith("cmd.exe did not answer")
    assert [Path(words[0]).name for words, _cwd in popen.calls] == ["rundll32.exe", "cmd.exe"]
    from cm1core.gui.browser import helper_outcome, run_helper

    assert helper_outcome(["x"], popen=_Popen({"x": None}), wait_s=0.01) == ("timeout", None)
    assert helper_outcome(["x"], popen=_Popen({"x": 2})) == ("exited", 2)
    assert helper_outcome(["x"], popen=_Popen(missing=("x",))) == ("failed", None)

    assert run_helper(["x"], popen=_Popen({"x": None}), wait_s=0.01) is None
    assert run_helper(["x"], popen=_Popen({"x": 0})) == 0
    assert run_helper(["x"], popen=_Popen(missing=("x",))) is None


def test_a_real_helper_is_detached_and_reaped() -> None:
    """With the real Popen: a helper's exit code comes back, and one that hangs is gone afterwards."""

    from cm1core.gui.browser import run_helper

    assert run_helper([sys.executable, "-c", "raise SystemExit(0)"]) == 0
    assert run_helper([sys.executable, "-c", "raise SystemExit(3)"]) == 3
    marker = f"gm1-hung-helper-{os.getpid()}"
    assert run_helper([sys.executable, "-c", f"import time; time.sleep(60)  # {marker}"], wait_s=0.5) is None
    listing = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    assert marker not in listing


def test_the_console_says_opened_or_gives_the_link_to_open() -> None:
    from cm1core.gui.server import opened_words

    assert opened_words(True, "rundll32.exe", URL) == (
        "Opened the link in your browser (rundll32.exe; gpuwm-cm1 gui --no-browser only prints it).")
    words = opened_words(False, "Windows did not open it (rundll32.exe tried)", URL)
    assert "Opened" not in words and "Open this link in your browser:" in words and words.endswith(URL)
    # A helper that did not answer: the link is printed, with the plain reason.
    words = opened_words(False, "rundll32.exe did not answer in 6 s; the page may have opened all the same", URL)
    assert "Opened" not in words and "the page may have opened all the same" in words and words.endswith(URL)


def _gui(tmp_path: Path, monkeypatch, *extra: str, serve=None, runner: str = "--fake-runner") -> int:
    from cm1core.cli import main
    from cm1core.gui import server

    def closed(self, poll_interval=0.5):  # noqa: ANN001 - serve_forever's shape
        if serve is not None:
            serve(self)
        raise KeyboardInterrupt

    monkeypatch.setattr(server.GuiServer, "serve_forever", closed)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    args = ["gui", "--root", str(tmp_path / "runs"), "--port", "0", "--config-dir", str(tmp_path / "config"), *extra]
    return main(args + ([runner] if runner else []))


def test_gui_never_says_opened_when_the_opener_failed(tmp_path, monkeypatch, capsys) -> None:
    from cm1core.gui import browser

    monkeypatch.setattr(browser, "open_link", lambda url: (False, "Windows did not open it (rundll32.exe tried)"))
    assert _gui(tmp_path, monkeypatch) == 0
    out = capsys.readouterr().out
    assert "Opened" not in out
    assert "The browser was not opened (Windows did not open it (rundll32.exe tried)). Open this link in your browser:" in out
    link = [line.strip() for line in out.splitlines() if line.strip().startswith("http://127.0.0.1:")]
    assert len(link) == 2 and link[0] == link[1]


# ---------------------------------------------------------------- 2. opened again

def test_a_second_start_brings_the_open_page_up_or_gives_its_link(monkeypatch, capsys) -> None:
    from cm1core.gui import browser, server as S

    held: list[float] = []
    monkeypatch.setattr(S, "_hold_window", held.append)
    # From the Windows shortcut: a browser is tried, and the window would close when this returns.
    monkeypatch.setattr(browser, "why_not", lambda *a, **k: None)
    monkeypatch.setattr(browser, "open_link", lambda url: (True, "rundll32.exe"))
    assert S.show_running(URL, wsl=True) == 0
    out = capsys.readouterr().out
    assert "GM1 is already open. Opened it in your browser again (rundll32.exe)." in out and URL in out
    monkeypatch.setattr(browser, "open_link", lambda url: (False, "Windows did not open it (x tried)"))
    assert S.show_running(URL, wsl=True) == 0
    out = capsys.readouterr().out
    assert "Opened" not in out and "Open this link in your browser:" in out and URL in out
    # The window stays up long enough to read: a few seconds after the page came up, far longer when it did not.
    assert held == [S.SHOWN_AGAIN_S, S.NOT_SHOWN_S] and S.NOT_SHOWN_S >= 60
    # --no-browser: the link, and no waiting.
    assert S.show_running(URL, browser=False, wsl=True) == 0
    assert held == [S.SHOWN_AGAIN_S, S.NOT_SHOWN_S] and URL in capsys.readouterr().out
    # Not a terminal (a script, a test): nothing waits.
    monkeypatch.undo()
    S._hold_window(30.0)


def test_a_new_token_is_refused_while_gm1_is_open(tmp_path, monkeypatch, capsys) -> None:
    from cm1core.cli import main
    from cm1core.gui import server as S

    root = tmp_path / "runs"
    first, _client = start_server(root, tmp_path / "config")
    try:
        assert S.write_running(first) is not None
        monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
        assert main(["gui", "--root", str(root), "--config-dir", str(tmp_path / "config"), "--new-token"]) == 2
        assert "its link cannot change while it is" in capsys.readouterr().err
    finally:
        stop_server(first)


def test_the_first_screens_lead_with_the_running_storm() -> None:
    running = (STATIC / "js/running.js").read_text(encoding="utf-8")
    app = (STATIC / "js/app.js").read_text(encoding="utf-8")
    assert "`A storm is running: ${run.name}, ${Math.round(st.percent)}%`" in running
    assert 'button("Watch it"' in running
    # Watching in guided mode, Run in advanced: both modes.
    assert 'mode === "guided" ? `guided/watch/${run.url}` : `run/${run.url}`' in running
    # Exactly one run going, and the page just opened: straight to it, once per page load.
    assert "going.length === 1 ? watchRoute(going[0], mode) : null" in running
    assert "if (!landed) {" in app and "landingRoute(api.currentMode())" in app and "history.replaceState" in app
    # The card is on each mode's first screen and on Runs, not on a run's own pages.
    assert 'const FIRST_SCREENS = ["guided", "new-run", "runs"];' in app
    assert 'FIRST_SCREENS.includes(screen) && !(screen === "guided" && args[0])' in app


def test_the_runs_list_tells_a_fresh_page_which_storm_is_going(tmp_path, monkeypatch) -> None:
    """What the card and the landing read: /api/runs names the run that is going, with its share done."""

    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.02")
    monkeypatch.setenv("GM1_FAKE_STARTUP_S", "0.05")
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        import_supercell(client, "done")
        import_supercell(client, "going")
        assert client.post("/api/runs/going/start").json["ok"]
        client.events("going", lambda e: e.get("event") == "progress" and e["data"].get("step"), timeout=30)
        rows = {row["id"]: row for row in client.get("/api/runs").json["runs"]}
        going = [row for row in rows.values() if row["status"]["state"] in ("running", "waiting")]
        assert [row["id"] for row in going] == ["going"] and going[0]["url"] == "going"
        assert isinstance(going[0]["status"]["percent"], (int, float))
        assert client.post("/api/runs/going/stop").json["ok"]
        assert wait_for(lambda: client.get("/api/runs/going/status").json["state"] == "stopped")
    finally:
        stop_server(server)


# ---------------------------------------------------------------- 3. the kept token

def test_the_token_is_kept_with_the_runs_folder_for_its_owner_only(tmp_path) -> None:
    from cm1core.gui import auth

    folder = tmp_path / "runs" / ".gm1-gui"
    first = auth.stored_token(folder)
    path = folder / auth.TOKEN_FILE
    assert len(first) == 43 and path.read_text(encoding="ascii").strip() == first
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert auth.stored_token(folder) == first == auth.stored_token(folder)
    # On request, a new one; the old one is gone for good.
    second = auth.stored_token(folder, rotate=True)
    assert second != first and auth.stored_token(folder) == second
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # A file others can read is no secret, and one that holds no token is no token: both are replaced.
    path.chmod(0o644)
    third = auth.stored_token(folder)
    assert third != second and stat.S_IMODE(path.stat().st_mode) == 0o600
    path.write_text("short\n", encoding="ascii")
    path.chmod(0o600)
    fourth = auth.stored_token(folder)
    assert fourth not in ("short", third) and len(fourth) == 43
    # A link in its place is not followed.
    path.unlink()
    (tmp_path / "elsewhere").write_text(first + "\n", encoding="ascii")
    (tmp_path / "elsewhere").chmod(0o600)
    path.symlink_to(tmp_path / "elsewhere")
    assert auth.stored_token(folder) != first and not path.is_symlink()


def _printed_link(capsys) -> str:
    out = capsys.readouterr().out
    return next(line.strip() for line in out.splitlines() if line.strip().startswith("http://127.0.0.1:"))


def test_the_link_survives_a_restart_until_a_new_token_is_asked_for(tmp_path, monkeypatch, capsys) -> None:
    from cm1core.gui import auth

    assert _gui(tmp_path, monkeypatch, "--no-browser") == 0
    first = _printed_link(capsys).split("token=")[1]
    assert first == (tmp_path / "runs" / ".gm1-gui" / auth.TOKEN_FILE).read_text(encoding="ascii").strip()
    assert _gui(tmp_path, monkeypatch, "--no-browser") == 0
    assert _printed_link(capsys).split("token=")[1] == first
    assert _gui(tmp_path, monkeypatch, "--no-browser", "--new-token") == 0
    out = capsys.readouterr().out
    assert first not in out and "links and tabs from earlier starts no longer work" in out
    # Another runs folder has its own.
    assert _gui(tmp_path / "other", monkeypatch, "--no-browser") == 0
    assert _printed_link(capsys).split("token=")[1] != first


def test_a_gui_other_computers_can_reach_never_uses_the_kept_token(tmp_path) -> None:
    import argparse

    from cm1core.gui.server import kept_token

    def arguments(**fields):  # noqa: ANN003
        return argparse.Namespace(**{"root": tmp_path / "runs", "bind": "127.0.0.1", "allow_remote": False,
                                     "new_token": False, **fields})

    local = kept_token(arguments())
    assert local and kept_token(arguments()) == local
    assert kept_token(arguments(allow_remote=True)) is None
    assert kept_token(arguments(bind="0.0.0.0", allow_remote=True)) is None


def test_a_browser_without_the_token_gets_a_page_and_a_program_gets_json(tmp_path) -> None:
    from cm1core.gui import auth

    # Under Windows (WSL2) the words name the Start menu; the Linux words have their own test below.
    os.environ["GPUWM_CM1_WSL2"] = "1"
    try:
        server, client = start_server(tmp_path / "runs", tmp_path / "config")
    finally:
        del os.environ["GPUWM_CM1_WSL2"]
    try:
        sentence = "This link is out of date. Start GM1 again (Start menu: GM1) and use the page it opens."
        assert auth.OUT_OF_DATE == sentence == auth.out_of_date(True)
        for path in ("/", "/?token=" + "y" * 43, "/index.html", "/some/old/page"):
            page = client.get(path, token=False)
            assert page.status == 403 and page.headers["content-type"] == "text/html; charset=utf-8", path
            text = page.raw.decode("utf-8")
            assert "This link is out of date." in text and "Start GM1 again (Start menu: GM1)" in text
            assert "use the page it opens" in text and "<script" not in text and "{" + '"' not in text
            # The same guards as every reply, and nothing of the server's in the page.
            assert page.headers["x-frame-options"] == "DENY" and server.token not in text
        # The API and the page's own files still answer a program in JSON.
        for path in ("/api/session", "/api/runs", "/static/js/app.js"):
            answer = client.get(path, token=False)
            assert answer.status == 403 and answer.headers["content-type"].startswith("application/json"), path
            assert answer.json["ok"] is False and answer.json["message"] == sentence
        post = client.post("/api/mode", {"mode": "advanced"}, token=False)
        assert post.status == 403 and post.headers["content-type"].startswith("application/json")
        # The token is still what lets a request in, and the cookie it sets is what it was.
        landed = client.get(f"/?token={server.token}", token=False)
        assert landed.status == 303 and landed.headers["location"] == "/"
        assert landed.headers["set-cookie"] == f"gm1_token_{server.port}={server.token}; Path=/; HttpOnly; SameSite=Strict"
        cookie = landed.headers["set-cookie"].split(";")[0]
        assert client.get("/", token=False, headers={"Cookie": cookie}).status == 200
        # The cookie alone still never authorizes a POST.
        assert client.post("/api/mode", {"mode": "advanced"}, token=False, headers={"Cookie": cookie}).status == 403
    finally:
        stop_server(server)


def test_the_record_of_a_closed_gui_goes(tmp_path, monkeypatch) -> None:
    from cm1core.gui import server as S
    from cm1core.gui.files import SERVER_DIR

    record = tmp_path / "runs" / SERVER_DIR / S.RUNNING_FILE
    record.parent.mkdir(parents=True)
    record.write_text('{"pid": 999999999, "port": 1, "url": "http://127.0.0.1:1/?token=x"}', encoding="utf-8")
    assert S.running_gui(tmp_path / "runs") is None
    S.forget_stale(tmp_path / "runs")
    assert not record.exists()
    S.forget_stale(tmp_path / "runs")  # nothing there: nothing happens

    # The window is closed (SIGHUP): the GUI leaves as it does on Ctrl+C, and its record goes with it.
    seen: dict[str, bool] = {}

    def window_closed(server) -> None:  # noqa: ANN001
        seen["record"] = record.is_file()
        os.kill(os.getpid(), signal.SIGHUP)
        signal.pause()

    before = signal.getsignal(signal.SIGHUP)
    assert _gui(tmp_path, monkeypatch, "--no-browser", serve=window_closed, runner="") == 0
    assert seen == {"record": True} and not record.exists()
    assert signal.getsignal(signal.SIGHUP) == before


# ---------------------------------------------------------------- 4. Windows names for folders

def test_a_linux_path_is_named_as_file_explorer_names_it() -> None:
    from cm1core.cm1run import wslpaths

    home = "/".join(("", "home", "ada", "gm1-runs"))
    wanted = BACK * 2 + BACK.join(("wsl.localhost", "Ubuntu-24.04", "home", "ada", "gm1-runs"))
    assert wslpaths.built_windows_path(home, "Ubuntu-24.04") == wanted
    assert wslpaths.built_windows_path("/mnt/c/Users/ada/Desktop", None) == "C:" + BACK + BACK.join(
        ("Users", "ada", "Desktop"))
    assert wslpaths.built_windows_path("/mnt/d", None) == "D:" + BACK
    assert wslpaths.built_windows_path(home, None) is None and wslpaths.built_windows_path("runs", "U") is None
    assert wslpaths.built_windows_path(home, "bad name") is None
    # Not WSL: there is no Windows name.
    assert wslpaths.windows_path(home, wsl=False) is None

    asked: list[list[str]] = []

    def wslpath(words, **kwargs):  # noqa: ANN001 - subprocess.run's shape
        asked.append(list(words))
        return subprocess.CompletedProcess(words, 0, stdout=(wanted + "\n").encode())

    env = {"WSL_DISTRO_NAME": "Other"}
    assert wslpaths.windows_path(home, env=env, wsl=True, run=wslpath, which=lambda _n: "/usr/bin/wslpath") == wanted
    assert asked == [["/usr/bin/wslpath", "-w", home]]

    def broken(words, **kwargs):  # noqa: ANN001
        raise OSError("no such program")

    # wslpath does not answer: the name is built from the distribution's name.
    built = wslpaths.windows_path(home, env=env, wsl=True, run=broken, which=lambda _n: "/usr/bin/wslpath")
    assert built == wanted.replace("Ubuntu-24.04", "Other")
    assert wslpaths.windows_path(home, env=env, wsl=True, which=lambda _n: None) == built


def test_the_windows_drive_that_holds_the_linux_disk_is_found_from_the_registry() -> None:
    from cm1core.cm1run import wslpaths

    key = "HKEY_CURRENT_USER" + BACK + "Software" + BACK + "Lxss" + BACK
    users = "D:" + BACK + "Users" + BACK + "ada" + BACK + "wsl"
    listing = "\r\n".join([
        key + "{1}", "    DistributionName    REG_SZ    Other", "    BasePath    REG_SZ    C:" + BACK + "other", "",
        key + "{2}", "    DistributionName    REG_SZ    Ubuntu-24.04",
        "    BasePath    REG_SZ    " + BACK * 2 + "?" + BACK + users, "    Flags    REG_DWORD    0xf", ""])
    assert wslpaths.base_path(listing, "Ubuntu-24.04") == BACK * 2 + "?" + BACK + users
    assert wslpaths.base_path(listing, "Other") == "C:" + BACK + "other"
    assert wslpaths.base_path(listing, "Missing") is None

    class Usage:
        free, total = 5 * 10**9, 100 * 10**9

    def reg(words, **kwargs):  # noqa: ANN001
        assert Path(words[0]).name == "reg.exe" and words[1] == "query" and words[-1] == "/s"
        return subprocess.CompletedProcess(words, 0, stdout=listing.encode())

    seen: list[str] = []

    def usage(mount: str):  # noqa: ANN202
        seen.append(mount)
        return Usage()

    found = wslpaths.windows_drive(env={"WSL_DISTRO_NAME": "Ubuntu-24.04"}, wsl=True, run=reg,
                                   which=lambda _n: None, exists=lambda _p: True, usage=usage, ismount=lambda _m: True)
    assert found == {"drive": "D:", "free_bytes": 5 * 10**9, "total_bytes": 100 * 10**9} and seen == ["/mnt/d"]
    common = {"env": {"WSL_DISTRO_NAME": "Ubuntu-24.04"}, "wsl": True, "run": reg, "which": lambda _n: None,
              "exists": lambda _p: True, "usage": usage}
    # The drive is not mounted in WSL, the distribution is not listed, or this is not WSL: nothing is claimed.
    assert wslpaths.windows_drive(**common, ismount=lambda _m: False) is None
    assert wslpaths.windows_drive(**{**common, "env": {"WSL_DISTRO_NAME": "Missing"}}, ismount=lambda _m: True) is None
    assert wslpaths.windows_drive(**{**common, "env": {}}, ismount=lambda _m: True) is None
    assert wslpaths.windows_drive(**{**common, "wsl": False}, ismount=lambda _m: True) is None


def _windows_place(server, name: str = "Ubuntu-24.04") -> str:
    """Make the server answer as it does under WSL2, with ``name`` as the distribution."""

    base = BACK * 2 + BACK.join(("wsl.localhost", name, "runs"))
    server.api._place = {"wsl": True, "root_windows": base, "can_open_folder": True}
    return base


def test_under_wsl_every_folder_is_also_named_for_windows(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.001")
    monkeypatch.setenv("GM1_FAKE_STARTUP_S", "0.05")
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        # On Linux: the Linux path alone.
        session = client.get("/api/session").json
        assert session["wsl"] is False and session["root_windows"] is None
        import_supercell(client)
        results = client.get("/api/runs/sc1/results").json
        assert results["folder"] == str(tmp_path / "runs" / "sc1") and results["folder_windows"] is None
        base = _windows_place(server)
        assert client.get("/api/session").json["root_windows"] == base
        runs = client.get("/api/runs").json
        assert runs["root_windows"] == base and runs["can_open_folder"] is True and runs["root"] == str(tmp_path / "runs")
        results = client.get("/api/runs/sc1/results").json
        assert results["folder_windows"] == base + BACK + "sc1"
        assert results["cm1out_folder_windows"] == base + BACK + "sc1" + BACK + "cm1out"
        assert results["cm1out_folder"] == str(tmp_path / "runs" / "sc1" / "cm1out")
    finally:
        stop_server(server)


def test_open_folder_shows_the_folder_in_the_file_manager(tmp_path, monkeypatch) -> None:
    from cm1core.gui import browser

    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        import_supercell(client)
        base = _windows_place(server)
        shown: list[tuple[str, str | None]] = []
        monkeypatch.setattr(browser, "folder_opener", lambda **_k: ("explorer.exe", WINDOWS["explorer.exe"]))
        monkeypatch.setattr(browser, "open_folder",
                            lambda path, windows=None, **_k: (shown.append((str(path), windows)) or True, "explorer.exe"))
        # Asked for the command only: nothing opens.
        dry = client.post("/api/open-folder", {"run": "sc1", "dry_run": True}).json
        assert dry["dry_run"] is True and dry["command"].startswith("explorer.exe ") and shown == []
        root = client.post("/api/open-folder", {}).json
        assert root["ok"] and root["windows_path"] == base and root["path"] == str(tmp_path / "runs")
        run = client.post("/api/open-folder", {"run": "sc1"}).json
        assert run["ok"] and run["windows_path"] == base + BACK + "sc1"
        assert shown == [(str(tmp_path / "runs"), base), (str(tmp_path / "runs" / "sc1"), base + BACK + "sc1")]
        # Only folders of the runs folder, and only with the token.
        assert client.post("/api/open-folder", {"run": "../x"}).status == 400
        assert client.post("/api/open-folder", {"run": "nothing"}).status == 404
        assert client.post("/api/open-folder", {}, token=False).status == 403
        # No screen of its own (an ssh session): refused, with the folder to open by hand.
        monkeypatch.setattr(browser, "folder_opener", lambda **_k: None)
        refused = client.post("/api/open-folder", {})
        assert refused.status == 409 and base in refused.json["fix"]
    finally:
        stop_server(server)


def test_file_explorer_is_given_the_windows_name_and_its_exit_code_is_not_judged() -> None:
    from cm1core.gui.browser import folder_opener, open_folder

    windows = BACK * 2 + BACK.join(("wsl.localhost", "Ubuntu-24.04", "runs"))
    # explorer.exe exits 1 even when its window opened.
    popen = _Popen({"explorer.exe": 1})
    assert open_folder("/runs", windows=windows, env={}, wsl=True, which=WINDOWS.get, exists=lambda _p: True,
                       popen=popen) == (True, "explorer.exe")
    assert popen.calls == [([WINDOWS["explorer.exe"], windows], "/mnt/c")]
    hung = _Popen({"explorer.exe": None})
    assert open_folder("/runs", windows=windows, env={}, wsl=True, which=WINDOWS.get, exists=lambda _p: True,
                       popen=hung, wait_s=0.01)[0] is False and hung.made[0].killed
    assert open_folder("/runs", windows=None, env={}, wsl=True, which=WINDOWS.get, exists=lambda _p: True,
                       popen=_Popen())[0] is False
    # A Linux desktop: xdg-open, given the Linux path; over ssh, or with no display, nothing.
    desktop = {"DISPLAY": ":0"}
    popen = _Popen()
    assert open_folder("/runs", env=desktop, wsl=False, platform="linux", which=lambda n: f"/usr/bin/{n}",
                       popen=popen) == (True, "xdg-open")
    assert popen.calls == [(["/usr/bin/xdg-open", "/runs"], None)]
    assert folder_opener(env={**desktop, "SSH_CONNECTION": "a b"}, wsl=False, platform="linux") is None
    assert folder_opener(env={}, wsl=False, platform="linux", which=lambda n: f"/usr/bin/{n}") is None
    assert folder_opener(env={"SSH_TTY": "/dev/pts/0"}, wsl=True, which=WINDOWS.get) is None


def test_the_console_names_the_runs_folder_for_windows(tmp_path, monkeypatch, capsys) -> None:
    from cm1core.cm1run import wslpaths

    wanted = BACK * 2 + BACK.join(("wsl.localhost", "Ubuntu-24.04", "runs"))
    monkeypatch.setenv("GPUWM_CM1_WSL2", "1")
    monkeypatch.setattr(wslpaths, "windows_path", lambda path, **_k: wanted)
    assert _gui(tmp_path, monkeypatch, "--no-browser") == 0
    out = capsys.readouterr().out
    assert f"Runs folder: {tmp_path / 'runs'}\n  in Windows (File Explorer): {wanted}\n" in out
    monkeypatch.setenv("GPUWM_CM1_WSL2", "0")
    assert _gui(tmp_path, monkeypatch, "--no-browser") == 0
    assert "in Windows" not in capsys.readouterr().out


# ---------------------------------------------------------------- 5. sizes and deleting

@pytest.fixture()
def finished(tmp_path, monkeypatch):
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.001")
    monkeypatch.setenv("GM1_FAKE_STARTUP_S", "0.05")
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    import_supercell(client, "storm")
    assert client.post("/api/runs/storm/start").json["ok"]
    client.events("storm", lambda e: e.get("event") == "end", timeout=60)
    assert wait_for(lambda: client.get("/api/runs/storm/status").json["state"] == "finished")
    wait_for(lambda: not server.jobs.launched("storm"), timeout=60)
    yield server, client, tmp_path / "runs"
    stop_server(server)


def test_the_runs_list_says_each_size_and_the_total(finished) -> None:
    server, client, root = finished
    import_supercell(client, "second")
    listing = client.get("/api/runs").json
    sizes = {row["id"]: row["disk_bytes"] for row in listing["runs"]}
    assert sizes["storm"] > sizes["second"] > 0
    assert listing["total_bytes"] == sum(sizes.values())
    assert all(row["trimmed"] is False for row in listing["runs"])


def test_delete_removes_a_finished_run_and_only_that(finished) -> None:
    server, client, root = finished
    import_supercell(client, "keep")
    size = client.get("/api/runs").json["runs"]
    size = next(row["disk_bytes"] for row in size if row["id"] == "storm")
    # Asked for its command only: nothing is deleted.
    dry = client.post("/api/runs/storm/delete", {"dry_run": True}).json
    assert dry["dry_run"] is True and dry["command"] == f"rm -r -- {root / 'storm'}" and (root / "storm").is_dir()
    # Without the token: refused.
    assert client.post("/api/runs/storm/delete", {}, token=False).status == 403 and (root / "storm").is_dir()
    done = client.post("/api/runs/storm/delete", {}).json
    assert done["ok"] and done["message"] == "Deleted storm." and done["kept_pictures"] is False
    assert 0.99 * size <= done["freed_bytes"] <= 1.01 * size and done["command"] == f"rm -r -- {root / 'storm'}"
    assert not (root / "storm").exists() and (root / "keep" / "run.json").is_file()
    assert [row["id"] for row in client.get("/api/runs").json["runs"]] == ["keep"]
    # Gone is gone; a name that leads outside the runs folder is never a run.
    assert client.post("/api/runs/storm/delete", {}).status == 404
    assert client.post("/api/runs/..%2Fkeep/delete", {}).status == 400
    assert client.post("/api/runs/keep/delete", {"keep_pictures": "yes"}).status == 400


def test_the_big_files_go_and_the_pictures_stay(finished) -> None:
    server, client, root = finished
    rundir = root / "storm"
    pictures = sorted(path.relative_to(rundir) for path in (rundir / "quicklook").rglob("*.png"))
    before = client.get("/api/runs/storm/results?mode=guided").json
    assert pictures and before["disk"]["cm1out_bytes"] > 0 and before["trimmed"] is None
    done = client.post("/api/runs/storm/delete", {"keep_pictures": True}).json
    assert done["ok"] and done["kept_pictures"] is True and "its pictures are kept" in done["message"]
    assert done["freed_bytes"] >= before["disk"]["cm1out_bytes"] + before["disk"]["frames_bytes"]
    assert not (rundir / "cm1out").exists()
    assert not list(rundir.glob("segments/*/frames")) and not list(rundir.glob("segments/*/restarts"))
    # The pictures, the record of the run and its log are all still there.
    assert sorted(path.relative_to(rundir) for path in (rundir / "quicklook").rglob("*.png")) == pictures
    assert (rundir / "run.json").is_file() and list(rundir.glob("segments/*/events.jsonl"))
    after = client.get("/api/runs/storm/results?mode=guided").json
    assert after["status"]["state"] == "finished" and after["trimmed"]["freed_bytes"] == done["freed_bytes"]
    assert after["cm1out"] == [] and len(after["looks"][0]["frames"]) == len(before["looks"][0]["frames"])
    assert client.get(after["looks"][0]["frames"][-1]["url"]).status == 200
    # Nothing is left to continue from, so Continue is not offered; the list marks the run.
    status = client.get("/api/runs/storm/status").json
    assert status["trimmed"] is True and "resume" not in status["actions"]
    row = client.get("/api/runs").json["runs"][0]
    assert row["trimmed"] is True and row["disk_bytes"] < before["disk"]["total_bytes"] - done["freed_bytes"] // 2
    assert "rm -r -- " in (rundir / "commands.log").read_text()
    # A second time there is nothing big left; the whole run can still be deleted.
    assert client.post("/api/runs/storm/delete", {"keep_pictures": True}).status == 409
    assert client.post("/api/runs/storm/delete", {}).json["ok"] and not rundir.exists()


def test_a_storm_that_is_running_is_never_deleted(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.02")
    monkeypatch.setenv("GM1_FAKE_STARTUP_S", "0.05")
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        import_supercell(client, "going")
        assert client.post("/api/runs/going/start").json["ok"]
        client.events("going", lambda e: e.get("event") == "progress" and e["data"].get("step"), timeout=30)
        for body in ({}, {"keep_pictures": True}):
            refused = client.post("/api/runs/going/delete", body)
            assert refused.status == 409 and refused.json["message"] == "This storm is still running."
            assert refused.json["fix"] == "Stop it first, then delete it." and "command" not in refused.json
        assert (tmp_path / "runs" / "going" / "run.json").is_file()
        assert client.post("/api/runs/going/stop").json["ok"]
        assert wait_for(lambda: client.get("/api/runs/going/status").json["state"] == "stopped")
        wait_for(lambda: not server.jobs.launched("going"), timeout=60)
        assert client.post("/api/runs/going/delete", {}).json["ok"]
    finally:
        stop_server(server)


def test_a_run_whose_pictures_are_still_being_drawn_is_not_deleted(finished) -> None:
    """The picture worker of a run started before this GUI was opened is not this server's process: its lock
    on the run folder is what says it is still drawing.  Seen on a real run: deleting the big files under it
    left 16 of 25 pictures."""

    import fcntl

    from cm1core.gui.api import work_in_progress

    server, client, root = finished
    rundir = root / "storm"
    assert work_in_progress(rundir) == []
    lock = rundir / ".quicklook-follow.lock"
    lock.touch()
    with open(lock, "rb") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert work_in_progress(rundir) == ["its pictures are still being drawn"]
        for body in ({"keep_pictures": True}, {}):
            refused = client.post("/api/runs/storm/delete", body)
            assert refused.status == 409 and "its pictures are still being drawn" in refused.json["message"]
        assert (rundir / "cm1out").is_dir() and list(rundir.glob("segments/*/frames"))
        fcntl.flock(held, fcntl.LOCK_UN)
    assert work_in_progress(rundir) == []
    assert client.post("/api/runs/storm/delete", {"keep_pictures": True}).json["ok"]


def test_a_folder_gm1_did_not_make_is_not_deleted(tmp_path) -> None:
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        plain = tmp_path / "runs" / "plain"
        plain.mkdir(parents=True)
        (plain / "events.jsonl").write_text("", encoding="utf-8")
        refused = client.post("/api/runs/plain/delete", {})
        assert refused.status == 409 and "does not delete it" in refused.json["message"] and plain.is_dir()
    finally:
        stop_server(server)


def test_the_delete_actions_are_not_gpuwm_cm1_commands_so_the_api_table_does_not_list_them() -> None:
    from cm1core.gui import commands as C

    assert not {"delete", "open-folder"} & {endpoint.action for endpoint in C.API}
    assert C.endpoint_for(["runs", "x", "delete"]) is None and C.endpoint_for(["open-folder"]) is None


def test_the_runs_page_asks_before_it_deletes_naming_the_run_and_its_size() -> None:
    runs = (STATIC / "js/runs.js").read_text(encoding="utf-8")
    assert "window.confirm(deleteWords(r.name, bytes(r.disk_bytes || 0), keepPictures))" in runs
    assert "`Delete ${name}?\\n\\nIt uses ${size}." in runs and "This cannot be undone." in runs
    assert "`Delete the big files of ${name} and keep its pictures?" in runs
    # Not offered while the run is going, and the size is shown in guided mode too (no advanced-only on it).
    assert 'const going = ["running", "waiting", "stale"].includes(st.state);' in runs and "disabled: going" in runs
    assert 'h("td", { class: "num rundisk" }' in runs and '{ label: mode === "guided" ? "Size" : "Disk", num: true }' in runs
    assert "`${bytes(d.total_bytes || 0)} on disk`" in runs
    # The folder, as File Explorer names it, with Open folder: on Runs and on both Results views.
    folder = (STATIC / "js/folder.js").read_text(encoding="utf-8")
    results = (STATIC / "js/run/results.js").read_text(encoding="utf-8")
    assert 'api.post("/api/open-folder", run ? { run } : {}' in folder and 'button("Open folder"' in folder
    assert "windows: d.root_windows" in runs and results.count("windows: data.folder_windows") == 2


# ---------------------------------------------------------------- 6. the doctor's disk line

def test_under_wsl_the_disk_line_says_what_it_measures() -> None:
    from cm1core.cm1run import gpuinfo
    from cm1core.cm1run.report import scrub
    from cm1core.cm1run.system import disk_row

    def host(free: int, wsl2: bool = True) -> gpuinfo.HostInfo:
        return gpuinfo.HostInfo(disk_root="/x", disk_free_bytes=free, disk_total_bytes=10**12,
                                ram_total_bytes=64 * 10**9, ram_available_bytes=32 * 10**9, wsl2=wsl2)

    plain = disk_row(host(700 * 10**9, wsl2=False))
    assert plain.state == "ok" and "WSL" not in plain.text and "Windows" not in plain.text
    # The Windows drive is known: both figures, and the smaller one is judged.
    drive = {"drive": "C:", "free_bytes": 4 * 10**8, "total_bytes": 10**12}
    row = disk_row(host(700 * 10**9), drive=drive)
    assert "/x (the Linux disk inside WSL):" in row.text and "The Windows drive that holds it (C:) has" in row.text
    assert row.state == "fail" and row.fix
    roomy = disk_row(host(700 * 10**9), drive={**drive, "free_bytes": 400 * 10**9})
    assert roomy.state == "ok" and "count on the smaller of the two." in roomy.text
    assert "can use only" not in roomy.text
    # It is not known: the row says so in plain words.
    unknown = disk_row(host(700 * 10**9))
    assert unknown.state == "ok"
    assert "That is the Linux disk's figure: the Windows drive that holds it may have less free." in unknown.text
    # The runs folder as File Explorer names it; a help report shows no user name in it.
    name = BACK * 2 + BACK.join(("wsl.localhost", "Ubuntu-24.04", "home", "ada", "gm1-runs"))
    named = disk_row(host(700 * 10**9), in_windows=name)
    assert named.text.endswith(f" In File Explorer this folder is {name}") and "the runs folder" not in named.text
    # Each sentence ends before the next begins: the verdict comes before the folder's name, not after it.
    short = disk_row(host(700 * 10**9), drive=drive, in_windows=name)
    assert f"free; count on the smaller of the two. Not enough for even a small run's output. In File Explorer this folder is {name}" in short.text
    assert "gm1-runs Not enough" not in short.text
    # A runs folder on a Windows drive (--root /mnt/d/storms) is not on the Linux disk, and is not called that.
    on_drive = gpuinfo.HostInfo(disk_root="/mnt/d/storms", disk_free_bytes=700 * 10**9, disk_total_bytes=10**12,
                                ram_total_bytes=64 * 10**9, ram_available_bytes=32 * 10**9, wsl2=True)
    mounted = disk_row(on_drive, drive=drive, in_windows="D:" + BACK + "storms")
    assert mounted.state == "ok" and "/mnt/d/storms (on the Windows drive D:):" in mounted.text
    assert "Linux disk" not in mounted.text and "smaller of the two" not in mounted.text
    scrubbed = scrub(named.text, home="/".join(("", "home", "ada")), hosts=[])
    assert "ada" not in scrubbed and BACK.join(("home", "<user>", "gm1-runs")) in scrubbed


# ---------------------------------------------------------------- 7. small things on Watching, and the installer

def test_watching_keeps_its_words_in_step_with_the_run() -> None:
    run = (STATIC / "js/run/run.js").read_text(encoding="utf-8")
    strings = json.loads((ROOT / "src/cm1core/gui/copy/guided.json").read_text(encoding="utf-8"))["strings"]
    # "Starting" only while the badge says Starting.
    assert 'if (s === "running" && /^Starting/.test(status.label || "")) text = t("watch.starting");' in run
    assert "!status.model_time_s && !progress.model_time_s) text" not in run
    # "Stopping at the next step" goes once the run has ended.
    assert 'if (stopAsked && over) { stopAsked = false; notice(""); }' in run
    # The share done comes from the storm time shown beside it, in both views.
    assert run.count("percent: sharePercent(") == 2 and "percent: status.percent" not in run
    # The badge ("Running NN%", in the header and on the chip) is that same share, rounded the same way; the
    # server's own label is never drawn as it came while the run goes.
    assert "export function shownStatus(status, model, total)" in run
    assert "return { ...st, label: `Running ${Math.round(Math.max(0, Math.min(100, Number(share))))}%` };" in run
    assert "const badge = shownStatus(status, model, total);" in run
    assert "chip.replaceChildren(stateChip(badge));" in run
    # The header line: the run, its grid ("67 m grid, 33.8 x 33.8 km") and the same badge.
    assert 'page.setData([id, guidedGridWords(status.grid || runGrid), badge.label || ""].filter(Boolean).join(" · "));' in run
    assert "stateChip(shownStatus(st, model || 0, p.duration_s ?? st.duration_s))" in run
    assert "stateChip(status)" not in run and "stateChip(st)" not in run and "status.label || \"\"}`)" not in run
    # The Stop question says why a stopped storm cannot be run longer, without "no saved point yet".
    question = strings["watch.stop_confirm"]["text"]
    assert "no saved point yet" not in question and "when the storm reaches its end" in question
    assert strings["watch.next_picture_drawing"]["text"] == "The picture for {clock} is being drawn now."
    # The card's own line no longer promises a time; the line under it gives this GPU's.
    preset = json.loads((ROOT / "src/cm1core/presets/supercell-1km/preset.json").read_text(encoding="utf-8"))
    assert "most GPUs" not in preset["line"] and "minutes" not in preset["line"]


def test_the_installer_draws_the_icon_at_every_size_and_names_the_runs_folder() -> None:
    text = (ROOT / "install.ps1").read_text(encoding="utf-8")
    assert "foreach ($size in @(16, 24, 32, 48, 64, 256))" in text
    assert "[System.Drawing.Imaging.ImageFormat]::Png" in text and "$icon.Write([UInt16]$images.Count)" in text
    assert "function Get-RunsFolder([string]$Name)" in text and "'wslpath', '-w', '.'" in text
    assert "In File Explorer that folder is:" in text and "Open folder button" in text
    # An update, with storms already there, is not told the folder "appears with your first storm".
    assert "try { $there = Test-Path -LiteralPath $runs } catch { }" in text
    first, _, rest = text.partition("if ($there) {")
    assert "It appears with your first storm." not in rest.split("} else {", 1)[0]
    assert "It appears with your first storm." in rest.split("} else {", 1)[1].split("}", 1)[0]


# ---------------------------------------------------------------- 8. found by the review of the fixes above

def test_a_second_start_in_a_terminal_that_stays_returns_at_once(monkeypatch, capsys) -> None:
    """Over ssh, on a Linux desktop, or with the browser switched off, the terminal is the person's own: a
    second start prints the open GUI's link and returns.  It held an ssh terminal for 120 s."""

    from cm1core.gui import browser, server as S

    def never(_seconds: float) -> None:
        raise AssertionError("the terminal was held")

    monkeypatch.setattr(S, "_hold_window", never)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    for name in browser.SSH_ENV + (browser.NO_BROWSER_ENV, "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(name, raising=False)
    # ssh into the Windows computer's WSL: no browser is tried, so there is no window of GM1's to keep up.
    monkeypatch.setenv("SSH_CONNECTION", "192.0.2.1 50000 192.0.2.2 22")
    assert S.show_running(URL, wsl=True) == 0
    out = capsys.readouterr().out
    assert "GM1 is already open (the browser was not opened: this is an ssh session)." in out and URL in out
    assert "This window closes" not in out
    # The browser switched off by the environment, under Windows.
    monkeypatch.delenv("SSH_CONNECTION")
    monkeypatch.setenv(browser.NO_BROWSER_ENV, "1")
    assert S.show_running(URL, wsl=True) == 0 and URL in capsys.readouterr().out
    monkeypatch.delenv(browser.NO_BROWSER_ENV)
    # A Linux desktop: the page is opened again, and the terminal is given back at once.
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(browser, "open_link", lambda url: (True, "webbrowser"))
    assert S.show_running(URL, wsl=False) == 0
    assert "Opened it in your browser again (webbrowser)." in capsys.readouterr().out
    # A Linux session with no display.
    monkeypatch.delenv("DISPLAY")
    monkeypatch.undo()
    monkeypatch.setattr(S, "_hold_window", never)
    assert S.show_running(URL, wsl=False) == 0 and URL in capsys.readouterr().out


def test_a_named_pipe_where_the_token_belongs_does_not_hold_the_start(tmp_path) -> None:
    import threading

    from cm1core.gui import auth

    folder = tmp_path / "runs" / ".gm1-gui"
    folder.mkdir(parents=True)
    path = folder / auth.TOKEN_FILE
    os.mkfifo(path, 0o600)
    found: list[str] = []
    reader = threading.Thread(target=lambda: found.append(auth.stored_token(folder)), daemon=True)
    reader.start()
    reader.join(5.0)
    if reader.is_alive():
        # Let the stuck open go before failing, so the thread does not outlive the test.
        os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))
        raise AssertionError("reading the kept token waited on a named pipe")
    # It is replaced by a real file holding a new token, the user's alone.
    assert len(found) == 1 and len(found[0]) == 43
    assert stat.S_ISREG(path.lstat().st_mode) and stat.S_IMODE(path.stat().st_mode) == 0o600
    assert auth.stored_token(folder) == found[0]


def _link_out(rundir: Path, relative: str, elsewhere: Path) -> Path:
    """Move ``rundir/relative`` to ``elsewhere`` and leave a link to it (big files kept on another disk)."""

    folder = rundir / relative
    folder.rename(elsewhere)
    folder.symlink_to(elsewhere, target_is_directory=True)
    return folder


def test_pictures_only_is_refused_whole_when_a_big_folder_is_a_link(finished, tmp_path) -> None:
    """cm1out kept on another disk through a link: the frames and restarts went, the link stayed, the run was
    marked "pictures only" with Continue gone, and the reply was a 500.  Now nothing is deleted."""

    server, client, root = finished
    rundir = root / "storm"
    elsewhere = tmp_path / "other-disk-cm1out"
    _link_out(rundir, "cm1out", elsewhere)
    kept = sorted(path.name for path in elsewhere.iterdir())
    frames = [path for path in rundir.glob("segments/*/frames") if path.is_dir()]
    restarts = [path for path in rundir.glob("segments/*/restarts") if path.is_dir()]
    before = {path: sorted(item.name for item in path.iterdir()) for path in frames + restarts}
    assert kept and frames
    actions = client.get("/api/runs/storm/status").json["actions"]
    refused = client.post("/api/runs/storm/delete", {"keep_pictures": True})
    assert refused.status == 409 and refused.json["ok"] is False
    assert "cm1out in this run is a link to another place" in refused.json["message"]
    assert "nothing was deleted" in refused.json["message"] and str(rundir / "cm1out") in refused.json["fix"]
    # Nothing went: not what the link points to, not the frames, not the restarts; and the run is not marked.
    assert sorted(path.name for path in elsewhere.iterdir()) == kept and (rundir / "cm1out").is_symlink()
    assert {path: sorted(item.name for item in path.iterdir()) for path in frames + restarts} == before
    assert not (rundir / "trimmed.json").exists()
    status = client.get("/api/runs/storm/status").json
    assert not status.get("trimmed") and status["actions"] == actions
    assert client.get("/api/runs/storm/results?mode=guided").json["trimmed"] is None
    # A dry run answers the same way: it would not do it either.
    assert client.post("/api/runs/storm/delete", {"keep_pictures": True, "dry_run": True}).status == 409
    # The whole run can still be deleted; what the link points to is left alone.
    assert client.post("/api/runs/storm/delete", {}).json["ok"] and not rundir.exists()
    assert sorted(path.name for path in elsewhere.iterdir()) == kept


def test_pictures_only_is_refused_when_a_segments_frames_are_a_link(finished, tmp_path) -> None:
    server, client, root = finished
    rundir = root / "storm"
    frames = next(path for path in rundir.glob("segments/*/frames") if path.is_dir())
    elsewhere = tmp_path / "other-disk-frames"
    _link_out(rundir, str(frames.relative_to(rundir)), elsewhere)
    kept = sorted(path.name for path in elsewhere.iterdir())
    refused = client.post("/api/runs/storm/delete", {"keep_pictures": True})
    assert refused.status == 409 and "is a link to another place" in refused.json["message"]
    assert (rundir / "cm1out").is_dir() and any((rundir / "cm1out").iterdir())
    assert sorted(path.name for path in elsewhere.iterdir()) == kept and not (rundir / "trimmed.json").exists()


def test_a_run_with_pictures_only_draws_no_new_look_from_data_it_no_longer_has(finished) -> None:
    server, client, root = finished
    rundir = root / "storm"
    assert client.post("/api/runs/storm/delete", {"keep_pictures": True}).json["ok"]
    index = (rundir / "quicklook" / "index.json").read_bytes()
    pictures = sorted(path.name for path in (rundir / "quicklook").rglob("*.png"))
    for look in ("all", "environment", "parcels"):
        refused = client.post("/api/runs/storm/look", {"look": look})
        assert refused.status == 409, look
        assert "data files were deleted" in refused.json["message"] and "kept" in refused.json["fix"]
    # The two films drawn from the data are refused too; nothing was started, and the pictures are as they were.
    for look in ("swirl", "parcels"):
        refused = client.post("/api/runs/storm/film", {"look": look})
        assert refused.status == 409 and "data files were deleted" in refused.json["message"], look
    assert not server.jobs.launched("storm")
    assert (rundir / "quicklook" / "index.json").read_bytes() == index
    assert sorted(path.name for path in (rundir / "quicklook").rglob("*.png")) == pictures


def test_a_new_token_is_refused_while_gm1_is_open_whatever_the_port(tmp_path, monkeypatch, capsys) -> None:
    from cm1core.cli import main
    from cm1core.gui import auth, server as S

    root = tmp_path / "runs"
    first, _client = start_server(root, tmp_path / "config")
    try:
        assert S.write_running(first) is not None
        kept = auth.stored_token(root / ".gm1-gui")
        monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
        for port in ("0", "8799"):
            assert main(["gui", "--root", str(root), "--config-dir", str(tmp_path / "config"), "--new-token",
                         "--port", port, "--no-browser"]) == 2
            assert "its link cannot change while it is" in capsys.readouterr().err
        # The kept link is the one it was.
        assert auth.stored_token(root / ".gm1-gui") == kept
    finally:
        stop_server(first)


def test_a_new_token_is_kept_only_once_the_gui_has_its_port(tmp_path, monkeypatch, capsys) -> None:
    import socket

    from cm1core.cli import main
    from cm1core.gui import auth

    root = tmp_path / "runs"
    kept = auth.stored_token(root / ".gm1-gui")
    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen(1)
    try:
        monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
        code = main(["gui", "--root", str(root), "--config-dir", str(tmp_path / "config"), "--new-token",
                     "--port", str(taken.getsockname()[1]), "--no-browser", "--fake-runner"])
        assert code == 2 and "in use" in capsys.readouterr().err
    finally:
        taken.close()
    # The start failed: every earlier link still works.
    assert auth.stored_token(root / ".gm1-gui") == kept
    # A start that succeeds prints the new link, and that is the one kept.
    assert _gui(tmp_path, monkeypatch, "--no-browser", "--new-token") == 0
    new = _printed_link(capsys).split("token=")[1]
    assert new != kept and auth.stored_token(root / ".gm1-gui") == new


def test_a_verbose_log_never_shows_the_token(tmp_path, capfd) -> None:
    from cm1core.gui.server import hide_token

    assert hide_token("GET /?token=" + "a" * 43 + " HTTP/1.1") == "GET /?token=... HTTP/1.1"
    assert hide_token("GET /api/runs HTTP/1.1") == "GET /api/runs HTTP/1.1"
    assert hide_token("/x?a=1&token=Ab-_9&b=2") == "/x?a=1&token=...&b=2"
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        server.quiet = False  # gpuwm-cm1 gui --verbose
        assert client.get(f"/?token={server.token}", token=False).status == 303
        assert client.get("/?token=" + "z" * 43, token=False).status == 403
    finally:
        stop_server(server)
    logged = capfd.readouterr().err
    assert "GET /?token=... " in logged and server.token not in logged and "z" * 43 not in logged


def test_on_linux_the_out_of_date_page_names_the_command_not_the_start_menu(tmp_path, monkeypatch) -> None:
    from cm1core.gui import auth

    monkeypatch.setenv("GPUWM_CM1_WSL2", "0")
    sentence = "This link is out of date. Run gpuwm-cm1 gui again and use the page it opens."
    assert auth.out_of_date(False) == sentence
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        page = client.get("/", token=False)
        text = page.raw.decode("utf-8")
        assert page.status == 403 and page.headers["content-type"] == "text/html; charset=utf-8"
        assert "Start menu" not in text and "Run <code>gpuwm-cm1 gui</code> again and use the page it opens." in text
        assert "<script" not in text and server.token not in text
        answer = client.get("/api/session", token=False)
        assert answer.status == 403 and answer.json["message"] == sentence
    finally:
        stop_server(server)
    assert "Start menu" not in auth.out_of_date_page(False) and "Start menu: GM1" in auth.out_of_date_page(True)


def test_under_wsl_a_delete_says_when_the_windows_drive_shows_the_space(finished, monkeypatch) -> None:
    from cm1core.gui import api as A

    server, client, root = finished
    import_supercell(client, "second")
    plain = client.post("/api/runs/second/delete", {}).json
    assert plain["ok"] and plain["note"] == ""
    _windows_place(server)
    done = client.post("/api/runs/storm/delete", {"keep_pictures": True}).json
    assert done["ok"] and done["note"] == A.WSL_FREED_NOTE
    assert "free for new storms now" in done["note"] and "does not shrink by itself" in done["note"]
    runs = (STATIC / "js/runs.js").read_text(encoding="utf-8")
    assert 'freed.${reply.note ? ` ${reply.note}` : ""}`' in runs


def test_a_long_folder_name_wraps_in_its_box() -> None:
    css = (STATIC / "css/app.css").read_text(encoding="utf-8")
    assert ".folderline .copyline code { white-space: normal; overflow-wrap: anywhere; word-break: break-all; }" in css
