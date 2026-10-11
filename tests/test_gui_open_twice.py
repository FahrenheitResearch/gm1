"""Opening GM1 a second time (the shortcut clicked twice) never ends in a traceback.

1. A port in use: ``build_server`` moves to the next free port, and ``gpuwm-cm1 gui --port P`` on a busy port exits 2
   with "is in use" and no traceback (socketserver closes a server whose bind failed before ``__init__`` ends).
2. A GUI already open on the same runs folder: a second ``gpuwm-cm1 gui`` prints its link (and opens it) instead of
   starting another server; the record it reads is the owner's alone (mode 0600) and goes when the GUI closes.

CPU only, the fake runner, loopback ports.
"""

from __future__ import annotations

import os
from pathlib import Path
import socket
import stat
import sys

import pytest

from gui_support import start_server, stop_server


def _listening() -> tuple[socket.socket, int]:
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    return holder, holder.getsockname()[1]


def test_a_busy_port_moves_the_gui_to_the_next_one(tmp_path: Path) -> None:
    from cm1core.gui.jobs import fake_runner
    from cm1core.gui.prefs import Prefs
    from cm1core.gui.server import build_server

    holder, port = _listening()
    try:
        server = build_server(tmp_path / "runs", port=port, port_tries=10, runner=fake_runner(),
                              prefs=Prefs(tmp_path / "config"))
        try:
            assert port < server.port <= port + 9
        finally:
            server.server_close()
        # One try only: the port's own error, not an AttributeError from a half-built server.
        with pytest.raises(OSError) as error:
            build_server(tmp_path / "runs", port=port, port_tries=1, runner=fake_runner(),
                         prefs=Prefs(tmp_path / "config"))
        assert not isinstance(error.value, AttributeError)
    finally:
        holder.close()


def test_gui_on_a_busy_port_says_so_without_a_traceback(tmp_path: Path, monkeypatch, capsys) -> None:
    from cm1core.cli import main

    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    holder, port = _listening()
    try:
        code = main(["gui", "--root", str(tmp_path / "runs"), "--port", str(port), "--no-browser", "--fake-runner",
                     "--config-dir", str(tmp_path / "config")])
    finally:
        holder.close()
    out = capsys.readouterr()
    assert code == 2
    assert f"port {port} is in use" in out.err and "--port" in out.err
    assert "Traceback" not in out.err + out.out and "AttributeError" not in out.err


def test_a_second_gui_on_the_same_runs_folder_shows_the_first(tmp_path: Path, monkeypatch, capsys) -> None:
    from cm1core.cli import main
    from cm1core.gui import browser, server as S

    root = tmp_path / "runs"
    first, _client = start_server(root, tmp_path / "config")
    try:
        record = S.write_running(first)
        assert record is not None and record.is_file()
        if os.name != "nt":
            assert stat.S_IMODE(record.stat().st_mode) == 0o600
        found = S.running_gui(root)
        assert found is not None and found["url"] == first.url and found["port"] == first.port
        # Another runs folder is another GUI.
        assert S.running_gui(tmp_path / "other") is None

        opened: list[str] = []
        monkeypatch.setattr(browser, "open_link", lambda url: (opened.append(url) or True, "test"))

        def never(self, poll_interval=0.5):  # noqa: ANN001 - serve_forever's shape
            raise AssertionError("a second server was started")

        monkeypatch.setattr(S.GuiServer, "serve_forever", never)
        monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
        assert main(["gui", "--root", str(root), "--config-dir", str(tmp_path / "config")]) == 0
        out = capsys.readouterr().out
        assert "GM1 is already open" in out and first.url in out and opened == [first.url]
        # The GUI that wrote the record removes it when it closes; another process's record is left alone.
        S.forget_running(record)
        assert not record.is_file()
        record.write_text('{"pid": 1, "port": 1, "url": "x"}', encoding="utf-8")
        S.forget_running(record)
        assert record.is_file()
    finally:
        stop_server(first)
    # The GUI is gone: its record no longer answers, so a new one would start.
    assert S.running_gui(root) is None


def test_a_stale_record_is_ignored(tmp_path: Path) -> None:
    from cm1core.gui import server as S
    from cm1core.gui.files import SERVER_DIR

    folder = tmp_path / "runs" / SERVER_DIR
    folder.mkdir(parents=True)
    holder, port = _listening()
    holder.close()
    for record in ('{"pid": 999999999, "port": %d, "url": "http://127.0.0.1:%d/?token=x"}' % (port, port),
                   "not json", '{"pid": "x"}'):
        (folder / S.RUNNING_FILE).write_text(record, encoding="utf-8")
        assert S.running_gui(tmp_path / "runs") is None
