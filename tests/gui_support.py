"""Shared helpers for the GUI server tests: a live server on 127.0.0.1 with the fake runner, an HTTP client
that can send any Host/Origin/token, and a Server-Sent Events reader.  No browser, no GPU."""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
from pathlib import Path
import time
from typing import Any, Callable
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
SUPERCELL = ROOT / "tests" / "data" / "cm1import" / "r21.1-supercell.namelist.input"


@dataclass
class Answer:
    status: int
    headers: dict[str, str]
    raw: bytes

    @property
    def json(self) -> Any:
        return json.loads(self.raw.decode("utf-8"))


class Client:
    """Talks to one server; by default as the page would (right Host, token header)."""

    def __init__(self, port: int, token: str) -> None:
        self.port = port
        self.token = token

    def request(self, method: str, path: str, body: Any = None, *, headers: dict[str, str] | None = None,
                token: bool = True, host: str | None = "default", raw_body: bytes | None = None,
                timeout: float = 60.0) -> Answer:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        sent: dict[str, str] = {}
        if host == "default":
            sent["Host"] = f"127.0.0.1:{self.port}"
        elif host is not None:
            sent["Host"] = host
        if token:
            sent["X-GM1-Token"] = self.token
        data = raw_body
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            sent["Content-Type"] = "application/json"
        sent.update(headers or {})
        connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for key, value in sent.items():
            connection.putheader(key, value)
        if data is not None:
            connection.putheader("Content-Length", str(len(data)))
        connection.endheaders(data)
        response = connection.getresponse()
        answer = Answer(response.status, {k.lower(): v for k, v in response.getheaders()}, response.read())
        connection.close()
        return answer

    def get(self, path: str, **kwargs: Any) -> Answer:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, body: Any = None, **kwargs: Any) -> Answer:
        return self.request("POST", path, {} if body is None else body, **kwargs)

    def events(self, run_id: str, until: Callable[[dict[str, Any]], bool], *, timeout: float = 60.0,
               last_event_id: str | None = None, cookie: str | None = None) -> list[dict[str, Any]]:
        """Read the run's SSE stream until ``until(event)`` is true; every event read, in order."""

        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        headers = {"Host": f"localhost:{self.port}", "Accept": "text/event-stream"}
        if cookie is not None:
            headers["Cookie"] = cookie
        else:
            headers["X-GM1-Token"] = self.token
        if last_event_id is not None:
            headers["Last-Event-ID"] = last_event_id
        connection.request("GET", f"/api/runs/{quote(run_id, safe='')}/events", headers=headers)
        response = connection.getresponse()
        assert response.status == 200, response.read()
        assert response.getheader("Content-Type").startswith("text/event-stream")
        events: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                line = response.fp.readline()
                if not line:
                    break
                text = line.decode("utf-8").rstrip("\n")
                if text == "":
                    if "event" in current or "data" in current:
                        events.append(current)
                        if until(current):
                            return events
                    current = {}
                    continue
                if text.startswith(":"):
                    current.setdefault("comments", []).append(text)
                    continue
                name, _, value = text.partition(":")
                value = value[1:] if value.startswith(" ") else value
                if name == "data":
                    current["data"] = json.loads(value)
                elif name in ("event", "id", "retry"):
                    current[name] = value
        finally:
            connection.close()
        raise AssertionError(f"the stream did not reach the awaited event; read {[e.get('event') for e in events]}")


def wait_for(predicate: Callable[[], Any], timeout: float = 30.0, every: float = 0.05) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(every)
    raise AssertionError("timed out waiting")


def start_server(root: Path, config: Path, **kwargs: Any):
    from cm1core.gui.jobs import fake_runner
    from cm1core.gui.prefs import Prefs
    from cm1core.gui.server import build_server, serve_in_thread

    kwargs.setdefault("runner", fake_runner())
    server = build_server(root, port=0, prefs=Prefs(config), **kwargs)
    serve_in_thread(server)
    return server, Client(server.port, server.token)


def stop_server(server) -> None:
    """Close the server, then end what the test launched (the server itself never stops a run)."""

    import os
    import signal

    server.shutdown()
    server.server_close()
    for row in server.jobs.launched():
        try:
            os.killpg(row.process.pid, signal.SIGKILL)
        except OSError:
            pass
        row.process.wait(timeout=10)


def import_supercell(client: Client, name: str = "sc1", **extra: Any) -> dict[str, Any]:
    body = {"name": name, "namelist": {"text": SUPERCELL.read_text(encoding="utf-8")}, **extra}
    answer = client.post("/api/import", body)
    assert answer.status == 200, answer.raw
    return answer.json
