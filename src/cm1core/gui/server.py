"""``gpuwm-cm1 gui``: the local web GUI's HTTP server.

A stdlib ``ThreadingHTTPServer``.  Every request passes :class:`.auth.Guard`
first (Host, Origin, loopback peer, token).  ``/`` and ``/static/...`` serve
``static/``; ``/api/...`` is :class:`.api.Api`; ``/api/runs/RUN/events`` is a
Server-Sent Events stream of the run's event log.

The server process never imports CuPy: a meta-path guard makes any attempt
an ImportError, so a host-side import that reaches for the GPU fails loudly
here instead of opening a CUDA context in the GUI.
"""

from __future__ import annotations

import argparse
import errno
import http.server
import json
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import auth
from .api import Api, Reply, parse_event_id
from .files import SERVER_DIR, PathRefused, Root, content_type, download_name, is_inline, run_file_type, static_file
from .jobs import Jobs, Runner, fake_runner, mixed_runner, real_runner
from .prefs import MODES, Prefs

DEFAULT_PORT = 8765
PORT_TRIES = 10
BODY_LIMIT = 24 * 1024 * 1024
SSE_LIMIT = 16
FILE_CHUNK = 1024 * 1024
SOCKET_TIMEOUT_S = 120.0
BLOCKED_MODULES = ("cupy", "cupyx", "cupy_backends")
DEFAULT_ROOT = Path("~/gm1-runs")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; font-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; "
        "object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
}


class _NoGpuImports:
    """A meta-path finder that refuses CuPy inside the GUI server."""

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name.split(".", 1)[0] in BLOCKED_MODULES:
            raise ImportError(
                f"the GUI server never imports {name}: GPU work runs in the run's own process")
        return None


def block_gpu_imports() -> None:
    if not any(isinstance(finder, _NoGpuImports) for finder in sys.meta_path):
        sys.meta_path.insert(0, _NoGpuImports())


def _version() -> str:
    try:
        from .._version import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


class GuiServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, address: tuple[str, int], *, root: Root, runner: Runner, prefs: Prefs,
                 token: str | None = None, allow_remote: bool = False,
                 extra_hosts: tuple[str, ...] = (), quiet: bool = True, native: dict | None = None) -> None:
        host = address[0]
        self.address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
        # Set before binding: socketserver calls server_close() when the bind fails (a port in use), and that
        # must find what it closes, so build_server sees the OSError and tries the next port.
        self.stopping = threading.Event()
        super().__init__(address, Handler)
        self.port = int(self.server_address[1])
        self.bind_host = host
        self.token = token or auth.new_token()
        self.guard = auth.Guard(self.token, self.port, host, allow_remote, tuple(extra_hosts))
        self.root = root
        self.quiet = quiet
        self.jobs = Jobs(runner, root.server_dir)
        self.api = Api(root, self.jobs, prefs, token=self.token, port=self.port,
                       version=_version(), bind=host, native=native)
        self.streams = threading.BoundedSemaphore(SSE_LIMIT)

    @property
    def url(self) -> str:
        host = self.bind_host if self.bind_host not in auth.WILDCARD_BINDS else "127.0.0.1"
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}/?token={self.token}"

    def server_close(self) -> None:
        # Also called on a half-built server (the bind failed): close only what was made.
        stopping = getattr(self, "stopping", None)
        if stopping is not None:
            stopping.set()
        api = getattr(self, "api", None)
        if api is not None:
            api.close()
        jobs = getattr(self, "jobs", None)
        if jobs is not None:
            jobs.close()
        super().server_close()


class Handler(http.server.BaseHTTPRequestHandler):
    server: GuiServer
    server_version = "gm1-gui"
    sys_version = ""
    protocol_version = "HTTP/1.0"
    # Socket timeout: a connection that sends no request, or stops reading an
    # event stream, frees its thread after this long.  The event stream writes
    # at least every KEEPALIVE_S (15 s), so a live one never meets it.
    timeout = SOCKET_TIMEOUT_S

    # ---- plumbing

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        if not self.server.quiet:
            super().log_message(format, *args)

    def _headers(self, status: int, kind: str, length: int | None, extra: dict[str, str] | None = None,
                 cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        # extra may tighten a security header (a run file's CSP); each is sent once.
        for key, value in {**SECURITY_HEADERS, **(extra or {})}.items():
            self.send_header(key, value)
        self.end_headers()

    def _json(self, status: int, body: Any, extra: dict[str, str] | None = None) -> None:
        data = (json.dumps(body, indent=None, sort_keys=True, allow_nan=False, default=str) + "\n").encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(data), extra)
        if self.command != "HEAD":
            self.wfile.write(data)

    def _text(self, status: int, message: str) -> None:
        data = (message + "\n").encode("utf-8")
        self._headers(status, "text/plain; charset=utf-8", len(data))
        if self.command != "HEAD":
            self.wfile.write(data)

    def _file(self, path: Path, cache: str = "no-store", run_file: bool = False) -> None:
        extra: dict[str, str] = {}
        if run_file:
            kind, inline = run_file_type(path)
            extra["Content-Security-Policy"] = "sandbox; default-src 'none'"
        else:
            kind = content_type(path)
            inline = is_inline(kind)
        size = path.stat().st_size
        if not inline:
            extra["Content-Disposition"] = f'attachment; filename="{download_name(path.name)}"'
        self._headers(200, kind, size, extra, cache)
        if self.command == "HEAD":
            return
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(FILE_CHUNK)
                if not chunk:
                    break
                self.wfile.write(chunk)

    # ---- methods

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        self._dispatch("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("HEAD")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._text(405, "Only GET and POST are used here.")

    do_DELETE = do_PATCH = do_OPTIONS = do_PUT  # noqa: N815

    def _dispatch(self, method: str) -> None:
        try:
            self._route(method)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:  # noqa: BLE001 - one bad request never takes the server down
            try:
                self._json(500, {"ok": False, "message": f"The server hit an error: {type(error).__name__}: {error}",
                                 "fix": "Try again; if it repeats, run gpuwm-cm1 gui --verbose and send the output."})
            except OSError:
                pass

    def _route(self, method: str) -> None:
        parts = urlsplit(self.path)
        raw_path = parts.path or "/"
        query = parse_qs(parts.query, keep_blank_values=True)
        # A second Host, Origin or token header would let one check read a
        # value another does not; such a request is refused outright.
        for name in ("Host", "Origin", auth.TOKEN_HEADER, "Cookie"):
            if len(self.headers.get_all(name) or []) > 1:
                self._drain_body()
                self._json(400, {"ok": False, "message": f"The request has more than one {name} header.",
                                 "fix": ""})
                return
        view = _Caseless({key.lower(): value for key, value in self.headers.items()})
        decision = self.server.guard.check(method, raw_path, view, query, self.client_address[0])
        if not decision.allowed:
            self._drain_body()
            self._json(decision.status, {"ok": False, "message": decision.message,
                                         "fix": "Open the link gpuwm-cm1 gui printed."
                                         if decision.status == 403 else ""})
            return
        if decision.set_cookie:
            self._headers(303, "text/plain; charset=utf-8", 0,
                          {"Location": "/", "Set-Cookie": self.server.guard.cookie_header()})
            return
        if raw_path == "/" or raw_path == "/index.html":
            self._static("index.html")
            return
        if raw_path.startswith("/static/"):
            self._static(raw_path[len("/static/"):])
            return
        if raw_path == "/favicon.ico":
            self._text(404, "No icon.")
            return
        if raw_path.startswith("/api/"):
            self._api(method, raw_path, query, view)
            return
        self._json(404, {"ok": False, "message": "No such page.", "fix": "Go to /."})

    def _drain_body(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return
        if 0 < length <= BODY_LIMIT:
            self.rfile.read(length)

    def _static(self, relative: str) -> None:
        try:
            path = static_file(relative)
        except (PathRefused, FileNotFoundError):
            self._json(404, {"ok": False, "message": "No such file.", "fix": ""})
            return
        self._file(path, cache="no-cache")

    def _api(self, method: str, raw_path: str, query: dict[str, list[str]], headers: "_Caseless") -> None:
        body = b""
        if method == "POST":
            try:
                length = int(headers.get("content-length") or 0)
            except ValueError:
                length = -1
            if length < 0:
                # rfile.read(-1) would wait for the client to close the connection.
                self._json(400, {"ok": False, "message": "Bad Content-Length.", "fix": ""})
                return
            if length > BODY_LIMIT:
                self._json(413, {"ok": False, "message": "The request is too large.",
                                 "fix": "Give a file path instead of uploading a large file."})
                return
            body = self.rfile.read(length) if length else b""
        segments = [part for part in raw_path.split("/") if part]
        if method in ("GET", "HEAD") and len(segments) == 4 and segments[:2] == ["api", "runs"] \
                and segments[3] == "events":
            self._events(segments[2], headers, query)
            return
        reply: Reply = self.server.api.handle(method, raw_path, query, body,
                                              headers.get("content-type") or "")
        if reply.file is not None:
            self._file(reply.file, run_file=reply.run_file)
            return
        self._json(reply.status, reply.body, reply.headers)

    # ---- Server-Sent Events

    def _events(self, encoded_run: str, headers: "_Caseless", query: dict[str, list[str]]) -> None:
        from urllib.parse import unquote

        run_id = unquote(encoded_run)
        try:
            rundir = self.server.root.existing_run(run_id)
        except (PathRefused, FileNotFoundError) as error:
            self._json(404, {"ok": False, "message": str(error), "fix": ""})
            return
        if not self.server.streams.acquire(blocking=False):
            self._json(503, {"ok": False, "message": "Too many live views are open.",
                             "fix": "Close a few GM1 tabs and reload."}, {"Retry-After": "5"})
            return
        try:
            since = parse_event_id(headers.get("last-event-id") or (query.get("since") or [""])[0])
            self._headers(200, "text/event-stream; charset=utf-8", None,
                          {"X-Accel-Buffering": "no", "Connection": "close"})
            if self.command == "HEAD":
                return
            self.wfile.write(b"retry: 2000\n\n")
            self.wfile.flush()
            client_gone = threading.Event()

            def gone() -> bool:
                return client_gone.is_set() or self.server.stopping.is_set()

            stop = _EitherEvent(client_gone, self.server.stopping)
            for event, event_id, data in self.server.api.follow(run_id, rundir, since, stop):
                if event == "keepalive":
                    chunk = b": keepalive\n\n"
                else:
                    lines = []
                    if event_id is not None:
                        lines.append(f"id: {event_id}")
                    lines.append(f"event: {event}")
                    lines.append("data: " + json.dumps(data, sort_keys=True, allow_nan=False, default=str))
                    chunk = ("\n".join(lines) + "\n\n").encode("utf-8")
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except OSError:
                    client_gone.set()
                    break
                if gone():
                    break
        finally:
            self.server.streams.release()


class _Caseless:
    """A read-only header mapping looked up case-insensitively."""

    def __init__(self, lowered: dict[str, str]) -> None:
        self._data = lowered

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key.lower(), default)


class _EitherEvent:
    """Set when either of two events is; waits on both (for the SSE loop)."""

    def __init__(self, first: threading.Event, second: threading.Event) -> None:
        self.first, self.second = first, second

    def is_set(self) -> bool:
        return self.first.is_set() or self.second.is_set()

    def wait(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not self.is_set():
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            self.first.wait(min(left, 0.05))
        return True


# ---------------------------------------------------------------- building

def build_server(root: Path | str, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 allow_remote: bool = False, extra_hosts: tuple[str, ...] = (),
                 runner: Runner | None = None, prefs: Prefs | None = None, token: str | None = None,
                 quiet: bool = True, port_tries: int = 1, block_gpu: bool = False,
                 native: dict | None = None) -> GuiServer:
    """Bind a GUI server (not yet serving).  Refuses a remote bind without allow_remote.

    ``block_gpu``: make any CuPy import in this process an ImportError.  The
    ``gui`` command sets it, since its process is the server alone; a test
    that builds a server inside a larger process leaves it off.
    """

    auth.require_bind(host, allow_remote)
    if block_gpu:
        block_gpu_imports()
    root_path = Path(root).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    runs = Root(root_path)
    last_error: OSError | None = None
    for attempt in range(max(1, port_tries)):
        try:
            return GuiServer((host, port + attempt if port else 0), root=runs,
                             runner=runner or real_runner(), prefs=prefs or Prefs(),
                             token=token, allow_remote=allow_remote,
                             extra_hosts=extra_hosts, quiet=quiet, native=native)
        except OSError as error:
            if error.errno != errno.EADDRINUSE or not port:
                raise
            last_error = error
    assert last_error is not None
    raise last_error


def serve_in_thread(server: GuiServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2},
                              name="gm1-gui-server", daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------- the command

def add_command(commands: "argparse._SubParsersAction") -> None:
    command = commands.add_parser(
        "gui",
        help="open GM1 in your browser (start here): the local web GUI at http://127.0.0.1:8765/?token=...",
    )
    command.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                         help="the folder that holds your runs (default: ~/gm1-runs)")
    command.add_argument("--port", type=int, default=None,
                         help=f"port to listen on (default {DEFAULT_PORT}, or the next free one); 0 = any free port")
    command.add_argument("--bind", default="127.0.0.1",
                         help="address to listen on (default 127.0.0.1, this computer only)")
    command.add_argument("--allow-remote", action="store_true",
                         help="allow a --bind address other computers can reach, and requests from them")
    command.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                         help="with --allow-remote: another host name the browser uses to reach this server")
    command.add_argument("--mode", choices=MODES,
                         help="open in this mode and remember it (guided: presets and plain words; "
                              "advanced: every setting)")
    command.add_argument("--native-dir", type=Path, metavar="DIR",
                         help="source checkouts only: the folder holding the built libcm1_state.so and "
                              "cm1_forecast_publish (cargo build --release); Start, Continue, Measure speed and the "
                              "job script pass them to gpuwm-cm1 (an installed wheel has its own)")
    command.add_argument("--no-browser", action="store_true",
                         help="only print the link; by default it also opens in your browser when this computer "
                              "has a screen (never over ssh)")
    command.add_argument("--verbose", action="store_true", help="log every request on stderr")
    command.add_argument("--fake-runner", action="store_true", help=argparse.SUPPRESS)
    # Real host-only commands, fake GPU work (screen checks on a machine whose GPU is busy or absent).
    command.add_argument("--fake-gpu", action="store_true", help=argparse.SUPPRESS)
    command.add_argument("--config-dir", type=Path, help=argparse.SUPPRESS)
    command.set_defaults(handler=gui_door)


NATIVE_FILES = {"library": "libcm1_state.so", "publisher": "cm1_forecast_publish"}


def native_tools(directory: Path | None) -> dict | None:
    """``--native-dir``'s two files, or ValueError naming the one missing."""

    if directory is None:
        return None
    directory = Path(directory).expanduser().resolve()
    found = {}
    for role, name in NATIVE_FILES.items():
        path = directory / name
        if not path.is_file():
            raise ValueError(f"{path} is not there; build it with: cargo build --release --locked "
                             "--manifest-path rust/cm1-state/Cargo.toml")
        found[role] = str(path)
    return found


def gui_door(arguments: argparse.Namespace) -> int:
    try:
        native = native_tools(arguments.native_dir)
    except ValueError as error:
        print(f"gpuwm-cm1 gui: {error}", file=sys.stderr)
        return 2
    prefs = Prefs(arguments.config_dir)
    if arguments.mode:
        prefs.set_mode(arguments.mode)
    port = DEFAULT_PORT if arguments.port is None else int(arguments.port)
    # Opened a second time (the GM1 shortcut clicked twice): the GUI already open on this runs folder is
    # shown again instead of a second server on the next port.
    fake = arguments.fake_runner or arguments.fake_gpu
    if arguments.port is None and arguments.bind in LOOPBACK_BINDS and not fake:
        running = running_gui(Path(arguments.root).expanduser())
        if running is not None:
            print("GM1 is already open. Its link:", f"  {running['url']}", sep="\n", flush=True)
            if not arguments.no_browser:
                from .browser import open_link

                opened, how = open_link(running["url"])
                if opened:
                    print(f"Opened it in your browser ({how}).", flush=True)
            return 0
    try:
        server = build_server(
            arguments.root, host=arguments.bind, port=port, allow_remote=arguments.allow_remote,
            extra_hosts=tuple(arguments.allow_host),
            runner=(fake_runner() if arguments.fake_runner else mixed_runner() if arguments.fake_gpu
                    else real_runner()),
            prefs=prefs, quiet=not arguments.verbose,
            port_tries=PORT_TRIES if arguments.port is None else 1, block_gpu=True, native=native)
    except auth.BindRefused as error:
        print(f"gpuwm-cm1 gui: {error.what}\n{error.fix}", file=sys.stderr)
        return 2
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            tried = f"ports {port} to {port + PORT_TRIES - 1} are" if arguments.port is None else f"port {port} is"
            print(f"gpuwm-cm1 gui: {tried} in use.\nIf GM1 is already open, use its window and link. Otherwise "
                  "pass --port with another number, for example: gpuwm-cm1 gui --port 8800", file=sys.stderr)
            return 2
        raise
    from ..cm1run.gpuinfo import is_wsl2

    wsl = is_wsl2()
    mode, _ = prefs.mode(server.root.has_runs())
    lines = [
        "GM1 is ready. Open this link in your browser:",
        f"  {server.url}",
        f"Runs folder: {server.root.path}",
        f"Mode: {mode} (switch at the top of the page)",
    ]
    if arguments.fake_runner:
        lines.append("Fake runner: nothing here runs a model; every result is made up for testing.")
    elif arguments.fake_gpu:
        lines.append("Fake GPU: checks and estimates are real; Start and Measure speed make up their results.")
    else:
        blocker = gpu_blocker()
        if blocker:
            lines.append(f"GM1 opened, but storms cannot run yet: {blocker}")
    # Under WSL2 the GUI is for the Windows computer it runs in; an ssh hint would only confuse.
    if server.bind_host in LOOPBACK_BINDS and not wsl:
        lines.append(f"From another computer: ssh -L {server.port}:localhost:{server.port} "
                     f"{socket.gethostname()}, then open the link there.")
    if wsl:
        # A run started here keeps Ubuntu up until it ends (cm1run.wsl_keepalive): only Windows going down ends it.
        lines.append("Closing this window closes the GUI, never a storm: a running storm keeps going, and GM1 "
                     "shows it when opened again. Shutting down or restarting Windows stops a storm. "
                     "Press Ctrl+C to close the GUI.")
    else:
        lines.append("Closing this stops the GUI, never a run. Press Ctrl+C to close it.")
    print("\n".join(lines), flush=True)
    record = write_running(server) if server.bind_host in LOOPBACK_BINDS and not arguments.fake_runner else None
    if not arguments.no_browser and server.bind_host in LOOPBACK_BINDS:
        from .browser import open_link

        opened, how = open_link(server.url)
        if opened:
            print(f"Opened the link in your browser ({how}; gpuwm-cm1 gui --no-browser only prints it).",
                  flush=True)
        else:
            print(f"The browser was not opened ({how}): open the link above yourself.", flush=True)
        if record is not None:
            # doctor --report says how the link was last opened (a "the browser did not open" report).
            try:
                (server.root.server_dir / "browser.json").write_text(
                    json.dumps({"opened": bool(opened), "how": str(how)}) + "\n", encoding="utf-8")
            except OSError:
                pass
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nClosed the GUI. Runs keep going; gpuwm-cm1 gui opens it again.", flush=True)
    finally:
        if record is not None:
            forget_running(record)
        server.server_close()
    return 0


# ---------------------------------------------------------------- one GUI per runs folder

LOOPBACK_BINDS = ("127.0.0.1", "localhost", "::1")
#: The open GUI of a runs folder: its link (with the token), port and process; readable by its owner only.
RUNNING_FILE = "gui.json"


def write_running(server: GuiServer) -> Path | None:
    """Record the open GUI's link in the runs folder (mode 0600), so a second ``gpuwm-cm1 gui`` shows it."""

    import os

    path = server.root.server_dir / RUNNING_FILE
    record = {"pid": os.getpid(), "port": server.port, "url": server.url, "root": str(server.root.path)}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{RUNNING_FILE}.{os.getpid()}")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream)
        os.replace(temporary, path)
    except OSError:
        return None
    return path


def forget_running(path: Path) -> None:
    import os

    try:
        if json.loads(path.read_text(encoding="utf-8")).get("pid") == os.getpid():
            path.unlink()
    except (OSError, ValueError, AttributeError):
        pass


def running_gui(root: Path, timeout_s: float = 2.0) -> dict[str, Any] | None:
    """The GUI already open on ``root`` and answering its link, or None."""

    import os
    import urllib.request

    try:
        record = json.loads((Path(root) / SERVER_DIR / RUNNING_FILE).read_text(encoding="utf-8"))
        pid, port, url = int(record["pid"]), int(record["port"]), str(record["url"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if not url.startswith(f"http://127.0.0.1:{port}/?token="):
        return None
    try:
        os.kill(pid, 0)
    except PermissionError:
        pass
    except OSError:
        return None
    token = url.split("token=", 1)[1]
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/session", headers={auth.TOKEN_HEADER: token})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as reply:  # noqa: S310 - loopback only
            session = json.loads(reply.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    try:
        same = Path(str(session.get("root"))).resolve() == Path(root).resolve()
    except (OSError, RuntimeError):
        same = False
    return {"url": url, "port": port, "pid": pid} if same else None


def gpu_blocker() -> str:
    """The doctor's fix line when no NVIDIA card can run a storm here (no driver, too old, no GPU), else ""."""

    try:
        from ..cm1run import gpuinfo
        from ..cm1run.system import driver_row, libcuda_row

        query = gpuinfo.query_gpus()
        row = driver_row(query)
        if row.state == "ok":
            row = libcuda_row(query) or row
    except Exception:  # noqa: BLE001 - the startup lines never fail on the check
        return ""
    if row.state != "fail":
        return ""
    return f"{row.text} Fix: {row.fix}".strip()


__all__ = [
    "DEFAULT_PORT",
    "GuiServer",
    "Handler",
    "add_command",
    "block_gpu_imports",
    "build_server",
    "gui_door",
    "running_gui",
    "serve_in_thread",
]
