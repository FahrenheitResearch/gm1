"""Who may talk to the GUI server.

- It binds to a loopback address unless ``--allow-remote`` is given.
- A 32-byte token is printed in the first URL.  ``GET /?token=...`` puts it in
  an HttpOnly, SameSite=Strict cookie and redirects to ``/`` without it.
- The token of a GUI on this computer alone is kept with the runs folder
  (:func:`stored_token`: ``.gm1-gui/token``, readable by its owner only), so the
  link of an earlier start, a bookmark or an open tab still works after GM1 is
  started again.  ``gpuwm-cm1 gui --new-token`` makes a new one.  What that
  changes: a link that leaked stays good until then, where before it died with
  the GUI.  What it does not change: the token is still 32 random bytes, still
  required on every request, still only ever sent to 127.0.0.1, and anyone who
  can read the file could already read ``gui.json`` beside it (the open GUI's
  link) or run programs as the user.  A GUI that other computers can reach
  (``--allow-remote``) never uses the kept token: it makes one per start.
- Every POST carries the token in the ``X-GM1-Token`` header.  The cookie
  alone never authorizes a POST, so a page on another site cannot make the
  browser post on the user's behalf (it cannot set that header).
- The Host header must name this server (``localhost:PORT`` or
  ``127.0.0.1:PORT``), and an Origin header, when present, must too.  This is
  the guard against DNS rebinding.
- A peer that is not a loopback address is refused unless ``--allow-remote``.

Status codes: a wrong Host or Origin is 400; a missing or wrong token, or a
remote peer, is 403.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hmac
import ipaddress
import os
from pathlib import Path
import re
import secrets
import socket
import stat
from typing import Iterable, Mapping
from urllib.parse import urlsplit

TOKEN_BYTES = 32
TOKEN_HEADER = "X-GM1-Token"
LOOPBACK_NAMES = ("localhost", "127.0.0.1", "[::1]")
WILDCARD_BINDS = ("0.0.0.0", "::", "")


class BindRefused(ValueError):
    """The address the GUI was asked to listen on is not allowed."""

    def __init__(self, what: str, fix: str) -> None:
        super().__init__(f"{what} {fix}")
        self.what = what
        self.fix = fix


def new_token() -> str:
    """32 random bytes, URL-safe (43 characters)."""

    return secrets.token_urlsafe(TOKEN_BYTES)


#: The kept token's file, in the runs folder's ``.gm1-gui``.
TOKEN_FILE = "token"
_TOKEN_SHAPE = re.compile(r"^[A-Za-z0-9_-]{43}$")


def _read_kept(path: Path) -> str | None:
    """The token in ``path``, when the file is this user's alone and holds one; else None."""

    # O_NONBLOCK: a named pipe put where the file belongs would otherwise hold the start for ever; opened this
    # way it is seen below not to be a file, and replaced.
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return None
    try:
        found = os.fstat(descriptor)
        if not stat.S_ISREG(found.st_mode):
            return None
        if os.name != "nt":
            # Another user's file, or one others can read or change, is not a secret: it is replaced.
            if found.st_uid != os.geteuid() or stat.S_IMODE(found.st_mode) & 0o077:
                return None
        text = os.read(descriptor, 256).decode("ascii", "replace").strip()
    except OSError:
        return None
    finally:
        os.close(descriptor)
    return text if _TOKEN_SHAPE.match(text) else None


def stored_token(server_dir: Path | str, *, rotate: bool = False) -> str:
    """The runs folder's own token: read from ``server_dir/token``, or made and written there (mode 0600).

    ``rotate`` (``gpuwm-cm1 gui --new-token``) always makes a new one, so every earlier link stops working.
    A file that is not the user's alone, or does not hold a token, is replaced.  When the file cannot be
    written the token still works for this start; the next start makes another.
    """

    path = Path(server_dir) / TOKEN_FILE
    if not rotate:
        kept = _read_kept(path)
        if kept is not None:
            return kept
    token = new_token()
    keep_token(server_dir, token)
    return token


def keep_token(server_dir: Path | str, token: str) -> bool:
    """Write ``token`` as the runs folder's own (``server_dir/token``, mode 0600); whether it was written."""

    path = Path(server_dir) / TOKEN_FILE
    temporary = path.with_name(f".{TOKEN_FILE}.{os.getpid()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.unlink(temporary)
        except OSError:
            pass
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(token + "\n")
        os.replace(temporary, path)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        return False
    return True


def cookie_name(port: int) -> str:
    # Cookies are shared across ports on one host; a name per port keeps two
    # GUI servers on one machine from overwriting each other's token.
    return f"gm1_token_{int(port)}"


def _ip(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        address = ipaddress.ip_address(text.strip("[]").split("%", 1)[0])
    except ValueError:
        return None
    mapped = getattr(address, "ipv4_mapped", None)
    return mapped if mapped is not None else address


def is_loopback(address: str) -> bool:
    ip = _ip(address)
    return bool(ip is not None and ip.is_loopback)


def resolve_bind(host: str) -> list[str]:
    """Every address ``host`` names, as the socket layer would bind it."""

    if host in WILDCARD_BINDS:
        return ["0.0.0.0" if host != "::" else "::"]
    if _ip(host) is not None:
        return [host.strip("[]")]
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as error:
        raise BindRefused(f"The address {host!r} does not resolve ({error}).",
                          "Use --bind 127.0.0.1.") from error
    return sorted({str(info[4][0]) for info in infos})


def require_bind(host: str, allow_remote: bool) -> None:
    """Refuse a non-loopback listening address without ``--allow-remote``."""

    if allow_remote:
        return
    addresses = resolve_bind(host)
    remote = [address for address in addresses if not is_loopback(address)]
    if remote:
        raise BindRefused(
            f"--bind {host} listens on {', '.join(remote)}, which other computers can reach.",
            "Keep the default (127.0.0.1) and use ssh -L from the other computer, "
            "or pass --allow-remote if you mean it.",
        )


def _host_port(value: str) -> str:
    return value.strip().lower().rstrip(".")


def allowed_hosts(port: int, bind: str = "127.0.0.1", allow_remote: bool = False,
                  extra: Iterable[str] = ()) -> frozenset[str]:
    """The Host header values this server answers to, lower case."""

    names = set(LOOPBACK_NAMES)
    if allow_remote:
        if bind not in WILDCARD_BINDS:
            names.add(f"[{bind}]" if ":" in bind and not bind.startswith("[") else bind)
        try:
            names.add(socket.gethostname())
            names.add(socket.getfqdn())
        except OSError:
            pass
    names.update(extra)
    hosts = {f"{_host_port(name)}:{int(port)}" for name in names if name}
    if int(port) == 80:
        hosts.update(_host_port(name) for name in names if name)
    return frozenset(hosts)


def parse_cookies(header: str | None) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for part in (header or "").split(";"):
        name, sep, value = part.strip().partition("=")
        if sep and name:
            cookies[name] = value.strip().strip('"')
    return cookies


@dataclass(frozen=True)
class Decision:
    """What the guard says about one request."""

    status: int = 200                 # 200 = go ahead
    message: str = ""
    via: str = ""                     # header, cookie or query
    set_cookie: bool = False          # the query token was used: set the cookie
    #: Refused for the token alone (none, or not this server's): a browser is shown a page saying so.
    token_refused: bool = False

    @property
    def allowed(self) -> bool:
        return self.status == 200


@dataclass(frozen=True)
class Guard:
    token: str
    port: int
    bind: str = "127.0.0.1"
    allow_remote: bool = False
    extra_hosts: tuple[str, ...] = ()
    hosts: frozenset[str] = field(default=frozenset())
    #: GM1 is opened from the Windows Start menu here (WSL2): the out-of-date words say so.  None: ask the system.
    start_menu: bool | None = None

    def __post_init__(self) -> None:
        if not self.hosts:
            object.__setattr__(self, "hosts", allowed_hosts(
                self.port, self.bind, self.allow_remote, self.extra_hosts))
        if self.start_menu is None:
            try:
                from ..cm1run.gpuinfo import is_wsl2

                found = bool(is_wsl2())
            except Exception:  # noqa: BLE001 - the words fall back to the terminal's
                found = False
            object.__setattr__(self, "start_menu", found)

    # ---- single checks

    def host_ok(self, host: str | None) -> bool:
        return bool(host) and _host_port(host) in self.hosts

    def origin_ok(self, origin: str | None) -> bool:
        if origin is None:
            return True
        parts = urlsplit(origin.strip())
        if parts.scheme not in ("http", "https") or not parts.netloc:
            return False  # includes the literal "null"
        return _host_port(parts.netloc) in self.hosts

    def peer_ok(self, peer: str) -> bool:
        return self.allow_remote or is_loopback(peer)

    def token_ok(self, candidate: str | None) -> bool:
        if not candidate:
            return False
        return hmac.compare_digest(candidate.encode("utf-8", "replace"), self.token.encode("utf-8"))

    def cookie_header(self) -> str:
        return (f"{cookie_name(self.port)}={self.token}; Path=/; HttpOnly; SameSite=Strict")

    # ---- the whole decision

    def check(self, method: str, path: str, headers: Mapping[str, str],
              query: Mapping[str, list[str]], peer: str) -> Decision:
        if not self.host_ok(headers.get("Host")):
            return Decision(400, "This server answers only to "
                                 f"localhost:{self.port} or 127.0.0.1:{self.port}.")
        if not self.origin_ok(headers.get("Origin")):
            return Decision(400, "The request came from another site's page.")
        if not self.peer_ok(peer):
            return Decision(403, "This server takes requests from this computer only. "
                                 "Use ssh -L from the other computer.")
        if self.token_ok(headers.get(TOKEN_HEADER)):
            return Decision(200, via="header")
        if method not in ("GET", "HEAD"):
            return Decision(403, f"A {method} needs the {TOKEN_HEADER} header.")
        # The first link is answered with the cookie and a redirect to a bare "/"
        # even when a cookie is already set, so the token never stays in the
        # address bar or the history.
        if path == "/" and self.token_ok((query.get("token") or [""])[0]):
            return Decision(200, via="query", set_cookie=True)
        if self.token_ok(parse_cookies(headers.get("Cookie")).get(cookie_name(self.port))):
            return Decision(200, via="cookie")
        return Decision(403, out_of_date(bool(self.start_menu)), token_refused=True)


#: What a request without this server's token is told (a link or tab from before a new token was made).  Under
#: Windows GM1 is started from the Start menu; on Linux there is no such thing, and the command is named.
OUT_OF_DATE_HEAD = "This link is out of date."
OUT_OF_DATE_START_MENU = "Start GM1 again (Start menu: GM1) and use the page it opens."
OUT_OF_DATE_TERMINAL = "Run gpuwm-cm1 gui again and use the page it opens."
OUT_OF_DATE = f"{OUT_OF_DATE_HEAD} {OUT_OF_DATE_START_MENU}"
OUT_OF_DATE_FIX = "From a terminal: gpuwm-cm1 gui prints the link and opens it."


def out_of_date(start_menu: bool = True) -> str:
    """The out-of-date sentence for this computer: the Start menu under Windows, the command elsewhere."""

    return f"{OUT_OF_DATE_HEAD} {OUT_OF_DATE_START_MENU if start_menu else OUT_OF_DATE_TERMINAL}"


def out_of_date_page(start_menu: bool = True) -> str:
    """The page a browser gets in place of GM1 when its request has no token, or an old one.

    Plain HTML with its style inline and no script: it must read well with nothing else loaded.
    """

    if start_menu:
        how = (f"<p>{OUT_OF_DATE_START_MENU}</p>\n"
               "<p class=\"sub\">From a terminal: <code>gpuwm-cm1 gui</code> prints the link and opens it. ")
    else:
        how = ("<p>Run <code>gpuwm-cm1 gui</code> again and use the page it opens.</p>\n"
               "<p class=\"sub\">It prints the link, and opens it when this computer has a screen. ")
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<meta name=\"color-scheme\" content=\"dark\">\n<title>GM1: this link is out of date</title>\n"
        "<style>body{margin:0;background:#0f1724;color:#e8edf5;font:16px/1.5 system-ui,Segoe UI,sans-serif}"
        "main{max-width:34em;margin:12vh auto;padding:0 24px}h1{font-size:22px;margin:0 0 12px}"
        "p{margin:0 0 10px}.sub{color:#9fb0c8}code{background:#1b2638;padding:1px 6px;border-radius:4px}"
        "</style>\n</head>\n<body>\n<main>\n<h1>This link is out of date.</h1>\n"
        + how +
        "Your storms are not affected: a running storm keeps going, and GM1 shows it when it opens.</p>\n"
        "</main>\n</body>\n</html>\n")


__all__ = [
    "BindRefused",
    "Decision",
    "Guard",
    "OUT_OF_DATE",
    "OUT_OF_DATE_FIX",
    "TOKEN_FILE",
    "keep_token",
    "out_of_date",
    "out_of_date_page",
    "stored_token",
    "TOKEN_BYTES",
    "TOKEN_HEADER",
    "allowed_hosts",
    "cookie_name",
    "is_loopback",
    "new_token",
    "parse_cookies",
    "require_bind",
    "resolve_bind",
]
