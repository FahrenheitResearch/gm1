"""Who may talk to the GUI server.

No token gives 403.  A non-loopback bind is refused without --allow-remote.  A POST without X-GM1-Token gives
403.  A Host other than localhost:PORT/127.0.0.1:PORT, or a mismatched Origin, gives 400.  CPU only.
"""

from __future__ import annotations

import pytest

from cm1core.gui import auth
from gui_support import start_server, stop_server


@pytest.fixture()
def live(tmp_path):
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    yield server, client
    stop_server(server)


# ---------------------------------------------------------------- tokens and cookies

def test_no_token_is_403_everywhere(live):
    server, client = live
    for path in ("/", "/index.html", "/static/js/app.js", "/api/session", "/api/runs", "/api/mode",
                 "/api/runs/x/events", "/nowhere"):
        answer = client.get(path, token=False)
        assert answer.status == 403, (path, answer.status, answer.raw)
        assert server.token.encode() not in answer.raw


def test_wrong_token_is_403(live):
    server, client = live
    assert client.get("/api/session", token=False, headers={"X-GM1-Token": "x" * 43}).status == 403
    assert client.get("/?token=" + "y" * 43, token=False).status == 403
    cookie = f"{auth.cookie_name(server.port)}={'z' * 43}"
    assert client.get("/api/session", token=False, headers={"Cookie": cookie}).status == 403


def test_first_link_sets_an_httponly_cookie_and_drops_the_token_from_the_url(live):
    server, client = live
    answer = client.get(f"/?token={server.token}", token=False)
    assert answer.status == 303
    assert answer.headers["location"] == "/"
    cookie = answer.headers["set-cookie"]
    assert cookie.startswith(f"{auth.cookie_name(server.port)}={server.token};")
    for flag in ("HttpOnly", "SameSite=Strict", "Path=/"):
        assert flag in cookie
    jar = cookie.split(";", 1)[0]
    page = client.get("/", token=False, headers={"Cookie": jar})
    assert page.status == 200 and page.headers["content-type"].startswith("text/html")
    assert b'src="/static/js/app.js"' in page.raw
    session = client.get("/api/session", token=False, headers={"Cookie": jar})
    assert session.status == 200 and session.json["token"] == server.token


def test_the_query_token_opens_only_the_first_page(live):
    server, client = live
    assert client.get(f"/api/session?token={server.token}", token=False).status == 403
    assert client.get(f"/static/js/app.js?token={server.token}", token=False).status == 403


def test_post_needs_the_header_even_with_the_cookie(live):
    server, client = live
    jar = f"{auth.cookie_name(server.port)}={server.token}"
    for path, body in (("/api/mode", {"mode": "advanced"}), ("/api/import", {"name": "a"})):
        no_header = client.post(path, body, token=False, headers={"Cookie": jar})
        assert no_header.status == 403, no_header.raw
        wrong = client.post(path, body, token=False, headers={"Cookie": jar, "X-GM1-Token": "nope"})
        assert wrong.status == 403
    assert client.post("/api/mode", {"mode": "advanced"}).status == 200
    assert (server.api.prefs.load() or {}).get("mode") == "advanced"


def test_a_refused_post_changes_nothing(live, tmp_path):
    server, client = live
    before = server.api.prefs.load()
    assert client.post("/api/mode", {"mode": "advanced"}, token=False).status == 403
    assert server.api.prefs.load() == before
    assert not (tmp_path / "runs" / "evil").exists()
    assert client.post("/api/import", {"name": "evil", "namelist": {"text": "&param1\n/\n"}},
                       token=False).status == 403
    assert not (tmp_path / "runs" / "evil").exists()


def test_cookie_names_differ_by_port():
    assert auth.cookie_name(8765) != auth.cookie_name(8766)


def test_tokens_are_32_random_bytes():
    tokens = {auth.new_token() for _ in range(50)}
    assert len(tokens) == 50
    assert all(len(token) == 43 for token in tokens)


# ---------------------------------------------------------------- Host and Origin

def test_host_must_name_this_server(live):
    server, client = live
    port = server.port
    for good in (f"localhost:{port}", f"127.0.0.1:{port}", f"LOCALHOST:{port}", f"[::1]:{port}"):
        assert client.get("/api/session", host=good).status == 200, good
    for bad in ("evil.example", f"evil.example:{port}", f"localhost:{port + 1}", "localhost",
                "127.0.0.1", f"10.1.1.10:{port}", f"localhost.evil.example:{port}", ""):
        answer = client.get("/api/session", host=bad)
        assert answer.status == 400, (bad, answer.status)
        assert server.token.encode() not in answer.raw
    assert client.get("/api/session", host=None).status == 400      # no Host header at all
    assert client.post("/api/mode", {"mode": "guided"}, host=f"evil.example:{port}").status == 400


def test_origin_must_match_when_present(live):
    server, client = live
    port = server.port
    for good in (f"http://localhost:{port}", f"http://127.0.0.1:{port}"):
        assert client.post("/api/mode", {"mode": "guided"}, headers={"Origin": good}).status == 200
    for bad in ("http://evil.example", f"http://evil.example:{port}", f"http://localhost:{port + 1}", "null",
                "file://", f"http://localhost.evil.example:{port}"):
        answer = client.post("/api/mode", {"mode": "advanced"}, headers={"Origin": bad})
        assert answer.status == 400, (bad, answer.status)
        assert client.get("/api/session", headers={"Origin": bad}).status == 400
    assert server.api.prefs.load().get("mode") == "guided"


def test_duplicate_host_or_token_headers_are_refused(live):
    server, client = live
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    connection.putrequest("GET", "/api/session", skip_host=True)
    connection.putheader("Host", f"127.0.0.1:{server.port}")
    connection.putheader("Host", "evil.example")
    connection.putheader("X-GM1-Token", server.token)
    connection.endheaders()
    assert connection.getresponse().status == 400
    connection.close()


def test_every_answer_carries_the_security_headers(live):
    server, client = live
    answer = client.get("/api/session")
    assert answer.headers["x-content-type-options"] == "nosniff"
    assert answer.headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in answer.headers["content-security-policy"]
    assert answer.headers["cache-control"] == "no-store"
    assert client.get("/api/session", token=False).headers["x-frame-options"] == "DENY"


def test_guard_decisions_in_order():
    guard = auth.Guard("t" * 43, 8765)
    ok = {"Host": "localhost:8765", "X-GM1-Token": "t" * 43}
    assert guard.check("GET", "/api/runs", ok, {}, "127.0.0.1").allowed
    assert guard.check("GET", "/api/runs", {**ok, "Host": "evil:8765"}, {}, "127.0.0.1").status == 400
    assert guard.check("GET", "/api/runs", {**ok, "Origin": "http://evil"}, {}, "127.0.0.1").status == 400
    # A peer on the network is refused even with the token, unless --allow-remote.
    assert guard.check("GET", "/api/runs", ok, {}, "10.20.30.40").status == 403
    assert guard.check("GET", "/api/runs", ok, {}, "::ffff:127.0.0.1").allowed
    assert guard.check("GET", "/api/runs", ok, {}, "::1").allowed
    remote = auth.Guard("t" * 43, 8765, "0.0.0.0", allow_remote=True, extra_hosts=("gpu-box",))
    assert remote.check("GET", "/api/runs", {**ok, "Host": "gpu-box:8765"}, {}, "10.20.30.40").allowed
    assert remote.check("GET", "/api/runs", {**ok, "Host": "evil:8765"}, {}, "10.20.30.40").status == 400
    assert remote.check("POST", "/api/mode", {"Host": "gpu-box:8765"}, {}, "10.20.30.40").status == 403


# ---------------------------------------------------------------- the bind rule

@pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.20.30.40", "10.0.0.1"])
def test_non_loopback_bind_is_refused_without_allow_remote(host):
    with pytest.raises(auth.BindRefused) as refused:
        auth.require_bind(host, allow_remote=False)
    assert "--allow-remote" in refused.value.fix
    auth.require_bind(host, allow_remote=True)      # allowed when asked for; nothing is bound here


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2"])
def test_loopback_binds_are_allowed(host):
    auth.require_bind(host, allow_remote=False)


def test_build_server_refuses_a_remote_bind_before_binding(tmp_path):
    from cm1core.gui.server import build_server

    with pytest.raises(auth.BindRefused):
        build_server(tmp_path / "runs", host="0.0.0.0", port=0)


def test_the_gui_command_refuses_a_remote_bind(tmp_path, capsys):
    from cm1core.cli import main

    rc = main(["gui", "--root", str(tmp_path / "runs"), "--bind", "0.0.0.0", "--port", "0",
               "--config-dir", str(tmp_path / "config")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "other computers can reach" in err and "--allow-remote" in err


# ---------------------------------------------------------------- paths

def test_paths_stay_inside_their_folders(live, tmp_path):
    server, client = live
    (tmp_path / "secret.txt").write_text("secret")
    for path in ("/static/../server.py", "/static/..%2Fserver.py", "/static/%2e%2e/%2e%2e/secret.txt",
                 "/api/runs/..%2F..%2Fsecret.txt", "/api/runs/%2E%2E/files/secret.txt"):
        answer = client.get(path)
        assert answer.status in (400, 404), (path, answer.status)
        assert b"secret" != answer.raw


def test_static_files_are_served_with_their_types(live):
    server, client = live
    js = client.get("/static/js/app.js")
    assert js.status == 200 and js.headers["content-type"].startswith("text/javascript")
    css = client.get("/static/css/gm1.css")
    assert css.status == 200 and css.headers["content-type"].startswith("text/css")
    assert client.request("DELETE", "/api/mode").status == 405


# ---------------------------------------------------------------- edge cases

def test_the_first_link_redirects_even_with_a_cookie_already_set(live):
    server, client = live
    jar = f"{auth.cookie_name(server.port)}={server.token}"
    answer = client.get(f"/?token={server.token}", token=False, headers={"Cookie": jar})
    assert answer.status == 303 and answer.headers["location"] == "/"
    # a wrong token in the link with a good cookie opens the page without setting anything
    wrong = client.get("/?token=" + "w" * 43, token=False, headers={"Cookie": jar})
    assert wrong.status == 200 and "set-cookie" not in wrong.headers


def test_a_negative_content_length_is_refused_at_once(live):
    import socket
    import time

    server, client = live
    for value in ("-1", "nope"):
        with socket.create_connection(("127.0.0.1", server.port), timeout=5) as sock:
            sock.sendall((f"POST /api/mode HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n"
                          f"X-GM1-Token: {server.token}\r\nContent-Type: application/json\r\n"
                          f"Content-Length: {value}\r\n\r\n").encode())
            began = time.monotonic()
            reply = sock.recv(200)
            assert reply.startswith(b"HTTP/1.0 400"), (value, reply)
            assert time.monotonic() - began < 2.0


def test_an_idle_connection_is_closed_after_the_socket_timeout(tmp_path, monkeypatch):
    import socket
    import time

    from cm1core.gui import server as server_module
    from gui_support import start_server, stop_server

    monkeypatch.setattr(server_module.Handler, "timeout", 0.5)
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=10) as sock:
            began = time.monotonic()
            assert sock.recv(100) == b""                    # the server hung up; nothing was sent
            assert time.monotonic() - began < 5.0
        assert client.get("/api/session").status == 200   # and it still serves
    finally:
        stop_server(server)


def test_files_from_a_run_folder_are_data_never_page_code(live, tmp_path):
    """A page or script written into a run folder must not run with the GUI's origin (it would read the token
    from /api/session): only PNG, JSON and plain text are shown inline, the rest is a download, and every run
    file carries a sandbox CSP."""

    from gui_support import import_supercell

    server, client = live
    import_supercell(client, "sc1")
    rundir = tmp_path / "runs" / "sc1"
    (rundir / "page.html").write_text('<script src="/api/runs/sc1/files/code.js"></script>')
    (rundir / "code.js").write_text("fetch('/api/session')")
    (rundir / "picture.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><script>1</script></svg>')
    (rundir / "style.css").write_text("body{}")
    for name in ("page.html", "code.js", "picture.svg", "style.css"):
        answer = client.get(f"/api/runs/sc1/files/{name}")
        assert answer.status == 200, name
        assert answer.headers["content-type"] == "application/octet-stream", name
        assert answer.headers["content-disposition"].startswith("attachment;"), name
        assert answer.headers["content-security-policy"] == "sandbox; default-src 'none'", name
    for name, kind in (("run.json", "application/json"), ("import-report.txt", "text/plain"),
                       ("namelist.input", "text/plain")):
        answer = client.get(f"/api/runs/sc1/files/{name}")
        assert answer.status == 200 and answer.headers["content-type"].startswith(kind), name
        assert "content-disposition" not in answer.headers and "sandbox" in answer.headers["content-security-policy"]
    # the GUI's own pages keep their policy
    assert "script-src 'self'" in client.get("/static/js/app.js").headers["content-security-policy"]


def test_a_download_name_cannot_break_the_headers(live, tmp_path):
    from gui_support import import_supercell

    server, client = live
    import_supercell(client, "sc2")
    odd = 'a"b\r\nSet-Cookie: x=1.bin'
    (tmp_path / "runs" / "sc2" / odd).write_bytes(b"data")
    from urllib.parse import quote

    answer = client.get("/api/runs/sc2/files/" + quote(odd))
    assert answer.status == 200 and answer.raw == b"data"
    assert "x=1" not in answer.headers.get("set-cookie", "")
    assert answer.headers["content-disposition"] == 'attachment; filename="a_b__Set-Cookie__x_1.bin"'
