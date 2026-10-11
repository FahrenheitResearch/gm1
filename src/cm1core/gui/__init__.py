"""The local web GUI: ``gpuwm-cm1 gui``.

A stdlib ``ThreadingHTTPServer`` bound to 127.0.0.1 that serves static pages
(``static/``) and JSON endpoints, and streams a run's ``gm1.run-events/v1``
log as Server-Sent Events.  It never imports CuPy and never touches a GPU:
every action is one ``gpuwm-cm1`` command, run as a host-only subprocess
(import, check, cartesian --plan, estimate) or launched detached in its own
session (run, resume, and the cm1out/quicklook sidecars).  The run directory
is the contract, so closing the GUI never stops a run, and a run started
from a terminal or a batch job looks the same here.

:mod:`.auth`       token, Host, Origin and loopback checks; the bind rule
:mod:`.files`      paths confined to ``--root``; run discovery; static files
:mod:`.runstate`   a run's status from its run directory and event log
:mod:`.commands`   the API table: every action and the command it runs
:mod:`.jobs`       running and launching commands; ``commands.log``
:mod:`.prefs`      the guided/advanced mode, stored per OS user
:mod:`.control`    ``gpuwm-cm1 stop`` and ``gpuwm-cm1 pause``
:mod:`.api`        the JSON endpoints
:mod:`.server`     the HTTP server, SSE, and ``gpuwm-cm1 gui``
:mod:`.fake_runner` a stand-in ``gpuwm-cm1`` for tests and screen work

Importing this package imports nothing outside the standard library.
"""

from __future__ import annotations
