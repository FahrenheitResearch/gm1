// The server's JSON endpoints and live run events. Every screen talks to the server through this module.
//
// GETs ride on the HttpOnly cookie the first link set. POSTs also carry the token in the X-GM1-Token header,
// which /api/session hands to this page (same origin only). Every POST reply carries `command`: the one
// gpuwm-cm1 line it ran, which the "Show command" component draws.

const HEADER = "X-GM1-Token";
const KINDS = ["admitted", "startup", "progress", "stats", "frame", "restart", "warning", "refused", "end",
  "status", "segment"];

export class ApiError extends Error {
  constructor(status, body) {
    super((body && body.message) || `The server answered ${status}.`);
    this.status = status;
    this.body = body || {};
    this.fix = this.body.fix || "";
    this.command = this.body.command || null;
  }
}

async function read(response) {
  let body = null;
  try { body = await response.json(); } catch (_) { body = null; }
  if (!response.ok) throw new ApiError(response.status, body);
  return body;
}

let sessionPromise = null;

// {version, token, mode, root, runner, host, port, commands_available, endpoints}
export function session(refresh = false) {
  if (!sessionPromise || refresh) {
    sessionPromise = fetch("/api/session", { credentials: "same-origin", cache: "no-store" }).then(read);
    sessionPromise.catch(() => { sessionPromise = null; });
  }
  return sessionPromise;
}

export async function get(path) {
  return read(await fetch(path, { credentials: "same-origin", cache: "no-store" }));
}

const commandListeners = new Set();

export async function post(path, body = {}, opts = {}) {
  const s = await session();
  const response = await fetch(path, {
    method: "POST", credentials: "same-origin", cache: "no-store",
    headers: { "Content-Type": "application/json", [HEADER]: s.token },
    body: JSON.stringify(body),
  });
  let data;
  try {
    data = await read(response);
  } catch (err) {
    if (!opts.quiet && err instanceof ApiError && err.command) for (const fn of commandListeners) fn(err.body);
    throw err;
  }
  if (!opts.quiet && data && typeof data.command === "string" && !data.dry_run) {
    for (const fn of commandListeners) fn(data);
  }
  return data;
}

// The exact command a POST would run, with nothing run and nothing changed ({"dry_run": true}): what
// "Show command" shows before a button is pressed. It never reaches the "Commands this page ran" list.
export function dryRun(path, body = {}) {
  return post(path, { ...body, dry_run: true }, { quiet: true });
}

// Called with every POST reply that carries a command.
export function onCommand(fn) {
  commandListeners.add(fn);
  return () => commandListeners.delete(fn);
}

export const runPath = (id) => `/api/runs/${encodeURIComponent(id)}`;

// Live events of one run. handlers: {progress, frame, end, status, segment, ..., any(kind, data), error()}.
// Returns a function that closes the stream. The browser reconnects on its own and the server resumes from
// the last event it sent.
export function follow(runId, handlers) {
  const source = new EventSource(`${runPath(runId)}/events`);
  for (const kind of KINDS) {
    source.addEventListener(kind, (ev) => {
      let data;
      try { data = JSON.parse(ev.data); } catch (_) { return; }
      if (handlers[kind]) handlers[kind](data);
      if (handlers.any) handlers.any(kind, data);
    });
  }
  source.onerror = () => { if (handlers.error) handlers.error(source.readyState); };
  return () => source.close();
}

// ---- the view mode: guided or advanced. It changes what is drawn, never what runs.

const modeListeners = new Set();

export function currentMode() {
  return document.body.dataset.mode || "guided";
}

export function applyMode(mode) {
  document.body.dataset.mode = mode;
  for (const fn of modeListeners) fn(mode);
}

export function onMode(fn) {
  modeListeners.add(fn);
  return () => modeListeners.delete(fn);
}

export async function setMode(mode) {
  const reply = await post("/api/mode", { mode });
  applyMode(reply.mode);
  return reply;
}
