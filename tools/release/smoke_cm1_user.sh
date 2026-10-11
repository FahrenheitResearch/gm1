#!/bin/bash
# A CM1 user's first hour with the built wheel, in a fresh venv, with CUDA_PATH and CUDA_HOME unset: doctor -> CM1's stock supercell namelist -> import -> run -> cm1out -> the CM1 user's own xarray
# script on the netCDF. Writes WORK/smoke.json and one log per step under WORK/logs/.
#
#   tools/release/smoke_cm1_user.sh --wheel WHEEL --constraints FILE --python PYTHON --work DIR
#       [--uv UV] [--budget-mib N] [--until T] [--cpu-only [--from-run RUNDIR]]
#
# --uv UV takes docs/INSTALL.md's uv route: `uv venv -p PYTHON` and `uv pip install` instead of venv and pip.
# GPU mode (the default) runs the whole storm on the first visible GPU; take the machine's GPU lock around it.
#   --budget-mib N   also set the run's GPU memory budget (gpu.memory_budget_mib) before it starts
#   --until T        stop the storm at T (default: the namelist's timax, 2 h for the stock deck)
# --cpu-only is for a machine with no NVIDIA GPU: the same install, doctor must say what is missing, import,
# estimate, the presets and the GUI server work, and with --from-run a finished run folder copied from a GPU
# machine is converted (cm1out), drawn (quicklook, the environment look and a film), shown by the GUI server's
# Results payload (every More looks row) and read by the user script there. Both modes install the wheel with its film extra (the GPU stack is in its base requirements).
set -uo pipefail

WHEEL= CONSTRAINTS= PY= WORK= BUDGET= UNTIL= CPU_ONLY=0 FROM_RUN= UV=
while [ $# -gt 0 ]; do
  case "$1" in
    --wheel) WHEEL=$(readlink -f "$2"); shift 2 ;;
    --constraints) CONSTRAINTS=$(readlink -f "$2"); shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --work) WORK=$(readlink -m "$2"); shift 2 ;;
    --budget-mib) BUDGET=$2; shift 2 ;;
    --until) UNTIL=$2; shift 2 ;;
    --cpu-only) CPU_ONLY=1; shift ;;
    --from-run) FROM_RUN=$(readlink -f "$2"); shift 2 ;;
    --uv) UV=$(readlink -f "$2"); shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -f "$WHEEL" ] && [ -f "$CONSTRAINTS" ] && [ -n "$PY" ] && [ -n "$WORK" ] || {
  echo "usage: $0 --wheel WHEEL --constraints FILE --python PYTHON --work DIR [...]" >&2; exit 2; }
HERE=$(dirname "$(readlink -f "$0")")
if [ -e "$WORK" ]; then echo "WORK exists: $WORK (each smoke gets a fresh folder)" >&2; exit 2; fi
mkdir -p "$WORK/logs" "$WORK/runs"
# What a CM1 user's shell does not have: no CUDA toolkit variables, no checkout on the path.
CUDA_PATH_BEFORE=${CUDA_PATH-<unset>} CUDA_HOME_BEFORE=${CUDA_HOME-<unset>}
unset CUDA_PATH CUDA_HOME PYTHONPATH GM1_CM1_R21_NAMELIST GPUWM_CM1_NVIDIA_SMI
export GPUWM_CM1_CACHE_DIR="$WORK/cache"   # the rates this smoke learns stay in its folder
V="$WORK/venv"
G="$V/bin/gpuwm-cm1"
STEPS="$WORK/steps.tsv"
: > "$STEPS"

# step NAME EXPECTED_RC COMMAND...: run one step, log it, record rc and seconds.
step() {
  local name=$1 expected=$2; shift 2
  local start end rc
  start=$(date +%s.%N)
  "$@" > "$WORK/logs/$name.out" 2> "$WORK/logs/$name.err"
  rc=$?
  end=$(date +%s.%N)
  printf '%s\t%s\t%s\t%s\n' "$name" "$expected" "$rc" "$(awk "BEGIN{printf \"%.1f\", $end - $start}")" >> "$STEPS"
  echo "[$(date -u +%H:%M:%S)] $name rc=$rc (expected $expected)"
  return $rc
}

# looks RUNDIR: a look drawn once per run (the environment) and a film (the wind swath) on the CPU, as Results
# does; with the film extra an MP4, else a GIF, beside the animated PNG.
looks() {
  step look-environment 0 "$G" quicklook "$1" --look environment --json
  step film-swath 0 "$G" quicklook "$1" --film swath --json
}

if [ -n "$UV" ]; then
  step venv 0 "$UV" venv -p "$PY" "$V"
  step install 0 "$UV" pip install --python "$V/bin/python" "$WHEEL[film]" -c "$CONSTRAINTS"
  PIP=("$UV" pip install --python "$V/bin/python")
else
  step venv 0 "$PY" -m venv "$V"
  step install 0 "$V/bin/pip" install --no-input "$WHEEL[film]" -c "$CONSTRAINTS"
  PIP=("$V/bin/pip" install --no-input)
fi
step version 0 "$G" version
if [ $CPU_ONLY = 1 ]; then
  step doctor 1 "$G" doctor --plain --root "$WORK/runs"
  # ... and say which thing is missing: the NVIDIA driver, on its own FIX line.
  step doctor-names-driver 0 grep -E "^FIX +Driver +No NVIDIA driver found" "$WORK/logs/doctor.out"
else
  step doctor 0 "$G" doctor --plain --root "$WORK/runs"
fi
step examples 0 "$G" examples --cm1 supercell --copy "$WORK/cm1"
RUN="$WORK/runs/supercell"
step import 0 "$G" import "$WORK/cm1/supercell/namelist.input" --out "$RUN"
EDITS=(--setting output.also_netcdf=true)
[ -n "$BUDGET" ] && EDITS+=(--setting "gpu.memory_budget_mib=$BUDGET")
step edit 0 "$G" edit "$RUN" "${EDITS[@]}"
step preset-list 0 "$G" preset list --json

if [ $CPU_ONLY = 1 ]; then
  step estimate 0 "$G" estimate "$RUN" --json
  step preset-fit 0 "$G" preset fit supercell-1km --budget 10m --json
  # The GUI server on a machine with no GPU: it starts, answers with its token, and never maps CuPy.
  step gui 0 "$V/bin/python" - "$G" "$WORK" <<'PYEOF'
import http.client, json, re, subprocess, sys, time
g, work = sys.argv[1], sys.argv[2]
server = subprocess.Popen([g, "gui", "--root", f"{work}/gui-runs", "--port", "0"], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
url = None
deadline = time.monotonic() + 60
while url is None and time.monotonic() < deadline:
    line = server.stdout.readline()
    match = re.search(r"http://127\.0\.0\.1:(\d+)/\?token=(\S+)", line or "")
    if match:
        url, port, token = match.group(0), int(match.group(1)), match.group(2)
assert url, "the GUI server printed no link"
answers = {}
for path in ("/", "/api/session", "/api/presets", "/api/system", "/api/runs"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    c.request("GET", path, headers={"Host": f"127.0.0.1:{port}", "X-GM1-Token": token})
    r = c.getresponse()
    body = r.read()
    answers[path] = r.status
maps = open(f"/proc/{server.pid}/maps").read()
server.terminate()
server.wait(timeout=30)
print(json.dumps({"url": re.sub(r"token=\S+", "token=...", url), "status": answers,
                  "cupy_mapped": "cupy" in maps or "libcuda" in maps}))
assert all(status == 200 for status in answers.values()), answers
assert "cupy" not in maps and "libcuda" not in maps
PYEOF
  if [ -n "$FROM_RUN" ]; then
    COPY="$WORK/runs/from-gpu"
    step copy-run 0 cp -a "$FROM_RUN" "$COPY"
    step cm1out 0 "$G" cm1out "$COPY" --force --json
    step quicklook 0 "$G" quicklook "$COPY" --json
    looks "$COPY"
    step gui-results 0 "$V/bin/python" "$HERE/smoke_gui_results.py" "$G" "$WORK/runs" from-gpu
    RUN=$COPY
  fi
else
  step estimate 0 "$G" estimate "$RUN" --json
  RUN_ARGS=()
  [ -n "$UNTIL" ] && RUN_ARGS+=(--until "$UNTIL")
  step run 0 "$G" run "$RUN" "${RUN_ARGS[@]}"
  step status 0 "$G" status "$RUN" --json
  step cm1out 0 "$G" cm1out "$RUN" --json
  step quicklook 0 "$G" quicklook "$RUN" --json
  looks "$RUN"
fi

NC=
if [ -f "$RUN/cm1out/cm1out.nc" ]; then NC="$RUN/cm1out/cm1out.nc"
else NC=$(ls "$RUN"/cm1out/cm1out_0*.nc 2>/dev/null | tail -1); fi
if [ -n "$NC" ]; then
  # The CM1 user's own tools: xarray is theirs, not a gpuwm-cm1 dependency.
  step user-xarray 0 "${PIP[@]}" xarray
  step user-script 0 "$V/bin/python" "$HERE/cm1_user_script.py" "$NC"
fi

"$V/bin/python" - "$WORK" "$WHEEL" "$PY" "$CPU_ONLY" "$CUDA_PATH_BEFORE" "$CUDA_HOME_BEFORE" "$NC" "$UV" <<'PYEOF'
import hashlib, json, os, platform, re, socket, subprocess, sys
work, wheel, py, cpu_only, cuda_path_before, cuda_home_before, nc, uv = sys.argv[1:9]
steps = []
for line in open(f"{work}/steps.tsv"):
    name, expected, rc, seconds = line.rstrip("\n").split("\t")
    steps.append({"step": name, "expected_rc": int(expected), "rc": int(rc), "seconds": float(seconds)})
def text(name, kind="out"):
    try:
        return open(f"{work}/logs/{name}.{kind}").read()
    except OSError:
        return ""
def sha(path):
    h = hashlib.sha256(); h.update(open(path, "rb").read()); return h.hexdigest()
def smi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    except OSError:
        return None
freeze = subprocess.run([f"{work}/venv/bin/python", "-m", "pip", "freeze"], capture_output=True, text=True).stdout.split()
if not freeze and uv:
    freeze = subprocess.run([uv, "pip", "freeze", "--python", f"{work}/venv/bin/python"], capture_output=True,
                            text=True).stdout.split()
user = text("user-script")
summary = {
    "schema": "gpuwm-cm1.release-smoke-cm1-user/v1",
    "host": socket.gethostname(),
    "mode": "cpu-only" if cpu_only == "1" else "gpu",
    "python": subprocess.run([f"{work}/venv/bin/python", "--version"], capture_output=True, text=True).stdout.strip(),
    "python_given": py,
    "install_route": "uv venv + uv pip install" if uv else "python -m venv + pip install",
    "platform": platform.platform(),
    "gpu": smi(),
    "environment": {"CUDA_PATH_before": cuda_path_before, "CUDA_HOME_before": cuda_home_before,
                    "CUDA_PATH_in_smoke": "<unset>", "CUDA_HOME_in_smoke": "<unset>"},
    "wheel": {"file": os.path.basename(wheel), "sha256": sha(wheel)},
    "installed": sorted(p for p in freeze if re.match(r"(gpuwm|cupy|numpy|netCDF4|netcdf4|matplotlib|nvidia-cuda-nvrtc|xarray)", p, re.I)),
    "steps": steps,
    "doctor": text("doctor").strip().splitlines()[-40:],
    "version": text("version").strip(),
    "user_script": user.strip().splitlines(),
    "netcdf": None if not nc else {"file": os.path.relpath(nc, work), "bytes": os.path.getsize(nc)},
    "cm1out_files": sorted(os.listdir(os.path.dirname(nc))) if nc else [],
}
match = re.search(r"wmax\(t\): (\[.*\])", user)
if match:
    summary["wmax_t"] = json.loads(match.group(1))
summary["passed"] = all(s["rc"] == s["expected_rc"] for s in steps) and bool(nc) == (
    cpu_only != "1" or os.path.isdir(f"{work}/runs/from-gpu"))
json.dump(summary, open(f"{work}/smoke.json", "w"), indent=2)
print(json.dumps({"passed": summary["passed"], "steps": [(s["step"], s["rc"]) for s in steps],
                  "wmax_t": summary.get("wmax_t")}))
sys.exit(0 if summary["passed"] else 1)
PYEOF
