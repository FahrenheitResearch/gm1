#!/bin/bash
# Build gpuwm-cm1's release artifacts inside a manylinux_2_28 container on a build node.
#
#   tools/release/build_manylinux.sh SOURCE.tar.gz OUTDIR [CACHEDIR]
#
# SOURCE.tar.gz is `git archive --format=tar.gz --prefix=gpuwm-cm1/ <commit>` of the release commit (a clean
# tree: no build/, no *.egg-info). OUTDIR receives the sdist, the manylinux wheel, the artifact inspection, the
# build receipt and SHA256SUMS. CACHEDIR (default OUTDIR/../cache) keeps rustup, cargo and pip downloads between
# builds. Needs docker or podman (ENGINE=podman) and network access for rustup and PyPI.
#
# The wheel is built in a manylinux container (docker or podman) on a Linux build machine, never from a
# developer's working checkout. The route is the one docs/INSTALL.md gives: an sdist, extracted outside the checkout, the wheel built from
# the extraction, then `auditwheel repair`, which proves the Rust tools and libcm1_state.so link only against
# manylinux_2_28 libraries and renames the wheel without touching the sealed payload.
#
# Environment:
#   ENGINE          docker (default) or podman
#   IMAGE           default quay.io/pypa/manylinux_2_28_x86_64
#   RUST_TOOLCHAIN  default 1.95.0 (the toolchain the release tree is tested with)
#   JOBS            cargo jobs and container CPUs (default 8)
#   SOURCE_COMMIT   the commit SOURCE.tar.gz was archived from (recorded in the receipt)
#   SOURCE_DATE_EPOCH  when set (tools/release/publish.sh sets the commit time), passed into the container so the
#                   sdist and wheel timestamps, and so their sha256, do not depend on when the build ran
set -euo pipefail

SOURCE=$(readlink -f "$1")
OUT=$(readlink -f "$2")
CACHE=$(readlink -m "${3:-$OUT/../cache}")
ENGINE=${ENGINE:-docker}
IMAGE=${IMAGE:-quay.io/pypa/manylinux_2_28_x86_64}
RUST_TOOLCHAIN=${RUST_TOOLCHAIN:-1.95.0}
JOBS=${JOBS:-8}
SOURCE_COMMIT=${SOURCE_COMMIT:-unknown}
REPRO_ENV=()
# Rootless podman already maps the container's root to the calling user; a chown to HOST_UID inside it
# would hand the files to a subordinate id the caller cannot remove.
CHOWN_BACK=1
[ "$ENGINE" = podman ] && CHOWN_BACK=0
[ -n "${SOURCE_DATE_EPOCH:-}" ] && REPRO_ENV=(-e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH")

[ -f "$SOURCE" ] || { echo "no source archive: $SOURCE" >&2; exit 2; }
mkdir -p "$OUT" "$CACHE/rustup" "$CACHE/cargo" "$CACHE/pip"
if [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "OUTDIR is not empty: $OUT (each build gets a fresh folder)" >&2
  exit 2
fi
cp "$SOURCE" "$OUT/.source.tar.gz"

set +e
"$ENGINE" run --rm --cpus="$JOBS" \
  -v "$OUT:/io" -v "$CACHE/rustup:/opt/rustup" -v "$CACHE/cargo:/opt/cargo" -v "$CACHE/pip:/opt/pip-cache" \
  -e RUSTUP_HOME=/opt/rustup -e CARGO_HOME=/opt/cargo -e PIP_CACHE_DIR=/opt/pip-cache \
  -e CARGO_BUILD_JOBS="$JOBS" -e RUST_TOOLCHAIN="$RUST_TOOLCHAIN" -e SOURCE_COMMIT="$SOURCE_COMMIT" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -e CHOWN_BACK="$CHOWN_BACK" "${REPRO_ENV[@]}" \
  "$IMAGE" nice -n 19 bash -euo pipefail -c '
    give_back() { [ "$CHOWN_BACK" = 1 ] || return 0; chown -R "$HOST_UID:$HOST_GID" /io /opt/rustup /opt/cargo /opt/pip-cache; }
    trap give_back EXIT
    PY=/opt/python/cp312-cp312/bin/python
    if [ ! -x /opt/cargo/bin/cargo ]; then
      curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "$RUST_TOOLCHAIN" --no-modify-path
    fi
    export PATH=/opt/cargo/bin:$PATH
    rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal >/dev/null
    rustup default "$RUST_TOOLCHAIN" >/dev/null
    $PY -m venv /work/venv
    /work/venv/bin/pip install -q "build==1.6.1" "auditwheel==6.8.2"
    mkdir -p /work/src /work/extract /work/sdist /work/wheel
    tar -xzf /io/.source.tar.gz -C /work/src --strip-components=1
    cd /work/src
    /work/venv/bin/python -m build --sdist --outdir /work/sdist
    SDIST=$(ls /work/sdist/*.tar.gz)
    if [ -n "${SOURCE_DATE_EPOCH:-}" ]; then
      # setuptools stamps the folders, PKG-INFO, setup.cfg and the gzip header with the build time; clamp
      # every time to SOURCE_DATE_EPOCH so the sdist sha256 depends on the source alone.
      $PY - "$SDIST" <<CLAMPEOF
import gzip, io, os, sys, tarfile
path, epoch = sys.argv[1], int(os.environ["SOURCE_DATE_EPOCH"])
with tarfile.open(path, "r:gz") as source:
    members = [(m, source.extractfile(m).read() if m.isfile() else None) for m in source.getmembers()]
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as out:
    for member, data in members:
        member.mtime = min(int(member.mtime), epoch)
        member.pax_headers = {}
        member.uid = member.gid = 0
        member.uname = member.gname = "root"
        out.addfile(member, io.BytesIO(data) if data is not None else None)
with open(path, "wb") as handle:
    with gzip.GzipFile(os.path.basename(path)[:-3], "wb", 9, handle, mtime=epoch) as stream:
        stream.write(buffer.getvalue())
CLAMPEOF
    fi
    tar -xzf "$SDIST" -C /work/extract
    cd /work/extract/*/
    /work/venv/bin/python -m build --wheel --outdir /work/wheel
    /work/venv/bin/auditwheel show /work/wheel/*.whl > /io/auditwheel-show.txt
    /work/venv/bin/auditwheel repair --plat manylinux_2_28_x86_64 -w /io /work/wheel/*.whl
    cp "$SDIST" /io/
    WHEEL=$(ls /io/*.whl)
    set +e
    # The inspector runs from the source archive, not the sdist: in the development repository that tree
    # also holds the release denylist gate, which is development tooling and is not in the sdist.
    DENY=()
    [ -f /work/src/tools/release/scan_release.py ] && DENY=(--require-denylist)
    /work/venv/bin/python /work/src/tools/inspect_artifacts.py --require-manylinux "${DENY[@]}" "$WHEEL" \
      "/io/$(basename "$SDIST")" > /io/artifact-inspection.json
    INSPECT_RC=$?
    set -e
    $PY - "$WHEEL" "/io/$(basename "$SDIST")" "$INSPECT_RC" <<PYEOF
import hashlib, json, os, platform, subprocess, sys
wheel, sdist, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
def out(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
receipt = {
    "schema": "gpuwm-cm1.release-build/v1",
    "source_commit": os.environ["SOURCE_COMMIT"],
    "image": out("bash", "-c", "cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2"),
    "glibc": platform.libc_ver()[1],
    "python": out("/opt/python/cp312-cp312/bin/python", "--version"),
    "rustc": out("rustc", "--version"),
    "cargo": out("cargo", "--version"),
    "auditwheel": out("/work/venv/bin/auditwheel", "--version"),
    "artifacts": {
        os.path.basename(p): {"sha256": sha(p), "bytes": os.path.getsize(p)} for p in (wheel, sdist)
    },
    "inspection_passed": rc == 0,
}
json.dump(receipt, open("/io/build-receipt.json", "w"), indent=2, sort_keys=True)
PYEOF
    exit $INSPECT_RC
  '
rc=$?
set -e
rm -f "$OUT/.source.tar.gz"
(cd "$OUT" && sha256sum -- *.whl *.tar.gz > SHA256SUMS)
echo "build rc=$rc; artifacts in $OUT"
exit $rc
