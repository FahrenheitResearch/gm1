#!/bin/sh
# GM1 (gpuwm-cm1) installer for Linux and for Ubuntu under WSL2 on Windows.
#
#   curl -LsSf https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.sh | sh
#
# It checks the computer (x86-64, glibc, the NVIDIA driver), installs uv into
# ~/.local/bin when it is missing (or older than the tested one), installs GM1
# with uv's own Python 3.12 (uv tool install, with the film add-on for MP4s),
# puts ~/.local/bin on PATH for new terminals, runs `gpuwm-cm1 doctor --plain`,
# and says what to type next. Nothing needs sudo. Running it again updates GM1
# to the newest release (close GM1 and let running storms finish first).
#
# Uninstall (your runs in ~/gm1-runs are kept; GM1's downloaded packages are
# removed from uv's cache):
#
#   curl -LsSf https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.sh | sh -s -- --uninstall
#
# Settings, as environment variables (GM1_VERSION=0.1.3 sh install.sh):
#
#   GM1_VERSION=X.Y.Z        install this release instead of the newest
#   GM1_WHEEL=FILE.whl       install this wheel file (a build not on PyPI yet)
#   GM1_FIND_LINKS=FOLDER    also look for gpuwm-cm1 wheels in FOLDER
#   GM1_CONSTRAINTS=FILE     the exact versions to install with; by default the
#                            release's own constraints-gpu-cu13.txt when it can be
#                            downloaded, and GM1's own version ranges otherwise
#   GM1_REINSTALL=1          reinstall every package (repairs a damaged install)
#   GM1_NO_MODIFY_PATH=1     leave the shell's startup files alone
#   GM1_SKIP_DOCTOR=1        do not run the check at the end

set -u

GM1_REPO_RAW="https://raw.githubusercontent.com/FahrenheitResearch/gm1"
GM1_RELEASES="https://github.com/FahrenheitResearch/gm1/releases/download"
GM1_PYPI_JSON="https://pypi.org/pypi/gpuwm-cm1/json"
# The uv this installer is tested with; an older uv is not used (it may lack what the install needs).
UV_TESTED=0.13.0
UV_INSTALLER="https://astral.sh/uv/$UV_TESTED/install.sh"
NVIDIA_DOWNLOAD="https://www.nvidia.com/Download/index.aspx"
MIN_DRIVER=580
PYTHON_VERSION=3.12
PATH_MARK="# Added by the GM1 installer"
# MP4 films of the looks (the film extra); the installer always adds it, so Save as film makes MP4s too.
FILM_PACKAGE="imageio-ffmpeg>=0.5"
# GM1's packages in uv's download cache (about 2.5 GB), removed on uninstall.
CACHED_PACKAGES="gpuwm-cm1 gpuwm gpuwm-data cupy-cuda13x cuda-toolkit cuda-pathfinder nvidia-cublas nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cufft nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-nvjitlink imageio-ffmpeg"

say() { printf '%s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }
warn() { printf '\n!!  %s\n' "$*"; }
fail() { printf '\nGM1 was not installed: %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# The folder the installer was started in: GM1_WHEEL and the other paths are relative to it.
START_DIR=$(pwd)
absolute() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$START_DIR/$1" ;;
  esac
}

# Download URL to FILE with curl or wget; non-zero when neither works.
fetch() {
  if have curl; then
    curl -fsSL --retry 3 --connect-timeout 20 "$1" -o "$2"
  elif have wget; then
    wget -q -T 20 -t 3 -O "$2" "$1"
  else
    return 127
  fi
}

# --------------------------------------------------------------- the computer

is_wsl() { grep -qi microsoft /proc/version 2>/dev/null; }

check_platform() {
  os=$(uname -s 2>/dev/null || echo unknown)
  if [ "$os" != Linux ]; then
    fail "GM1 runs on Linux, and on Windows inside WSL2. On Windows, open PowerShell and run:
  irm https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.ps1 | iex"
  fi
  arch=$(uname -m 2>/dev/null || echo unknown)
  if [ "$arch" != x86_64 ]; then
    fail "GM1 needs an x86-64 (Intel or AMD) computer; this one is $arch."
  fi
  glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')
  if [ -z "$glibc" ]; then
    fail "GM1 needs a Linux with glibc 2.28 or newer (Ubuntu 20.04, Debian 10, RHEL 8 or newer); this one has none."
  fi
  glibc_major=${glibc%%.*}
  glibc_minor=${glibc#*.}
  glibc_minor=${glibc_minor%%.*}
  if [ "$glibc_major" -lt 2 ] || { [ "$glibc_major" -eq 2 ] && [ "$glibc_minor" -lt 28 ]; }; then
    fail "GM1 needs glibc 2.28 or newer (Ubuntu 20.04, Debian 10, RHEL 8 or newer); this Linux has $glibc."
  fi
  WSL=0
  if is_wsl; then
    WSL=2
    case "$(uname -r)" in
      *microsoft-standard*|*WSL2*) ;;
      *) WSL=1 ;;
    esac
  fi
  if [ "$WSL" = 1 ]; then
    warn "This is WSL 1, which cannot use the graphics card. In Windows PowerShell, run:
    wsl --set-version ${WSL_DISTRO_NAME:-Ubuntu} 2
then open Ubuntu again and rerun this installer."
  fi
}

find_nvidia_smi() {
  # GM1_NVIDIA_SMI: another program to ask, or "none" for a computer without one (the installer's tests).
  if [ -n "${GM1_NVIDIA_SMI:-}" ]; then
    [ "$GM1_NVIDIA_SMI" = none ] || echo "$GM1_NVIDIA_SMI"
  elif have nvidia-smi; then
    command -v nvidia-smi
  elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
    echo /usr/lib/wsl/lib/nvidia-smi
  fi
}

driver_fix() {
  if [ "$WSL" != 0 ]; then
    say "    Fix: on Windows, install the newest NVIDIA driver (R$MIN_DRIVER or newer) from"
    say "    $NVIDIA_DOWNLOAD"
    say "    then in Windows PowerShell run:  wsl --update   and then:  wsl --shutdown"
    say "    and open Ubuntu again. Never install an NVIDIA driver inside WSL."
  else
    say "    Fix: install the NVIDIA driver R$MIN_DRIVER or newer from $NVIDIA_DOWNLOAD"
    say "    (on Ubuntu: sudo ubuntu-drivers install nvidia:$MIN_DRIVER), then restart the computer and check"
    say "    that nvidia-smi says $MIN_DRIVER or newer."
  fi
}

check_driver() {
  smi=$(find_nvidia_smi)
  if [ -z "$smi" ]; then
    if [ "$WSL" != 0 ]; then
      warn "No NVIDIA graphics card is visible inside WSL. GM1 installs anyway, but cannot run until this is fixed."
    else
      warn "No NVIDIA driver found (nvidia-smi is missing). GM1 installs anyway, but cannot run until this is fixed."
    fi
    driver_fix
    return 0
  fi
  if have timeout; then
    answer=$(timeout 30 "$smi" --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null | head -n 1)
  else
    answer=$("$smi" --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null | head -n 1)
  fi
  driver=$(printf '%s' "$answer" | cut -d, -f1 | tr -d ' ')
  card=$(printf '%s' "$answer" | cut -d, -f2- | sed 's/^ *//')
  major=${driver%%.*}
  case "$major" in
    ''|*[!0-9]*)
      warn "nvidia-smi did not answer. GM1 installs anyway, but cannot run until the driver works."
      driver_fix
      return 0 ;;
  esac
  if [ "$major" -lt "$MIN_DRIVER" ]; then
    warn "The NVIDIA driver is $driver; GM1 needs R$MIN_DRIVER or newer (CUDA 13). GM1 installs anyway."
    driver_fix
  else
    say "NVIDIA driver $driver, $card."
  fi
}

check_disk() {
  free_kb=$(df -Pk "$HOME" 2>/dev/null | awk 'NR == 2 {print $4}')
  case "$free_kb" in
    ''|*[!0-9]*) return 0 ;;
  esac
  if [ "$free_kb" -lt 4000000 ]; then
    warn "Only $((free_kb / 1000000)) GB is free in your home folder. GM1 needs about 3 GB to install and more for its storms."
  fi
}

# --------------------------------------------------------------- GM1 running now

# Whether this user runs GM1's window or a storm now: replacing GM1 under them would break them mid-run.
# GM1_PROC_ROOT: another /proc to read (the installer's tests).
gm1_running() {
  proc=${GM1_PROC_ROOT:-/proc}
  for cmdline in "$proc"/[0-9]*/cmdline; do
    [ -r "$cmdline" ] || continue
    [ -O "${cmdline%/cmdline}" ] || continue
    [ "${cmdline%/cmdline}" = "$proc/$$" ] && continue
    if tr '\000' ' ' < "$cmdline" 2>/dev/null | grep -Eq '(gpuwm-cm1|cm1core) (gui|run|resume|cartesian)( |$)'; then
      return 0
    fi
  done
  return 1
}

refuse_while_running() {
  if gm1_running; then
    printf '\n%s\n' "GM1 was not changed: a storm (or the GM1 window) is running. Close the GM1 window and wait for the storm to finish, or stop it in GM1, then paste the line again." >&2
    exit 1
  fi
}

# --------------------------------------------------------------- uv

UV=""

# 0 when version $1 (X.Y.Z) is at least $2.
at_least() {
  printf '%s\n%s\n' "$2" "$1" | awk -F. '
    NR == 1 { for (i = 1; i <= 3; i++) want[i] = $i + 0 }
    NR == 2 { for (i = 1; i <= 3; i++) { have = $i + 0; if (have > want[i]) exit 0; if (have < want[i]) exit 1 } exit 0 }'
}

find_uv() {
  UV_OLD=""
  for candidate in "$(command -v uv 2>/dev/null)" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ] && "$candidate" tool --help >/dev/null 2>&1; then
      found=$("$candidate" --version 2>/dev/null | awk '{print $2}')
      if [ -n "$found" ] && at_least "$found" "$UV_TESTED"; then
        UV=$candidate
        return 0
      fi
      UV_OLD="${UV_OLD:-$candidate $found}"
    fi
  done
  return 1
}

# Any uv that can remove a tool (uninstalling never needs the newest).
find_any_uv() {
  find_uv && return 0
  for candidate in "$(command -v uv 2>/dev/null)" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ] && "$candidate" tool --help >/dev/null 2>&1; then
      UV=$candidate
      return 0
    fi
  done
  return 1
}

install_uv() {
  if find_uv; then
    say "uv: $("$UV" --version 2>/dev/null)"
    return 0
  fi
  if [ -n "$UV_OLD" ]; then
    say "The uv here ($UV_OLD) is older than the one GM1 is tested with ($UV_TESTED)."
  fi
  step "Installing uv $UV_TESTED (the tool that installs GM1 and its Python) into ~/.local/bin"
  tmp_uv=$(mktemp) || fail "could not make a temporary file."
  if ! fetch "$UV_INSTALLER" "$tmp_uv"; then
    rm -f "$tmp_uv"
    if ! have curl && ! have wget; then
      fail "this computer has neither curl nor wget. Install one (on Ubuntu: sudo apt install curl) and rerun."
    fi
    fail "could not download uv from $UV_INSTALLER. Check the internet connection and rerun."
  fi
  UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh "$tmp_uv" || {
    rm -f "$tmp_uv"
    fail "the uv installer failed (see above). Rerun this installer; if it fails again, ask for help with the text above."
  }
  rm -f "$tmp_uv"
  find_uv || fail "uv $UV_TESTED was installed but cannot be run from $HOME/.local/bin/uv."
}

# --------------------------------------------------------------- PATH for new terminals

ensure_path() {
  bin_dir=$1
  [ "${GM1_NO_MODIFY_PATH:-0}" = 1 ] && return 0
  case "$bin_dir" in
    "$HOME"/*) shown="\$HOME/${bin_dir#"$HOME"/}" ;;
    *) shown=$bin_dir ;;
  esac
  line="case \":\$PATH:\" in *\":$shown:\"*) ;; *) export PATH=\"$shown:\$PATH\" ;; esac"
  for rc in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.zshrc"; do
    case "$rc" in
      "$HOME/.profile") ;;
      *) [ -f "$rc" ] || continue ;;
    esac
    if [ -f "$rc" ] && grep -qF "$PATH_MARK" "$rc"; then
      continue
    fi
    printf '\n%s\n%s\n' "$PATH_MARK: where gpuwm-cm1 and uv are." "$line" >> "$rc" || true
  done
}

# --------------------------------------------------------------- what to install

# The newest gpuwm-cm1 on PyPI, or nothing.
latest_version() {
  tmp_json=$(mktemp) || return 0
  if fetch "$GM1_PYPI_JSON" "$tmp_json" 2>/dev/null; then
    tr ',' '\n' < "$tmp_json" | sed -n 's/^ *"version": *"\([^"]*\)".*/\1/p' | head -n 1
  fi
  rm -f "$tmp_json"
}

# A file with the release's exact versions, or nothing (GM1's own ranges are used then).
constraints_for() {
  version=$1
  [ -n "$version" ] || return 0
  tmp_c=$(mktemp) || return 0
  for url in "$GM1_REPO_RAW/v$version/constraints/gpu-cu13-validated.txt" \
             "$GM1_RELEASES/v$version/constraints-gpu-cu13.txt"; do
    if fetch "$url" "$tmp_c" 2>/dev/null && grep -q '^cupy-cuda13x==' "$tmp_c"; then
      echo "$tmp_c"
      return 0
    fi
  done
  rm -f "$tmp_c"
}

# --------------------------------------------------------------- install and uninstall

do_install() {
  step "Checking this computer"
  check_platform
  refuse_while_running
  check_driver
  check_disk

  step "Getting uv"
  install_uv

  spec=gpuwm-cm1
  version=${GM1_VERSION:-}
  if [ -n "${GM1_WHEEL:-}" ]; then
    [ -f "$(absolute "$GM1_WHEEL")" ] || fail "GM1_WHEEL=$GM1_WHEEL is not a file."
    spec=$(absolute "$GM1_WHEEL")
    wheel_name=$(basename "$GM1_WHEEL")
    version=$(printf '%s' "$wheel_name" | sed -n 's/^gpuwm_cm1-\([^-]*\)-.*/\1/p')
  elif [ -n "$version" ]; then
    spec="gpuwm-cm1==$version"
  elif [ -z "${GM1_FIND_LINKS:-}" ]; then
    # The newest release, pinned, so that it and its constraints file agree.
    version=$(latest_version)
    [ -n "$version" ] && spec="gpuwm-cm1==$version"
  fi

  constraints=""
  made_constraints=0
  if [ -n "${GM1_CONSTRAINTS:-}" ]; then
    constraints=$(absolute "$GM1_CONSTRAINTS")
    [ -f "$constraints" ] || fail "GM1_CONSTRAINTS=$GM1_CONSTRAINTS is not a file."
  else
    constraints=$(constraints_for "$version")
    [ -n "$constraints" ] && made_constraints=1
  fi

  # uv's own Python, never the system's: an apt autoremove or an Ubuntu upgrade must not break GM1.
  set -- tool install --python "$PYTHON_VERSION" --python-preference only-managed --force --upgrade
  if [ "${GM1_REINSTALL:-0}" = 1 ]; then
    set -- "$@" --reinstall
  else
    set -- "$@" --reinstall-package gpuwm-cm1
  fi
  [ -n "$constraints" ] && set -- "$@" --constraints "$constraints"
  [ -n "${GM1_FIND_LINKS:-}" ] && set -- "$@" --find-links "$(absolute "$GM1_FIND_LINKS")"
  set -- "$@" --with "$FILM_PACKAGE" "$spec"

  # An update finds the big packages already on this computer: the 2.5 GB is the first install's.
  if "$UV" tool list 2>/dev/null | grep -q '^gpuwm-cm1 '; then
    step "Updating GM1${version:+ to $version} with Python $PYTHON_VERSION (only what changed is downloaded)"
  else
    step "Installing GM1${version:+ $version} with Python $PYTHON_VERSION (about 2.5 GB, mostly NVIDIA's libraries)"
  fi
  if [ -n "$constraints" ]; then
    say "With the exact versions GM1 was tested with."
  else
    say "The list of tested versions could not be downloaded; installing with GM1's own version ranges."
  fi
  # uv warns when its folder is not on PATH; this installer puts it there for new terminals itself.
  bin_dir=$("$UV" tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")
  user_path=$PATH
  PATH="$bin_dir:$PATH" "$UV" "$@"
  status=$?
  [ "$made_constraints" = 1 ] && rm -f "$constraints"
  if [ "$status" -ne 0 ]; then
    fail "the install failed (see above). Rerun the same command; if it fails again, ask for help with the text above."
  fi

  ensure_path "$bin_dir"
  gm1="$bin_dir/gpuwm-cm1"
  [ -x "$gm1" ] || fail "gpuwm-cm1 was installed but is not in $bin_dir."

  if [ "${GM1_SKIP_DOCTOR:-0}" != 1 ]; then
    step "Checking that this computer can run GM1 (gpuwm-cm1 doctor --plain)"
    "$gm1" doctor --plain || true
  fi

  # From the Windows installer, it says what comes next (the GM1 shortcut), in Windows' words.
  [ "${GM1_FROM_WINDOWS:-0}" = 1 ] && return 0
  step "GM1 is installed"
  case ":$user_path:" in
    *":$bin_dir:"*) ;;
    *) say "Open a new terminal first, so it finds gpuwm-cm1 (or type: export PATH=\"$bin_dir:\$PATH\")." ;;
  esac
  say "Start GM1:                gpuwm-cm1 gui"
  say "If something is wrong:    gpuwm-cm1 doctor --report   (paste what it prints when you ask for help)"
  if [ "$WSL" != 0 ]; then
    say "On Windows, the link opens in your Windows browser."
  fi
}

do_uninstall() {
  step "Uninstalling GM1"
  refuse_while_running
  removed=0
  installed=""
  if find_any_uv && "$UV" tool list 2>/dev/null | grep -q '^gpuwm-cm1 '; then
    # Every package GM1's environment holds, so their downloads go from uv's cache too.
    tool_python="$("$UV" tool dir 2>/dev/null)/gpuwm-cm1/bin/python"
    if [ -x "$tool_python" ]; then
      installed=$("$UV" pip list --python "$tool_python" --format freeze 2>/dev/null | sed -n 's/==.*//p' | tr '\n' ' ')
    fi
    "$UV" tool uninstall gpuwm-cm1 || fail "uv could not uninstall gpuwm-cm1 (see above)."
    say "GM1 is uninstalled."
    removed=1
  else
    say "GM1 is not installed with uv here; nothing to uninstall."
  fi
  if [ -n "$UV" ]; then
    # Only GM1's packages: anything else in uv's cache is left alone.
    # shellcheck disable=SC2086 # one word per package
    "$UV" cache clean $CACHED_PACKAGES $installed >/dev/null 2>&1 || true
    [ "$removed" = 1 ] && say "Freed the downloaded packages (about 2.5 GB)."
  fi
  say "Your runs in ~/gm1-runs are kept: delete that folder yourself if you no longer want them."
}

usage() {
  say "GM1 installer: sh install.sh [--uninstall]"
  say "Settings: GM1_VERSION, GM1_WHEEL, GM1_FIND_LINKS, GM1_CONSTRAINTS, GM1_REINSTALL=1, GM1_NO_MODIFY_PATH=1,"
  say "          GM1_SKIP_DOCTOR=1"
}

main() {
  cd "$HOME" 2>/dev/null || true
  case "${1:-}" in
    --uninstall|uninstall) do_uninstall ;;
    -h|--help) usage ;;
    '') do_install ;;
    *) usage; exit 2 ;;
  esac
}

main "$@"
