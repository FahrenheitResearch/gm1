"""What GPUs this machine has, how much of their memory is actually free, and
how much RAM and disk the host has.

Everything comes from ``nvidia-smi`` and ``/proc``; nothing here imports CuPy,
opens a CUDA context or needs a GPU.  A machine without ``nvidia-smi`` answers
at once with no GPUs, which is what a login node or a laptop without an NVIDIA
card should see.

Free memory is ``nvidia-smi``'s ``memory.free``: total minus what every
process holds minus what the driver reserves.  On Windows with WSL2 that
includes what the Windows desktop holds (several GB on a card that drives a
display), and the process list is empty because WSL2 cannot see Windows
processes; both are reported as such.

A test, or a user who wants to see what the estimator would say on another
machine, points ``GPUWM_CM1_NVIDIA_SMI`` at a program that answers the two
queries below the way ``nvidia-smi`` does.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

GPUINFO_SCHEMA = "gpuwm-cm1.gpuinfo/v1"

#: A program to run instead of ``nvidia-smi`` (a fixture, or a wrapper);
#: ``none`` means "this machine has no nvidia-smi".
NVIDIA_SMI_ENV = "GPUWM_CM1_NVIDIA_SMI"
#: ``1``/``0``: say the host is (is not) WSL2, for a fixture of the other kind.
WSL2_ENV = "GPUWM_CM1_WSL2"
#: A JSON file with a :meth:`HostInfo.record` to use instead of this host's RAM and
#: disk (a test, or "what would the fit say on that machine"), as the
#: ``nvidia-smi`` fixture does for the GPU.
HOST_ENV = "GPUWM_CM1_HOST_FIXTURE"

#: Where WSL2 puts the Windows driver's ``nvidia-smi`` when ``PATH`` lacks it.
WSL_NVIDIA_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")

#: The full query, then the one old drivers (before ``memory.reserved`` and
#: ``compute_cap`` were queryable) answer.
GPU_FIELDS = (
    "index", "uuid", "name", "memory.total", "memory.used", "memory.free",
    "memory.reserved", "compute_cap", "driver_version", "pci.bus_id",
)
GPU_FIELDS_OLD = (
    "index", "uuid", "name", "memory.total", "memory.used", "memory.free",
    "driver_version", "pci.bus_id",
)
APP_FIELDS = ("gpu_uuid", "pid", "process_name", "used_memory")

MIB = 1024 * 1024

#: How long ``nvidia-smi`` may take.  A healthy one answers in well under a
#: second; one that hangs is a wedged driver, which is reported, not waited on.
DEFAULT_TIMEOUT_S = 8.0


class GpuQueryError(RuntimeError):
    """``nvidia-smi`` ran but its answer could not be read."""


def _number(text: str) -> float | None:
    text = text.strip()
    if not text or text.startswith("[") or text.upper() in {"N/A", "NA"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _mib_bytes(text: str) -> int | None:
    value = _number(text)
    return None if value is None else int(round(value * MIB))


def is_wsl2() -> bool:
    """Windows Subsystem for Linux 2: the kernel says ``microsoft``."""

    forced = os.environ.get(WSL2_ENV)
    if forced in ("0", "1"):
        return forced == "1"
    try:
        release = platform.release().lower()
    except Exception:  # noqa: BLE001 - platform never matters enough to fail on
        return False
    if "microsoft" in release or "wsl" in release:
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


@dataclass(frozen=True)
class ComputeApp:
    """One process ``nvidia-smi`` lists on a GPU."""

    gpu_uuid: str
    pid: int | None
    name: str
    used_bytes: int | None

    def record(self) -> dict[str, Any]:
        return {
            "gpu_uuid": self.gpu_uuid, "pid": self.pid, "name": self.name,
            "used_bytes": self.used_bytes,
        }


@dataclass(frozen=True)
class GpuInfo:
    """One GPU as ``nvidia-smi`` reports it at one moment."""

    index: int
    uuid: str
    name: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    reserved_bytes: int | None = None
    compute_capability: str | None = None
    driver_version: str | None = None
    pci_bus_id: str | None = None
    compute_apps: tuple[ComputeApp, ...] = ()
    wsl2: bool = False
    source: str = "nvidia-smi"

    @property
    def in_use_bytes(self) -> int:
        """Everything not free: processes, a desktop, the driver's reserve."""

        return max(0, self.total_bytes - self.free_bytes)

    @property
    def sm(self) -> str | None:
        """``sm_86`` for compute capability 8.6."""

        if not self.compute_capability:
            return None
        major, _, minor = self.compute_capability.partition(".")
        if not major.isdigit() or not (minor or "0").isdigit():
            return None
        return f"sm_{int(major)}{int(minor or 0)}"

    @property
    def processes_visible(self) -> bool:
        """False under WSL2, whose ``nvidia-smi`` cannot name Windows processes."""

        return not self.wsl2

    def memory_line(self) -> str:
        gb = 1e9
        text = (
            f"{self.free_bytes / gb:.1f} GB free of {self.total_bytes / gb:.1f} GB"
            f" ({self.in_use_bytes / gb:.1f} GB in use"
        )
        if self.wsl2 and not self.compute_apps and self.in_use_bytes > 0:
            text += ", which under WSL2 includes the Windows desktop"
        return text + ")"

    def record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "name": self.name,
            "total_bytes": self.total_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
            "reserved_bytes": self.reserved_bytes,
            "in_use_bytes": self.in_use_bytes,
            "compute_capability": self.compute_capability,
            "sm": self.sm,
            "driver_version": self.driver_version,
            "pci_bus_id": self.pci_bus_id,
            "compute_apps": [app.record() for app in self.compute_apps],
            "processes_visible": self.processes_visible,
            "wsl2": self.wsl2,
            "source": self.source,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "GpuInfo":
        """The inverse of :meth:`record` (a stored estimate, a GUI request)."""

        return cls(
            index=int(record.get("index", 0)),
            uuid=str(record.get("uuid", "")),
            name=str(record["name"]),
            total_bytes=int(record["total_bytes"]),
            used_bytes=int(record.get("used_bytes", record["total_bytes"] - record["free_bytes"])),
            free_bytes=int(record["free_bytes"]),
            reserved_bytes=record.get("reserved_bytes"),
            compute_capability=record.get("compute_capability"),
            driver_version=record.get("driver_version"),
            pci_bus_id=record.get("pci_bus_id"),
            compute_apps=tuple(
                ComputeApp(
                    str(app.get("gpu_uuid", "")), app.get("pid"), str(app.get("name", "")),
                    app.get("used_bytes"),
                )
                for app in record.get("compute_apps", ())
            ),
            wsl2=bool(record.get("wsl2", False)),
            source=str(record.get("source", "record")),
        )


@dataclass(frozen=True)
class GpuQuery:
    """What one ``nvidia-smi`` query found."""

    status: str  # "ok" | "no-nvidia-smi" | "error"
    gpus: tuple[GpuInfo, ...] = ()
    message: str = ""
    nvidia_smi: str | None = None
    elapsed_s: float = 0.0
    wsl2: bool = False

    def select(self, index: int | None = None) -> GpuInfo | None:
        """GPU ``index`` (``nvidia-smi``'s numbering), or the first one."""

        if not self.gpus:
            return None
        if index is None:
            return self.gpus[0]
        for gpu in self.gpus:
            if gpu.index == index:
                return gpu
        raise ValueError(
            f"there is no GPU {index}; nvidia-smi lists "
            + ", ".join(f"{gpu.index} ({gpu.name})" for gpu in self.gpus)
        )

    def record(self) -> dict[str, Any]:
        return {
            "schema": GPUINFO_SCHEMA,
            "status": self.status,
            "message": self.message,
            "nvidia_smi": self.nvidia_smi,
            "elapsed_s": round(self.elapsed_s, 3),
            "wsl2": self.wsl2,
            "gpus": [gpu.record() for gpu in self.gpus],
        }


def locate_nvidia_smi(explicit: str | os.PathLike[str] | None = None) -> str | None:
    """``explicit``, then ``$GPUWM_CM1_NVIDIA_SMI``, ``PATH``, WSL2's copy."""

    for candidate in (explicit, os.environ.get(NVIDIA_SMI_ENV)):
        if candidate:
            return None if str(candidate).lower() == "none" else str(candidate)
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if WSL_NVIDIA_SMI.is_file() and os.access(WSL_NVIDIA_SMI, os.X_OK):
        return str(WSL_NVIDIA_SMI)
    return None


def _run(program: str, arguments: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [program, *arguments], capture_output=True, text=True,
        timeout=timeout_s, check=False,
        # nvidia-smi reads no CUDA_VISIBLE_DEVICES, but a fixture might.
        env={**os.environ, "LC_ALL": "C"},
    )


def _csv_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        rows.append([cell.strip() for cell in line.split(",")])
    return rows


def parse_gpu_rows(
    text: str, fields: Sequence[str], *, wsl2: bool, source: str,
    apps: Sequence[ComputeApp] = (),
) -> tuple[GpuInfo, ...]:
    """``--query-gpu`` CSV (``noheader,nounits``) as :class:`GpuInfo` rows."""

    gpus = []
    for row in _csv_rows(text):
        if len(row) < len(fields):
            # A GPU name never holds a comma in nvidia-smi's own table, so a
            # short row is a truncated answer, not a longer name.
            raise GpuQueryError(f"nvidia-smi answered {len(row)} fields, expected {len(fields)}: {row}")
        if len(row) > len(fields):
            # Defensive: fold any extra commas into the name column.
            extra = len(row) - len(fields)
            name_at = list(fields).index("name")
            row = row[:name_at] + [",".join(row[name_at:name_at + extra + 1])] + row[name_at + extra + 1:]
        cells = dict(zip(fields, row))
        total = _mib_bytes(cells["memory.total"])
        used = _mib_bytes(cells["memory.used"])
        free = _mib_bytes(cells["memory.free"])
        if total is None:
            raise GpuQueryError(f"nvidia-smi gave no memory.total for GPU {cells.get('index')}")
        reserved = _mib_bytes(cells["memory.reserved"]) if "memory.reserved" in cells else None
        if free is None and used is not None:
            free = max(0, total - used - (reserved or 0))
        if used is None and free is not None:
            used = max(0, total - free - (reserved or 0))
        if free is None or used is None:
            raise GpuQueryError(f"nvidia-smi gave no memory.used/free for GPU {cells.get('index')}")
        index_value = _number(cells["index"])
        uuid = cells["uuid"]
        capability = cells.get("compute_cap")
        if capability is not None and _number(capability) is None:
            capability = None
        gpus.append(GpuInfo(
            index=int(index_value) if index_value is not None else len(gpus),
            uuid=uuid,
            name=cells["name"],
            total_bytes=total,
            used_bytes=used,
            free_bytes=free,
            reserved_bytes=reserved,
            compute_capability=capability,
            driver_version=cells.get("driver_version") or None,
            pci_bus_id=cells.get("pci.bus_id") or None,
            compute_apps=tuple(app for app in apps if app.gpu_uuid == uuid),
            wsl2=wsl2,
            source=source,
        ))
    return tuple(gpus)


def parse_app_rows(text: str) -> tuple[ComputeApp, ...]:
    """``--query-compute-apps`` CSV (``noheader,nounits``)."""

    apps = []
    for row in _csv_rows(text):
        if len(row) < len(APP_FIELDS):
            continue
        uuid, pid, name, used = row[0], row[1], ",".join(row[2:-1]), row[-1]
        pid_value = _number(pid)
        apps.append(ComputeApp(
            gpu_uuid=uuid,
            pid=int(pid_value) if pid_value is not None else None,
            name=name,
            used_bytes=_mib_bytes(used),
        ))
    return tuple(apps)


def query_gpus(
    *, nvidia_smi: str | os.PathLike[str] | None = None, timeout_s: float = DEFAULT_TIMEOUT_S,
) -> GpuQuery:
    """Ask ``nvidia-smi`` once for every GPU and every process on them."""

    started = time.perf_counter()
    wsl2 = is_wsl2()
    program = locate_nvidia_smi(nvidia_smi)
    if program is None:
        return GpuQuery(
            "no-nvidia-smi",
            message="No NVIDIA GPU found: nvidia-smi is not installed or not on PATH.",
            elapsed_s=time.perf_counter() - started, wsl2=wsl2,
        )
    source = program
    try:
        fields: Sequence[str] = GPU_FIELDS
        answer = _run(program, [f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"], timeout_s)
        if answer.returncode != 0 and "not a valid field" in (answer.stdout + answer.stderr).lower():
            fields = GPU_FIELDS_OLD
            answer = _run(program, [f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"], timeout_s)
        if answer.returncode != 0:
            detail = (answer.stderr or answer.stdout).strip().splitlines()
            return GpuQuery(
                "error",
                message=(
                    "nvidia-smi could not list the GPUs"
                    + (f": {detail[-1]}" if detail else f" (exit {answer.returncode})")
                ),
                nvidia_smi=program, elapsed_s=time.perf_counter() - started, wsl2=wsl2,
            )
        apps_answer = _run(
            program, [f"--query-compute-apps={','.join(APP_FIELDS)}", "--format=csv,noheader,nounits"],
            timeout_s,
        )
        apps = parse_app_rows(apps_answer.stdout) if apps_answer.returncode == 0 else ()
        gpus = parse_gpu_rows(answer.stdout, fields, wsl2=wsl2, source=source, apps=apps)
    except subprocess.TimeoutExpired:
        return GpuQuery(
            "error",
            message=f"nvidia-smi did not answer within {timeout_s:.0f} s; the driver may be stuck.",
            nvidia_smi=program, elapsed_s=time.perf_counter() - started, wsl2=wsl2,
        )
    except (OSError, GpuQueryError) as error:
        return GpuQuery(
            "error", message=f"nvidia-smi could not be read: {error}",
            nvidia_smi=program, elapsed_s=time.perf_counter() - started, wsl2=wsl2,
        )
    if not gpus:
        return GpuQuery(
            "ok", gpus=(),
            message="nvidia-smi lists no GPUs.", nvidia_smi=program,
            elapsed_s=time.perf_counter() - started, wsl2=wsl2,
        )
    return GpuQuery(
        "ok", gpus=gpus, nvidia_smi=program,
        elapsed_s=time.perf_counter() - started, wsl2=wsl2,
    )


@dataclass(frozen=True)
class HostInfo:
    """The host's memory and the free disk under the runs folder."""

    ram_total_bytes: int | None
    ram_available_bytes: int | None
    disk_root: str
    disk_total_bytes: int | None
    disk_free_bytes: int | None
    cpu_count: int | None = None
    wsl2: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {
            "ram_total_bytes": self.ram_total_bytes,
            "ram_available_bytes": self.ram_available_bytes,
            "disk_root": self.disk_root,
            "disk_total_bytes": self.disk_total_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "cpu_count": self.cpu_count,
            "wsl2": self.wsl2,
            **dict(self.extra),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "HostInfo":
        return cls(
            ram_total_bytes=record.get("ram_total_bytes"),
            ram_available_bytes=record.get("ram_available_bytes"),
            disk_root=str(record.get("disk_root", ".")),
            disk_total_bytes=record.get("disk_total_bytes"),
            disk_free_bytes=record.get("disk_free_bytes"),
            cpu_count=record.get("cpu_count"),
            wsl2=bool(record.get("wsl2", False)),
        )


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        text = Path("/proc/meminfo").read_text(encoding="ascii")
    except OSError:
        return values
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        words = rest.split()
        if words and words[0].isdigit():
            scale = 1024 if len(words) > 1 and words[1].lower() == "kb" else 1
            values[name.strip()] = int(words[0]) * scale
    return values


def query_host(root: str | os.PathLike[str] = ".") -> HostInfo:
    """RAM from ``/proc/meminfo``; disk from the nearest existing folder of ``root``.

    ``$GPUWM_CM1_HOST_FIXTURE`` names a JSON host record to report instead.
    """

    path = Path(root).expanduser()
    fixture = os.environ.get(HOST_ENV)
    if fixture:
        record = json.loads(Path(fixture).read_text(encoding="utf-8"))
        return HostInfo.from_record({**record, "disk_root": str(path)})
    info = _meminfo()
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
        total, free = int(usage.total), int(usage.free)
    except OSError:
        total = free = None
    return HostInfo(
        ram_total_bytes=info.get("MemTotal"),
        ram_available_bytes=info.get("MemAvailable"),
        disk_root=str(path),
        disk_total_bytes=total,
        disk_free_bytes=free,
        cpu_count=os.cpu_count(),
        wsl2=is_wsl2(),
    )


__all__ = [
    "APP_FIELDS",
    "ComputeApp",
    "GPUINFO_SCHEMA",
    "GPU_FIELDS",
    "GPU_FIELDS_OLD",
    "GpuInfo",
    "GpuQuery",
    "GpuQueryError",
    "HOST_ENV",
    "HostInfo",
    "NVIDIA_SMI_ENV",
    "WSL2_ENV",
    "is_wsl2",
    "locate_nvidia_smi",
    "parse_app_rows",
    "parse_gpu_rows",
    "query_gpus",
    "query_host",
]
