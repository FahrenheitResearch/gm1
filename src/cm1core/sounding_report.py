"""The t=0 environment report for one Cartesian run, computed in Rust.

CM1 prints this block unconditionally for a moist run: ``base.F:3143-3189``
lifts three parcels -- surface, most-unstable and mixed-layer -- through
``getcape.F`` over the base-state column at ``i=1,j=1`` and writes the source
parcel, the LCL, LFC and EL, and the CAPE and CIN of each.  GM1 does the same,
and adds the three kinematic parameters CM1 has no equivalent for: the bulk
shear over 0-1 and 0-6 km, the Bunkers et al. (2000) internal-dynamics storm
motion, and the storm-relative helicity over 0-1 and 0-3 km taken against it.

The whole computation is Rust (``rust/cm1-state/src/getcape.rs`` and
``sounding_report.rs``) because it is a transform of the base state, and it
runs on the host before any device is opened: a run that cannot describe the
atmosphere it was given should say so before it burns two hours of GPU, not
after.  Nothing here touches CUDA.
"""

from __future__ import annotations

import ctypes
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import refuse


SOUNDING_REPORT_SCHEMA = "gm1.sounding-parameters/v1"
#: Bunkers movers the Rust side implements.  CM1 r21.1 implements none.
STORM_MOTIONS = ("bunkers-right", "bunkers-left")
CM1_REPORT_AUTHORITY = "CM1 r21.1 src/base.F:3143-3189 via src/getcape.F:14-594"


@dataclass(frozen=True, slots=True)
class SoundingReportSelection:
    """``getcape.F``'s own source parameters plus the named storm motion."""

    adiabat: int = 1
    mixed_layer_depth_m: float = 500.0
    pressure_increment_pa: float = 200.0
    storm_motion: str = "bunkers-right"


def _configure(library: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    f32p = ctypes.POINTER(ctypes.c_float)
    library.gm1_sounding_report_json.argtypes = (
        u8p,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_size_t,
        ctypes.c_float,
        f32p,
        f32p,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_float,
        u8p,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_size_t),
    )
    library.gm1_sounding_report_json.restype = u8p
    # The v2 entry carries base.F's words beyond isnd/iwnd (testcase, imoist,
    # the moist benchmark's thec_mb/qt_mb, tsk0); an older library has only v1.
    if hasattr(library, "gm1_sounding_report_json_v2"):
        library.gm1_sounding_report_json_v2.argtypes = (
            *library.gm1_sounding_report_json.argtypes[:-1],
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_size_t),
        )
        library.gm1_sounding_report_json_v2.restype = u8p
    library.gm1_sounding_report_free.argtypes = (u8p, ctypes.c_size_t)
    library.gm1_sounding_report_free.restype = None
    library.gm1_sounding_report_last_error_len.argtypes = ()
    library.gm1_sounding_report_last_error_len.restype = ctypes.c_size_t
    library.gm1_sounding_report_last_error_copy.argtypes = (u8p, ctypes.c_size_t)
    library.gm1_sounding_report_last_error_copy.restype = ctypes.c_size_t


def _last_error(library: ctypes.CDLL) -> str:
    length = int(library.gm1_sounding_report_last_error_len())
    if length == 0:
        return "the Rust sounding report failed without a message"
    buffer = (ctypes.c_uint8 * length)()
    copied = int(
        library.gm1_sounding_report_last_error_copy(
            ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)), length
        )
    )
    return bytes(buffer[:copied]).decode("utf-8", "replace")


def initial_sounding_report(
    library_path: str | Path,
    sounding_bytes: bytes,
    *,
    isnd: int,
    iwnd: int,
    nz: int,
    dz_m: float,
    levels: Any = None,
    selection: SoundingReportSelection = SoundingReportSelection(),
    u_move_m_s: float = 0.0,
    v_move_m_s: float = 0.0,
    testcase: int = 0,
    imoist: int = 1,
    thec_mb_k: float = 320.0,
    qt_mb: float = 0.020,
    tsk0_k: float = 299.28,
    eqtset: int = 2,
) -> dict[str, Any]:
    """Return this run's own t=0 parameters as a JSON-shaped mapping.

    ``levels`` is a :class:`cm1core.cm1_vertical_grid.VerticalLevels` or
    ``None`` for ``param.F:7789-7811``'s uniform coordinate.  ``u_move_m_s``
    and ``v_move_m_s`` are ``base.F:2663-2669``'s frame: the Rust side adds
    them back before reading the hodograph, because a storm motion measured in
    a frame that is already translating with the storm would report a
    storm-relative helicity near zero for any sounding at all.
    """

    if not isinstance(sounding_bytes, bytes):
        raise TypeError("sounding_bytes must be an immutable byte string")
    if selection.storm_motion not in STORM_MOTIONS:
        refuse(
            code="CM1-SOUNDING-REPORT-MOTION",
            gate="support-matrix",
            reason=(
                "storm-relative helicity is a line integral taken against one storm "
                "motion, and CM1 r21.1 computes none anywhere in its source tree; an "
                "unimplemented method would publish a helicity in the receipt that no "
                "reader could reproduce, and the sign of the reported SRH flips "
                "between the right and left mover"
            ),
            remedy="declare diagnostics.storm_motion as bunkers-right or bunkers-left",
            option="diagnostics.storm_motion",
            value=selection.storm_motion,
        )
    path = Path(library_path).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError("the sounding report library must be a regular file")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as error:
        refuse(
            code="CM1-SOUNDING-REPORT-LIBRARY",
            gate="support-matrix",
            reason=(
                f"the cm1-state initializer library cannot be loaded ({error}); the "
                "run would integrate an atmosphere whose CAPE, CIN, LCL, LFC, shear "
                "and helicity the receipt never states, and the same library is what "
                "builds the base state the run would then integrate"
            ),
            remedy="rebuild the Rust initializer library from this source tree",
            option="initializer-library",
            value=str(path),
        )
    if not hasattr(library, "gm1_sounding_report_json"):
        refuse(
            code="CM1-SOUNDING-REPORT-ABI",
            gate="support-matrix",
            reason=(
                "this cm1-state library exports no sounding report, so it predates "
                "the getcape.F transcription; the run would integrate an atmosphere "
                "whose CAPE, CIN, LCL, LFC, shear and helicity the receipt never "
                "states, which is the defect this report exists to end"
            ),
            remedy="rebuild the Rust initializer library from this source tree",
            option="initializer-library",
            value=str(path),
        )
    _configure(library)

    mass_pointer = ctypes.POINTER(ctypes.c_float)()
    face_pointer = ctypes.POINTER(ctypes.c_float)()
    keep: list[Any] = []
    if levels is not None and getattr(levels, "stretch_z", 0) != 0:
        import numpy as np

        mass = np.ascontiguousarray(np.asarray(levels.zh, dtype=np.float32)[1 : nz + 1])
        face = np.ascontiguousarray(np.asarray(levels.zf, dtype=np.float32)[1 : nz + 2])
        keep.extend((mass, face))
        mass_pointer = mass.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        face_pointer = face.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    payload = (ctypes.c_uint8 * max(len(sounding_bytes), 1)).from_buffer_copy(
        sounding_bytes or b"\0"
    )
    motion = selection.storm_motion.encode("utf-8")
    motion_buffer = (ctypes.c_uint8 * len(motion)).from_buffer_copy(motion)
    length = ctypes.c_size_t(0)
    head = (
        ctypes.cast(payload, ctypes.POINTER(ctypes.c_uint8)),
        len(sounding_bytes),
        int(isnd),
        int(iwnd),
        int(nz),
        ctypes.c_float(float(dz_m)),
        mass_pointer,
        face_pointer,
        int(selection.adiabat),
        ctypes.c_float(float(selection.mixed_layer_depth_m)),
        ctypes.c_float(float(selection.pressure_increment_pa)),
        ctypes.cast(motion_buffer, ctypes.POINTER(ctypes.c_uint8)),
        len(motion),
        ctypes.c_float(float(u_move_m_s)),
        ctypes.c_float(float(v_move_m_s)),
    )
    words = (int(testcase), int(imoist), float(thec_mb_k), float(qt_mb), float(tsk0_k), int(eqtset))
    if hasattr(library, "gm1_sounding_report_json_v2"):
        pointer = library.gm1_sounding_report_json_v2(
            *head,
            ctypes.c_int32(words[0]),
            ctypes.c_int32(words[1]),
            ctypes.c_float(words[2]),
            ctypes.c_float(words[3]),
            ctypes.c_float(words[4]),
            ctypes.c_int32(words[5]),
            ctypes.byref(length),
        )
    elif words != (0, 1, 320.0, 0.020, 299.28, 2):
        refuse(
            code="CM1-SOUNDING-REPORT-ABI",
            gate="support-matrix",
            reason=(
                "this cm1-state library's sounding report takes no testcase, imoist or base.F "
                "words, so the column it would report is CM1's default one, not this run's"
            ),
            remedy="rebuild the Rust initializer library from this source tree",
            option="initializer-library",
            value=str(path),
        )
    else:
        pointer = library.gm1_sounding_report_json(*head, ctypes.byref(length))
    del keep
    if not pointer:
        refuse(
            code="CM1-SOUNDING-REPORT",
            gate="support-matrix",
            reason=(
                "the t=0 environment report could not be built from this run's own "
                f"base state: {_last_error(library)}. A run that cannot state the "
                "CAPE, CIN, LCL, LFC, shear and helicity of the atmosphere it was "
                "given would publish frames nobody can interpret"
            ),
            remedy=(
                "correct the initialization selectors, the vertical coordinate or the "
                "supplied input_sounding named in the message above"
            ),
            option="initialization",
            value=f"isnd={isnd}, iwnd={iwnd}, nz={nz}",
        )
    try:
        raw = bytes(bytearray(pointer[index] for index in range(int(length.value))))
    finally:
        library.gm1_sounding_report_free(pointer, length)
    report = json.loads(raw.decode("utf-8"))
    if report.get("schema") != SOUNDING_REPORT_SCHEMA:
        raise ValueError(
            f"the Rust sounding report declared {report.get('schema')!r}, not "
            f"{SOUNDING_REPORT_SCHEMA!r}"
        )
    for name, value in report.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"the t=0 sounding report word {name} is not finite")
    return report


def _fixed(value: Any, digits: int = 1) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def format_sounding_report(report: dict[str, Any]) -> str:
    """The console block ``base.F:3175-3180`` prints, with the hodograph added."""

    lines = [
        "  Thermodynamic properties of the initial sounding (i=1,j=1):",
    ]
    for key, label in (
        ("surface_based", "surface parcel"),
        ("most_unstable", "most-unstable parcel"),
        ("mixed_layer", "mixed-layer parcel"),
    ):
        block = report[key]
        lines.append(f"    for {label}:")
        lines.append(
            "        source p(mb),T(K),qv(g/kg) = "
            f"{_fixed(block['source_pressure_pa'] * 0.01):>10}"
            f"{_fixed(block['source_temperature_k']):>10}"
            f"{_fixed(block['source_qv_kg_per_kg'] * 1000.0, 2):>10}"
        )
        lines.append(
            "        LCL,LFC,EL (m AGL)         = "
            f"{_fixed(block['lcl_m_agl']):>10}"
            f"{_fixed(block['lfc_m_agl']):>10}"
            f"{_fixed(block['el_m_agl']):>10}"
        )
        lines.append(
            "        CAPE,CIN (J/kg)            = "
            f"{_fixed(block['cape_j_per_kg']):>10}"
            f"{_fixed(block['cin_j_per_kg']):>10}"
        )
        if not block["converged"]:
            lines.append(
                "        (getcape.F:383-396 abandoned the moist adiabat here; "
                "the zeros above are not a measurement)"
            )
    motion = report["storm_motion"]
    prefix = "bunkers_right" if motion == "bunkers-right" else "bunkers_left"
    lines.append(f"    hodograph ({motion}, Bunkers et al. 2000 internal dynamics):")
    lines.append(
        "        storm motion u,v (m/s)     = "
        f"{_fixed(report[f'{prefix}_u_m_per_s']):>10}"
        f"{_fixed(report[f'{prefix}_v_m_per_s']):>10}"
    )
    if report.get("bunkers_deviation_undefined"):
        lines.append(
            "        (no 0-0.5 to 5.5-6 km shear: the Bunkers deviation is "
            "undefined, so the motion is the 0-6 km mean wind)"
        )
    lines.append(
        "        0-1,0-6 km shear (m/s)     = "
        f"{_fixed(report['shear_0_1km_m_per_s']):>10}"
        f"{_fixed(report['shear_0_6km_m_per_s']):>10}"
    )
    lines.append(
        "        0-1,0-3 km SRH (m2/s2)     = "
        f"{_fixed(report['srh_0_1km_m2_per_s2']):>10}"
        f"{_fixed(report['srh_0_3km_m2_per_s2']):>10}"
    )
    lines.append(
        "        lowest mass level (m AGL)  = "
        f"{_fixed(report['lowest_mass_level_m_agl']):>10}"
    )
    lines.append(
        "        CIN is getcape.F's positive magnitude of the negative area."
    )
    return "\n".join(lines)


__all__ = [
    "CM1_REPORT_AUTHORITY",
    "SOUNDING_REPORT_SCHEMA",
    "STORM_MOTIONS",
    "SoundingReportSelection",
    "format_sounding_report",
    "initial_sounding_report",
]
