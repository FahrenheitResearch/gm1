"""Zero-disk views of the Rust-owned CM1 idealized carrier map.

The Rust initializer owns every scientific transform.  This module only
loads its narrow C ABI, exposes immutable NumPy views while the opaque owner
is alive, and optionally makes explicit host copies for CUDA staging.  It is
not an initialized-product, checkpoint, restart, or forecast format.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


from .gm1_runtime import MAX_FRAME_SPEED_M_S

TERRAIN_OPEN_GROUP = 0
BASE_WIND_GROUP = 1
MOIST_GROUP = 2
TERRAIN_STATIC_GROUP = 3
#: base.F's qc0/qi0 (padded mass layout); zero unless the sounding carries
#: base-state condensate (isnd = 4, 9, 10, 11).
BASE_CONDENSATE_GROUP = 4
GROUP_NAMES = MappingProxyType(
    {
        TERRAIN_OPEN_GROUP: "terrain_open",
        BASE_WIND_GROUP: "base_wind",
        MOIST_GROUP: "moist",
        TERRAIN_STATIC_GROUP: "terrain_static",
        BASE_CONDENSATE_GROUP: "base_condensate",
    }
)
EXPECTED_GROUP_INVENTORY = MappingProxyType(
    {
        "base_wind": ("u0", "v0"),
        "base_condensate": ("qc0", "qi0"),
        "moist": ("qa", "q3d", "qten", "qten1"),
        "terrain_static": (
            "terrain_msl",
            "zh",
            "zf",
            "mh",
            "rmh",
            "mf",
            "rmf",
            "c1",
            "c2",
            "rho0_3d",
            "rr0",
            "rf0",
            "rrf0",
            "gz",
            "rgz",
            "gzu",
            "rgzu",
            "gzv",
            "rgzv",
            "dzdx",
            "dzdy",
            "gx",
            "gxu",
            "gy",
            "gyv",
        ),
    }
)
_SIZE_MAX = ctypes.c_size_t(-1).value


class Gm1IdealizedOptions(ctypes.Structure):
    """``rust/cm1-state/src/idealized_ffi.rs``'s complete selector struct.

    Every field is one of CM1's own namelist or named-parameter words.  It is
    one extensible ABI rather than a longer positional signature, so a new
    CM1 selector is added by growing this struct on both sides.
    """

    _fields_ = [
        ("isnd", ctypes.c_int32),
        ("iwnd", ctypes.c_int32),
        ("iinit", ctypes.c_int32),
        ("irst", ctypes.c_int32),
        ("imoist", ctypes.c_int32),
        ("testcase", ctypes.c_int32),
        ("ibalance", ctypes.c_int32),
        ("irandp", ctypes.c_int32),
        ("random_seed", ctypes.c_uint64),
        ("hurr_vg_m_per_s", ctypes.c_float),
        ("hurr_rotate_deg", ctypes.c_float),
        ("ric_m", ctypes.c_float),
        ("rjc_m", ctypes.c_float),
        ("zc_m", ctypes.c_float),
        ("bhrad_m", ctypes.c_float),
        ("bvrad_m", ctypes.c_float),
        ("bptpert_k", ctypes.c_float),
        ("maintain_rh", ctypes.c_int32),
        ("nbub", ctypes.c_int32),
        ("bubble_spacing_y_m", ctypes.c_float),
        ("zdep_m", ctypes.c_float),
        ("random_line_amplitude_k", ctypes.c_float),
        ("umove_m_per_s", ctypes.c_float),
        ("vmove_m_per_s", ctypes.c_float),
        # 0: GM1's seeded stream; 1: CM1's own (init3d.F:427-440, libgfortran).
        ("random_generator", ctypes.c_int32),
        # init3d.F:697-1256 (iinit=7): fcor, the periodic images and iice.
        ("fcor_s_inv", ctypes.c_float),
        ("periodic_x", ctypes.c_int32),
        ("periodic_y", ctypes.c_int32),
        ("iice", ctypes.c_int32),
        # base.F:267-268 (isnd=4, init3d.F's iinit=4) and the namelist tsk0
        # (isnd=11, 21).
        ("thec_mb_k", ctypes.c_float),
        ("qt_mb", ctypes.c_float),
        ("tsk0_k", ctypes.c_float),
        # param.F:5178-5205 (ibalance=2's poiss): mirror x / y.
        ("imirror", ctypes.c_int32),
        ("jmirror", ctypes.c_int32),
        # CM1's eqtset after param.F:936-950 (base.F's isnd=10 reads it).
        ("eqtset", ctypes.c_int32),
    ]


#: ``gm1_idealized_state_selector`` codes, matching the Rust doc comment.
IDEALIZED_SELECTOR_CODES: Mapping[str, int] = MappingProxyType(
    {
        "isnd": 0,
        "effective_iwnd": 1,
        "requested_iwnd": 2,
        "requested_iinit": 3,
        "effective_iinit": 4,
        "irst": 5,
        "imoist": 6,
        "testcase": 7,
        "ibalance": 8,
        "irandp": 9,
    }
)


@dataclass(frozen=True, slots=True)
class IdealizedGeometry:
    shape_zyx: tuple[int, int, int]
    spacing_m: tuple[float, float, float]
    origin_xy_m: tuple[float, float]
    bubble_center_xy_m: tuple[float, float]

    def validate(self) -> None:
        nz, ny, nx = self.shape_zyx
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (nz, ny, nx)):
            raise TypeError("idealized shape extents must be integers")
        if nz < 4 or ny < 1 or nx < 1:
            raise ValueError("idealized shape requires nz>=4 and positive x/y extents")
        for label, values in (
            ("spacing", self.spacing_m),
            ("origin", self.origin_xy_m),
            ("bubble center", self.bubble_center_xy_m),
        ):
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
                raise TypeError(f"idealized {label} values must be numeric")
        if any(float(value) <= 0.0 for value in self.spacing_m):
            raise ValueError("idealized spacing must be positive")


def _configure(library: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    usizep = ctypes.POINTER(ctypes.c_size_t)
    f32p = ctypes.POINTER(ctypes.c_float)
    f32pp = ctypes.POINTER(f32p)
    usizepp = ctypes.POINTER(usizep)
    library.gm1_idealized_state_new.argtypes = (
        u8p,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        # base.F:2663-2669 umove/vmove.
        ctypes.c_float,
        ctypes.c_float,
    )
    library.gm1_idealized_state_new.restype = ctypes.c_void_p
    # The analytic base.F lane returns the same opaque owner, so every carrier
    # accessor and the free below apply to it unchanged.  Older libraries do
    # not export it; its absence is what tells the caller the lane is missing.
    if hasattr(library, "gm1_analytic_state_new"):
        library.gm1_analytic_state_new.argtypes = (
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            # base.F:2663-2669 umove/vmove.
            ctypes.c_float,
            ctypes.c_float,
        )
        library.gm1_analytic_state_new.restype = ctypes.c_void_p
    # param.F:7814-8139.  The two level entry points take zh(1..nk) and
    # zf(1..nk+1) where the flat ones derive them from dz, and return the same
    # opaque owner.  An older library exports neither; a stretched request is
    # then refused rather than silently built on param.F:7789-7811's uniform
    # coordinate, which would put the first scalar level at 0.5*dz.
    for name in ("gm1_idealized_state_new_levels", "gm1_analytic_state_new_levels"):
        entry = getattr(library, name, None)
        if entry is None:
            continue
        head = (
            (u8p, ctypes.c_size_t, ctypes.c_int32)
            if name.startswith("gm1_idealized")
            else (ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)
        )
        entry.argtypes = (
            *head,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            f32p,
            ctypes.c_size_t,
            f32p,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            # base.F:2663-2669 umove/vmove.
            ctypes.c_float,
            ctypes.c_float,
        )
        entry.restype = ctypes.c_void_p
    # The frame readback is how a caller proves the loaded library honoured
    # the translation it asked for.  An older library exports neither name,
    # and a nonzero frame is refused rather than passed to a callee that
    # would ignore the two extra registers and build an untranslated state.
    for name in ("gm1_idealized_state_umove", "gm1_idealized_state_vmove"):
        if hasattr(library, name):
            entry = getattr(library, name)
            entry.argtypes = (ctypes.c_void_p,)
            entry.restype = ctypes.c_float
    # The one extensible entry point.  An older library exports none of these
    # three names; a caller that needs a selector beyond the historical
    # (isnd, iwnd, iinit) triple is then refused by name rather than silently
    # built on CM1's defaults for everything it asked for.
    entry = getattr(library, "gm1_idealized_state_new_options", None)
    if entry is not None:
        entry.argtypes = (
            u8p,
            ctypes.c_size_t,
            ctypes.POINTER(Gm1IdealizedOptions),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_float,
            ctypes.c_float,
            f32p,
            ctypes.c_size_t,
            f32p,
            ctypes.c_size_t,
        )
        entry.restype = ctypes.c_void_p
    for name, restype in (
        ("gm1_idealized_state_selector", ctypes.c_int32),
        ("gm1_idealized_state_random_seed", ctypes.c_uint64),
    ):
        entry = getattr(library, name, None)
        if entry is None:
            continue
        entry.argtypes = (
            (ctypes.c_void_p, ctypes.c_uint32)
            if name.endswith("selector")
            else (ctypes.c_void_p,)
        )
        entry.restype = restype
    library.gm1_idealized_state_free.argtypes = (ctypes.c_void_p,)
    library.gm1_idealized_state_free.restype = None
    library.gm1_idealized_carrier_count.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    library.gm1_idealized_carrier_count.restype = ctypes.c_size_t
    library.gm1_idealized_carrier_view.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_size_t,
        usizep,
        usizepp,
        usizep,
        f32pp,
        usizep,
    )
    library.gm1_idealized_carrier_view.restype = u8p
    library.gm1_idealized_last_error_len.argtypes = ()
    library.gm1_idealized_last_error_len.restype = ctypes.c_size_t
    library.gm1_idealized_last_error_copy.argtypes = (u8p, ctypes.c_size_t)
    library.gm1_idealized_last_error_copy.restype = ctypes.c_size_t


def _last_error(library: ctypes.CDLL) -> str:
    count = int(library.gm1_idealized_last_error_len())
    if count <= 0:
        return "Rust idealized bridge failed without an error message"
    buffer = (ctypes.c_uint8 * count)()
    actual = int(library.gm1_idealized_last_error_copy(buffer, count))
    if actual != count:
        return "Rust idealized bridge error changed while being copied"
    return bytes(buffer).decode("utf-8", errors="strict")


def _declared_levels(
    library: ctypes.CDLL, levels: object | None, *, nz: int, dz_m: float
) -> tuple[object, object] | None:
    """``(zh(1..nk), zf(1..nk+1))`` as C-contiguous f32, or ``None``.

    ``None`` means "take the entry point that derives the coordinate from
    ``dz``", which is what ``param.F:7789-7811`` does and what every uniform
    caller has always reached.
    """

    if levels is None:
        return None
    import numpy as np

    from .cm1_vertical_grid import VerticalLevels

    if not isinstance(levels, VerticalLevels):
        raise TypeError("levels must be cm1_vertical_grid.VerticalLevels")
    if levels.nk != nz:
        raise ValueError("levels must carry exactly the geometry's nz scalar levels")
    if float(np.float32(levels.dz_reference_m)) != float(np.float32(dz_m)):
        raise ValueError(
            "levels dz reference must equal the geometry's dz; param.F:8170 "
            "divides every mh word by exactly one number"
        )
    if levels.stretch_z == 0:
        # The uniform coordinate reaches the established entry point, so an
        # unstretched caller that passes its own levels gets the same raw
        # words and the same receipt it got before this argument existed.
        return None
    for name in ("gm1_idealized_state_new_levels", "gm1_analytic_state_new_levels"):
        if not hasattr(library, name):
            raise ValueError(
                "this cm1-state library exports no declared-vertical initializer, "
                "so it predates param.F:7814-8139's stretched grid and would "
                "ignore the levels: the state would be built on the uniform "
                f"coordinate with its first scalar level at {0.5 * dz_m} m "
                f"instead of {levels.lowest_mass_level_m} m, and every base-state "
                "word, warm bubble height and hodograph sample would sit at the "
                "wrong altitude with nothing to say so"
            )
    mass = np.ascontiguousarray(np.asarray(levels.zh, dtype=np.float32)[1 : nz + 1])
    face = np.ascontiguousarray(np.asarray(levels.zf, dtype=np.float32)[1 : nz + 2])
    return mass, face


#: Every word of :class:`Gm1IdealizedOptions` that is not one of the three
#: historical positional selectors or the two frame velocities, with CM1's own
#: distributed value.  ``ric_m``/``rjc_m`` are ``None`` for CM1's
#: ``centerx``/``centery`` (``init3d.F:456-457``), which the geometry supplies.
_OPTION_DEFAULTS: Mapping[str, object] = MappingProxyType(
    {
        "irst": 0,
        "imoist": 1,
        "testcase": 0,
        "ibalance": 0,
        "irandp": 0,
        "random_seed": 0,
        "hurr_vg_m_per_s": 40.0,
        "hurr_rotate_deg": 0.0,
        "ric_m": None,
        "rjc_m": None,
        "zc_m": 1400.0,
        "bhrad_m": 10000.0,
        "bvrad_m": 1400.0,
        "bptpert_k": 1.0,
        "maintain_rh": False,
        "nbub": 3,
        "bubble_spacing_y_m": 30000.0,
        "zdep_m": 2500.0,
        "random_line_amplitude_k": 0.20,
        "random_generator": 0,
        "fcor_s_inv": 5.0e-5,
        "periodic_x": 0,
        "periodic_y": 0,
        "iice": 1,
        "thec_mb_k": 320.0,
        "qt_mb": 0.020,
        "tsk0_k": 299.28,
        "imirror": 1,
        "jmirror": 1,
        "eqtset": 2,
    }
)


def _options_struct(
    options: Mapping[str, object],
    isnd: int,
    iwnd: int,
    iinit: int,
    u_move_m_s: float,
    v_move_m_s: float,
    geometry: IdealizedGeometry,
) -> Gm1IdealizedOptions:
    """Fill the C selector struct, refusing any key the ABI has no field for."""

    unknown = set(options) - set(_OPTION_DEFAULTS)
    if unknown:
        raise ValueError(
            "the idealized selector ABI has no field for "
            + ", ".join(sorted(unknown))
            + "; a silently dropped selector would build a state the receipt misnames"
        )
    resolved = {**_OPTION_DEFAULTS, **options}
    # init3d.F:456-457: ric/rjc default to centerx/centery.  The geometry
    # already carries the domain centre the caller placed the bubble at.
    ric = resolved["ric_m"]
    rjc = resolved["rjc_m"]
    centre_x, centre_y = geometry.bubble_center_xy_m
    return Gm1IdealizedOptions(
        isnd=isnd,
        iwnd=iwnd,
        iinit=iinit,
        irst=int(resolved["irst"]),
        imoist=int(resolved["imoist"]),
        testcase=int(resolved["testcase"]),
        ibalance=int(resolved["ibalance"]),
        irandp=int(resolved["irandp"]),
        random_seed=int(resolved["random_seed"]),
        hurr_vg_m_per_s=float(resolved["hurr_vg_m_per_s"]),
        hurr_rotate_deg=float(resolved["hurr_rotate_deg"]),
        ric_m=float(centre_x if ric is None else ric),
        rjc_m=float(centre_y if rjc is None else rjc),
        zc_m=float(resolved["zc_m"]),
        bhrad_m=float(resolved["bhrad_m"]),
        bvrad_m=float(resolved["bvrad_m"]),
        bptpert_k=float(resolved["bptpert_k"]),
        maintain_rh=int(bool(resolved["maintain_rh"])),
        nbub=int(resolved["nbub"]),
        bubble_spacing_y_m=float(resolved["bubble_spacing_y_m"]),
        zdep_m=float(resolved["zdep_m"]),
        random_line_amplitude_k=float(resolved["random_line_amplitude_k"]),
        umove_m_per_s=float(u_move_m_s),
        vmove_m_per_s=float(v_move_m_s),
        random_generator=int(resolved["random_generator"]),
        fcor_s_inv=float(resolved["fcor_s_inv"]),
        periodic_x=int(resolved["periodic_x"]),
        periodic_y=int(resolved["periodic_y"]),
        iice=int(resolved["iice"]),
        thec_mb_k=float(resolved["thec_mb_k"]),
        qt_mb=float(resolved["qt_mb"]),
        tsk0_k=float(resolved["tsk0_k"]),
        imirror=int(resolved["imirror"]),
        jmirror=int(resolved["jmirror"]),
        eqtset=int(resolved["eqtset"]),
    )


class RustIdealizedCarrierMap:
    """Opaque Rust owner plus immutable, zero-copy NumPy carrier views."""

    def __init__(
        self,
        library_path: str | Path,
        sounding_bytes: bytes,
        *,
        iinit: int,
        geometry: IdealizedGeometry,
        isnd: int = 7,
        iwnd: int = 0,
        u_move_m_s: float = 0.0,
        v_move_m_s: float = 0.0,
        levels: object | None = None,
        options: Mapping[str, object] | None = None,
    ) -> None:
        """``levels`` is a :class:`cm1core.cm1_vertical_grid.VerticalLevels`.

        Left unset, or set to the ``stretch_z=0`` member of that family, the
        state is built on ``param.F:7789-7811``'s uniform coordinate through
        exactly the entry point it has always used.  Any other coordinate is
        handed to Rust as ``zh(1..nk)`` and ``zf(1..nk+1)``.
        """

        import numpy as np

        geometry.validate()
        if isinstance(iinit, bool) or not isinstance(iinit, int):
            raise TypeError("idealized bridge iinit must be an int")
        if options is None and iinit not in (0, 1):
            # The historical positional entry points carry (isnd, iwnd, iinit)
            # and nothing else, so only init3d.F's two coordinate-free arms
            # reach them.  Every other arm needs the option struct; asking for
            # one here would build a warm bubble and call it something else.
            raise ValueError(
                "the positional idealized bridge admits iinit=0 or iinit=1 only; "
                "pass options= to reach init3d.F:450-1416's other arms"
            )
        # CM1 base.F:2663-2669.  Refused here as well as in Rust so a bad
        # value never reaches the ABI at all; the Rust refusal carries the
        # authoritative wording and this one carries the Python key names.
        frame: list[float] = []
        for key, value in (("u_move_m_s", u_move_m_s), ("v_move_m_s", v_move_m_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"idealized bridge {key} must be a float in m/s")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(
                    f"idealized bridge {key}={value!r} is not finite; base.F:2667 "
                    "subtracts it from every base-state wind word, so u0/v0 and the "
                    "ua/va copied from them would be non-finite at every level and "
                    "the first large step would integrate a domain of NaN"
                )
            if abs(value) > MAX_FRAME_SPEED_M_S:
                raise ValueError(
                    f"idealized bridge {key}={value!r} m/s exceeds the "
                    f"{MAX_FRAME_SPEED_M_S} m/s this lane admits; the frame velocity "
                    "is a storm motion, and base.F:2667 would replace the declared "
                    "hodograph with a uniform frame-relative jet through the whole depth"
                )
            frame.append(value)
        u_move_m_s, v_move_m_s = frame
        if not isinstance(sounding_bytes, bytes):
            raise TypeError("sounding_bytes must be an immutable byte string")
        if isnd in (7, 17) and not sounding_bytes:
            # Only the isnd=7 lane reads a profile from bytes.  base.F builds
            # the analytic lanes from constants and opens no file, so requiring
            # bytes there would force a caller to supply a sounding that is
            # then ignored, and an ignored input is worse than no input.
            raise TypeError(
                "the single-sounding lane requires one nonempty input_sounding"
            )
        path = Path(library_path).resolve(strict=True)
        if path.is_symlink() or not path.is_file():
            raise ValueError("idealized Rust library must be a regular non-symlink file")
        library = ctypes.CDLL(str(path))
        _configure(library)
        if (u_move_m_s or v_move_m_s) and not (
            hasattr(library, "gm1_idealized_state_umove")
            and hasattr(library, "gm1_idealized_state_vmove")
        ):
            raise ValueError(
                "this cm1-state library exports no moving-frame readback, so it "
                "predates the base.F:2663-2669 translating frame and would ignore "
                f"u_move_m_s={u_move_m_s!r} / v_move_m_s={v_move_m_s!r}: the two "
                "extra arguments would sit unread in registers, the state would be "
                "built in the ground-relative frame, and every published position "
                "would be wrong by u_move_m_s*t with nothing to say so"
            )
        sounding = (ctypes.c_uint8 * len(sounding_bytes)).from_buffer_copy(sounding_bytes)
        nz, ny, nx = geometry.shape_zyx
        dx, dy, dz = geometry.spacing_m
        ox, oy = geometry.origin_xy_m
        bx, by = geometry.bubble_center_xy_m
        declared = _declared_levels(library, levels, nz=nz, dz_m=dz)
        if options is not None:
            entry = getattr(library, "gm1_idealized_state_new_options", None)
            if entry is None:
                raise ValueError(
                    "this cm1-state library exports no gm1_idealized_state_new_options, "
                    "so it predates the full base.F/init3d.F selector set and would "
                    "ignore every word beyond (isnd, iwnd, iinit): the state would be "
                    "built with CM1's defaults for ibalance, irandp, testcase, imoist "
                    "and the whole bubble parameter block while the receipt named the "
                    "declared values"
                )
            struct = _options_struct(options, isnd, iwnd, iinit, u_move_m_s, v_move_m_s, geometry)
            f32p = ctypes.POINTER(ctypes.c_float)
            if declared is None:
                mass_ptr, mass_len, face_ptr, face_len = None, 0, None, 0
            else:
                mass, face = declared
                mass_ptr, mass_len = mass.ctypes.data_as(f32p), mass.size
                face_ptr, face_len = face.ctypes.data_as(f32p), face.size
            handle = entry(
                sounding if sounding_bytes else None,
                len(sounding_bytes),
                ctypes.byref(struct),
                nz,
                ny,
                nx,
                dx,
                dy,
                dz,
                ox,
                oy,
                mass_ptr,
                mass_len,
                face_ptr,
                face_len,
            )
        elif declared is not None:
            mass, face = declared
            f32p = ctypes.POINTER(ctypes.c_float)
            head = (
                (sounding, len(sounding_bytes), iinit)
                if (isnd, iwnd) == (7, 0)
                else (isnd, iwnd, iinit)
            )
            entry = (
                library.gm1_idealized_state_new_levels
                if (isnd, iwnd) == (7, 0)
                else library.gm1_analytic_state_new_levels
            )
            handle = entry(
                *head, nz, ny, nx, dx, dy, dz,
                mass.ctypes.data_as(f32p), mass.size,
                face.ctypes.data_as(f32p), face.size,
                ox, oy, bx, by, u_move_m_s, v_move_m_s,
            )
        elif (isnd, iwnd) == (7, 0):
            handle = library.gm1_idealized_state_new(
                sounding,
                len(sounding_bytes),
                iinit,
                nz,
                ny,
                nx,
                dx,
                dy,
                dz,
                ox,
                oy,
                bx,
                by,
                u_move_m_s,
                v_move_m_s,
            )
        else:
            # base.F builds these profiles analytically and reads no sounding
            # file, so the bytes are not passed: an analytic run that silently
            # inherited a file's winds would not be the lane that was asked for.
            if not hasattr(library, "gm1_analytic_state_new"):
                raise ValueError(
                    f"this cm1-state library exports no analytic initializer, so "
                    f"the CM1 base.F isnd={isnd} / iwnd={iwnd} lane cannot be "
                    "built; it would silently fall back to the single-sounding "
                    "base state and integrate an atmosphere nobody asked for"
                )
            handle = library.gm1_analytic_state_new(
                isnd,
                iwnd,
                iinit,
                nz,
                ny,
                nx,
                dx,
                dy,
                dz,
                ox,
                oy,
                bx,
                by,
                u_move_m_s,
                v_move_m_s,
            )
        if not handle:
            raise ValueError(_last_error(library))
        self._library = library
        self._handle = ctypes.c_void_p(handle)
        self._views: Mapping[str, Mapping[str, object]]
        groups: dict[str, Mapping[str, object]] = {}
        try:
            for group, label in GROUP_NAMES.items():
                count = int(library.gm1_idealized_carrier_count(self._handle, group))
                if count == _SIZE_MAX:
                    raise ValueError(_last_error(library))
                carriers: dict[str, object] = {}
                for index in range(count):
                    name_len = ctypes.c_size_t()
                    shape_ptr = ctypes.POINTER(ctypes.c_size_t)()
                    rank = ctypes.c_size_t()
                    values_ptr = ctypes.POINTER(ctypes.c_float)()
                    value_len = ctypes.c_size_t()
                    name_ptr = library.gm1_idealized_carrier_view(
                        self._handle,
                        group,
                        index,
                        ctypes.byref(name_len),
                        ctypes.byref(shape_ptr),
                        ctypes.byref(rank),
                        ctypes.byref(values_ptr),
                        ctypes.byref(value_len),
                    )
                    if not name_ptr:
                        raise ValueError(_last_error(library))
                    name = ctypes.string_at(name_ptr, name_len.value).decode("ascii")
                    shape = tuple(int(shape_ptr[item]) for item in range(rank.value))
                    words = 1
                    for extent in shape:
                        words *= extent
                    if words != int(value_len.value) or name in carriers:
                        raise ValueError(f"Rust idealized carrier {name} layout/inventory changed")
                    view = np.ctypeslib.as_array(values_ptr, shape=(words,)).reshape(shape)
                    if view.dtype != np.dtype("float32") or not view.flags.c_contiguous:
                        raise ValueError(f"Rust idealized carrier {name} is not C-f32")
                    view.flags.writeable = False
                    carriers[name] = view
                groups[label] = MappingProxyType(carriers)
            for label, expected in EXPECTED_GROUP_INVENTORY.items():
                if tuple(groups[label]) != expected:
                    raise ValueError(f"Rust idealized {label} carrier inventory changed")
            # terrain_admission.TERRAIN_OPEN_CARRIER_NAMES's fifty base and
            # mutable carriers, plus init3d.F:2386-2466's four initialized
            # prs/rho/rr/rf.  The count is checked so a library that publishes
            # a different inventory cannot be read as this one.
            missing = {"prs", "rho", "rr", "rf"} - set(groups["terrain_open"])
            if len(groups["terrain_open"]) != 54 or missing:
                raise ValueError(
                    "Rust idealized terrain/open carrier inventory changed: "
                    f"{len(groups['terrain_open'])} carriers, missing {sorted(missing)}"
                )
            # Read the frame back out of the built state.  The ABI carries
            # floats by register, so a library that ignored them would still
            # return a valid handle; only the readback proves the carriers
            # above are expressed in the frame that was asked for.
            if hasattr(library, "gm1_idealized_state_umove"):
                for key, entry, wanted in (
                    ("u_move_m_s", library.gm1_idealized_state_umove, u_move_m_s),
                    ("v_move_m_s", library.gm1_idealized_state_vmove, v_move_m_s),
                ):
                    seen = float(entry(self._handle))
                    if seen != wanted:
                        raise ValueError(
                            f"the Rust idealized initializer built this state with "
                            f"{key}={seen!r} m/s, not the {wanted!r} m/s requested; "
                            "every wind carrier is in a frame the caller did not "
                            "declare and every position derived from it would be "
                            "wrong by the difference times the model time"
                        )
        except BaseException:
            library.gm1_idealized_state_free(self._handle)
            self._handle = ctypes.c_void_p()
            raise
        self._views = MappingProxyType(groups)
        self.iinit = iinit
        self.geometry = geometry
        self.library_path = path
        # base.F:2663-2669.  Ground-relative winds are u_move_m_s + u.
        self.u_move_m_s = u_move_m_s
        self.v_move_m_s = v_move_m_s

    @property
    def groups(self) -> Mapping[str, Mapping[str, object]]:
        if not self._handle:
            raise RuntimeError("Rust idealized carrier owner is closed")
        return self._views

    def carrier(self, group: str, name: str) -> object:
        try:
            return self.groups[group][name]
        except KeyError as error:
            raise KeyError(f"unknown Rust idealized carrier {group}.{name}") from error

    def copy_groups(self) -> Mapping[str, Mapping[str, object]]:
        """Copy every carrier to independently owned C-order NumPy storage."""

        import numpy as np

        return MappingProxyType(
            {
                group: MappingProxyType(
                    {
                        name: np.array(value, dtype="<f4", order="C", copy=True)
                        for name, value in carriers.items()
                    }
                )
                for group, carriers in self.groups.items()
            }
        )

    def close(self) -> None:
        if self._handle:
            self._library.gm1_idealized_state_free(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "RustIdealizedCarrierMap":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self.close()


__all__ = (
    "BASE_WIND_GROUP",
    "GROUP_NAMES",
    "IdealizedGeometry",
    "MOIST_GROUP",
    "RustIdealizedCarrierMap",
    "TERRAIN_OPEN_GROUP",
    "TERRAIN_STATIC_GROUP",
)
