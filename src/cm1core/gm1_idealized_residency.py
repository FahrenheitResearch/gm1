"""Explicit source views and budgeted storage for a fresh idealized CM1 state.

Rust owns the sounding/hydrostatic/bubble equations.  This module only selects
source-owned array rectangles, proves optional horizontal profile compression,
and uploads explicitly requested buffers.  It never allocates the whole source
estate implicitly, imports CuPy, launches an operator, or advances a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from . import cuda_graphs
from .cuda_memory_inventory import cuda_array_allocation, shape_bytes, snapshot_cuda_allocations
from .gm1_idealized_in_memory import IdealizedGeometry, RustIdealizedCarrierMap
from .gm1_runtime import MAX_FRAME_SPEED_M_S, cm1_real
from .grid_geometry import CartesianGridGeometry


Layout = Literal["native", "physical", "profile"]
_PRESSURE = frozenset(("ppi", "pp3d", "ppx", "ppd", "piadv", "ppterm", "ppten"))
_U = frozenset(("u0", "ua", "u3d", "uten", "rru"))
_V = frozenset(("v0", "va", "v3d", "vten", "rrv"))
_W = frozenset(("wa", "w3d", "wten", "rrw", "rf", "rf0", "rrf0"))


@dataclass(frozen=True, slots=True)
class SourceRequest:
    """One semantic buffer from a qualified Rust group/carrier name.

    Examples: ``padded.u3d <- terrain_open.u3d / native``;
    ``q3d <- moist.q3d / physical``;
    ``base.th0 <- terrain_open.th0 / profile``;
    ``metric.rmh <- terrain_static.rmh / native``.
    Read-only declares the caller's mutation contract, not a CUDA protection.
    """

    name: str
    source: str
    layout: Layout = "native"
    read_only: bool = False

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("semantic buffer name must be a nonempty string")
        if not isinstance(self.source, str) or self.source.count(".") != 1:
            raise ValueError("source must be a qualified Rust group.carrier name")
        if self.layout not in ("native", "physical", "profile"):
            raise ValueError("layout must be native, physical, or profile")
        if type(self.read_only) is not bool:
            raise TypeError("read_only must be bool")


def _owner_accepts(owner_factory: Any, names: set[str]) -> bool:
    import inspect

    try:
        parameters = inspect.signature(owner_factory).parameters
    except (TypeError, ValueError):
        parameters = {}
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ) or names <= set(parameters)


def _levels_keywords(owner_factory: Any, geometry: CartesianGridGeometry) -> dict[str, Any]:
    """Hand the declared vertical coordinate to the Rust initializer.

    ``param.F:7789-7811``'s uniform column is what every build of the Rust
    initializer derives from ``dz``, so an unstretched grid passes nothing and
    reaches exactly the entry point and the raw words it always has.  Any
    other coordinate must reach base.F, because the base state, the warm
    bubble's height and every hodograph sample are functions of altitude: a
    library that ignored the levels would build a 15 km-deep sounding on a
    column that is not this run's, and nothing downstream would say so.  The
    stretched grid's first scalar level and ``nz*dz`` are different heights.
    """

    if geometry.vertical is None or geometry.stretch_z == 0:
        return {}
    if not _owner_accepts(owner_factory, {"levels"}):
        raise ValueError(
            f"the installed Rust idealized initializer takes no declared vertical "
            f"coordinate, so grid.stretch_z={geometry.stretch_z} "
            f"(param.F:7814-8139) cannot reach base.F: the base state would be "
            f"built on the uniform column with its first scalar level at "
            f"{0.5 * geometry.dz_m} m instead of {geometry.lowest_mass_level_m} m, "
            "and every base-state word, bubble height and hodograph sample would "
            "sit at the wrong altitude with nothing to say so"
        )
    return {"levels": geometry.levels}


def _selector_keywords(
    owner_factory: Any, selection: "InitializerSelection"
) -> dict[str, object]:
    """Wire base.F's selectors only when the installed Rust owner takes them.

    The single-sounding lane is what every build of the Rust initializer has
    exposed, so it is passed positionally as before and never depends on this.
    Any other lane needs the selector arguments to actually reach base.F; a
    library that does not take them would silently build the isnd=7 base state
    and the run would carry an atmosphere nobody asked for.  That is refused
    here by name, before a single carrier is allocated.

    Anything outside the historical ``(isnd, iwnd, iinit)`` triple travels as
    one ``options`` mapping through the extensible selector ABI, and an owner
    that cannot take it is refused for the same reason.
    """

    keywords: dict[str, object] = {}
    if _owner_accepts(owner_factory, {"options"}):
        # One path for every selection.  The option struct's own defaults are
        # CM1's, so a legacy selection reaches exactly the same Rust entry with
        # exactly the same words; routing it differently is how the two paths
        # would drift.
        keywords["options"] = selection.options()
    elif not selection.expressible_by_legacy_abi:
        raise ValueError(
            "the installed Rust idealized initializer takes no options "
            "selector struct, so CM1's isnd=17 lane, ibalance, irandp, "
            "testcase, imoist and the init3d.F parameter blocks cannot reach "
            "base.F; the state would be built with CM1's defaults for every "
            "one of them while this run's receipt named the declared values"
        )
    if selection == SINGLE_SOUNDING_SELECTION and not keywords:
        return keywords
    if not _owner_accepts(owner_factory, {"isnd", "iwnd"}):
        raise ValueError(
            f"the installed Rust idealized initializer takes no isnd/iwnd "
            f"selector, so the CM1 base.F isnd={selection.isnd} / "
            f"iwnd={selection.iwnd} initializer lane is unavailable in this "
            "build; install a cm1-state library whose idealized entry point "
            "accepts the base.F sounding and base-state wind selectors"
        )
    keywords["isnd"] = selection.isnd
    keywords["iwnd"] = selection.iwnd
    return keywords


def _frame_keywords(
    owner_factory: Any, frame: "TranslatingFrame"
) -> dict[str, float]:
    """Wire base.F:2663-2669's umove/vmove only when they are actually used.

    A fixed frame is the historical call, so it is not passed at all and an
    older owner keeps working unchanged.  A translating frame that an owner
    cannot take is refused by name here: passed silently it would be dropped,
    the state would be built ground-relative, and every position the run
    reported would be wrong by ``u_move_m_s * t``.
    """

    if not frame.translates:
        return {}
    if not _owner_accepts(owner_factory, {"u_move_m_s", "v_move_m_s"}):
        raise ValueError(
            "the installed Rust idealized initializer takes no u_move_m_s / "
            "v_move_m_s, so the CM1 base.F:2663-2669 translating frame is "
            "unavailable in this build; the base state would be built in the "
            "ground-relative frame and every position this run reported would "
            "be wrong by u_move_m_s times the model time"
        )
    return frame.record()


def initial_prognostic_requests() -> tuple[SourceRequest, ...]:
    """Physical current/A fields; every mutable field requests separate storage.

    The pair is intentionally not aliased even when initialization is equal.
    Density/pressure initialization uses the exact source carrier convention
    of the admitted idealized initializer, with no Python EOS reconstruction.
    """

    requests = [
        SourceRequest(name, f"terrain_open.{name}", "physical")
        for name in ("ua", "u3d", "va", "v3d", "wa", "w3d", "ppi", "pp3d", "ppx", "tha", "th3d")
    ]
    requests += [SourceRequest(name, f"moist.{name}", "physical") for name in ("qa", "q3d")]
    # init3d.F:2386-2418 and :2432-2466.  These are the INITIALIZED pressure
    # and density, not the base-state prs0/rho0: CM1's rho at model time zero
    # is prs/((th0+tha)*(pi0+pp3d)*(rd+max(0,qv)*rv)), which carries both the
    # theta perturbation and the ibalance pressure.  Reading rho0 here started
    # every run from a density inconsistent with its own theta by the
    # perturbation's own amplitude -- 0.33% for CM1's 1 K bubble, the same
    # order as the buoyancy the bubble exists to produce.
    requests += [
        SourceRequest(name, f"terrain_open.{name}", "physical")
        for name in ("prs", "rho", "rr", "rf")
    ]
    return tuple(requests)


def base_profile_requests() -> tuple[SourceRequest, ...]:
    """Explicit compact profiles; every horizontal plane is checked raw-word exact."""

    return tuple(
        SourceRequest(f"base.{name}", f"terrain_open.{name}", "profile", True)
        for name in ("pi0", "th0", "qv0", "rho0", "prs0")
    ) + tuple(
        SourceRequest(f"base.{name}", f"base_wind.{name}", "profile", True)
        for name in ("u0", "v0")
    )


#: `base.F:2286-2640` implements these eleven wind branches; `iwnd=7` does not
#: exist upstream.
IMPLEMENTED_IWND: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11})
#: `init3d.F:450-1416`'s arms that this port builds: all of them.  4 (the
#: moist benchmark bubble) and 7's vortex are built after the base state
#: (rust idealized_initialization.rs, tc_vortex.rs).  9, 10 and 12 build
#: no `tha` (their arms only set control words) and are applied during the run
#: by `cm1core.initiation_forcing`.
IMPLEMENTED_IINIT: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13})
#: base.F's analytic sounding arms (rust analytic_soundings.rs and, for 5,
#: storm_initialization.rs); 7 and 17 read input_sounding.  6 stops CM1
#: (base.F:455-460) and 15 needs testcase 4/5's constants (constants.F:78-98).
ANALYTIC_ISND: frozenset[int] = frozenset(
    {1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23}
)
ADMITTED_ISND: frozenset[int] = ANALYTIC_ISND | {7, 17}
#: base.F:267-268's moist benchmark words and run/namelist.input's tsk0.
CM1_THEC_MB_K = 320.0
CM1_QT_MB = 0.020
CM1_TSK0_K = 299.28


@dataclass(frozen=True, slots=True)
class InitializerSelection:
    """CM1 ``base.F`` / ``init3d.F`` selectors for one idealized source.

    ``isnd=7`` is the single ``input_sounding`` lane this module has always
    driven; ``isnd=5`` is base.F:363-470, the Weisman-Klemp analytic sounding;
    ``isnd=17`` reads the same external file for thermodynamics only and takes
    its winds from ``iwnd`` (param.F:849-851).  ``iwnd`` is
    base.F:2263-2650's branch set.  base.F:466 and param.F:836-852 say iwnd is
    ignored -- and silently reset to zero -- when isnd=7, so that pair is
    refused here rather than wired and discarded.

    Every field beyond the first two is one of CM1's own namelist or
    named-parameter words, carried so the receipt can state by value what the
    initial condition was built from.
    """

    isnd: int = 7
    iwnd: int = 0
    iinit: int = 1
    irst: int = 0
    imoist: int = 1
    testcase: int = 0
    ibalance: int = 0
    irandp: int = 0
    random_seed: int = 0
    #: 0: GM1's own seeded stream; 1: CM1's (libgfortran's random_number as
    #: init3d.F:427-440 seeds it), which takes no seed.
    random_generator: int = 0
    #: init3d.F:697-1256 (iinit=7) only: CM1's fcor (param.F:1985 makes it 0
    #: when icor = 0) and whether the sides repeat (its periodic images).
    fcor_s_inv: float = 5.0e-5
    periodic_x: int = 0
    periodic_y: int = 0
    ric_m: float | None = None
    rjc_m: float | None = None
    zc_m: float = 1400.0
    bhrad_m: float = 10000.0
    bvrad_m: float = 1400.0
    bptpert_k: float = 1.0
    maintain_rh: bool = False
    nbub: int = 3
    bubble_spacing_y_m: float = 30000.0
    zdep_m: float = 2500.0
    random_line_amplitude_k: float = 0.20
    hurr_vg_m_per_s: float = 40.0
    hurr_rotate_deg: float = 0.0
    #: base.F:267-268 (isnd=4; init3d.F:625-645 reads qt_mb for iinit=4) and
    #: the namelist tsk0 (base.F's isnd=11 and 21).
    thec_mb_k: float = CM1_THEC_MB_K
    qt_mb: float = CM1_QT_MB
    tsk0_k: float = CM1_TSK0_K
    #: param.F:5178-5205's imirror/jmirror, read by ibalance=2's poiss only.
    imirror: int = 1
    jmirror: int = 1
    #: CM1's eqtset after param.F:936-950 (1 for ptype 7, 2 for ptype 5);
    #: base.F's isnd=10 moist lapse rate reads it.
    eqtset: int = 2

    def __post_init__(self) -> None:
        for name in ("isnd", "iwnd", "iinit", "irst", "imoist", "testcase",
                     "ibalance", "irandp", "random_seed", "random_generator", "nbub"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an int")
        if type(self.maintain_rh) is not bool:
            raise TypeError("maintain_rh must be a bool")
        for name in ("ric_m", "rjc_m"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a float in metres or None")
        if self.isnd not in ADMITTED_ISND:
            raise ValueError(
                "base.F's transcribed sounding arms are "
                f"{sorted(ADMITTED_ISND)}; got {self.isnd}"
            )
        if self.iwnd not in IMPLEMENTED_IWND:
            raise ValueError(
                "base.F:2286-2640 has no arm for iwnd="
                f"{self.iwnd}; it would leave base.F:2263-2280's zero wind in place"
            )
        if self.iinit not in IMPLEMENTED_IINIT:
            raise ValueError(
                "init3d.F:450-1416's transcribed arms are "
                f"{sorted(IMPLEMENTED_IINIT)}; got {self.iinit}"
            )
        if self.ibalance not in (0, 1, 2):
            raise ValueError("init3d.F:1735-1917 implements ibalance=0, 1 or 2")
        if self.imoist not in (0, 1):
            raise ValueError("README.namelist:308 admits imoist=0 or 1")
        if self.irandp not in (0, 1):
            raise ValueError("README.namelist:692-695 admits irandp=0 or 1")
        if self.random_generator not in (0, 1):
            raise ValueError("random_generator is 0 (GM1's seeded stream) or 1 (CM1's own)")
        if self.random_generator == 1 and self.random_seed != 0:
            raise ValueError(
                "CM1's own random stream is seeded by init3d.F:427-440's fixed "
                "scramble; a declared seed would be recorded and never read"
            )
        if self.irandp == 1 and self.random_seed == 0 and self.random_generator == 0:
            raise ValueError(
                "irandp=1 requires an explicit nonzero seed; init3d.F:416-443 "
                "seeds one generator for the whole run and an unrecorded seed "
                "makes the perturbation field unreproducible from its receipt"
            )
        if self.isnd == 7 and self.iwnd != 0:
            raise ValueError(
                "param.F:836-852 forces iwnd=0 for isnd=7; the declared wind "
                "branch would be discarded and the sounding winds used instead. "
                "Select isnd=17 to pair the file's thermodynamics with an "
                "analytic hodograph"
            )

    @property
    def reads_input_sounding(self) -> bool:
        """base.F:495: isnd=7 and isnd=17 take their profile from the file."""

        return self.isnd in (7, 17)

    @property
    def expressible_by_legacy_abi(self) -> bool:
        """Can the historical positional entry points build exactly this?

        They carry ``(isnd, iwnd, iinit)`` and the two frame velocities, and
        they route ``isnd`` by a pair rather than by lane, so the isnd=17
        thermodynamics-only lane cannot reach them at all.  Everything else
        here has to be at CM1's own default for the positional call to mean
        what this selection says.
        """

        return (
            self.isnd in (5, 7)
            and self.iinit in (0, 1)
            and self == InitializerSelection(
                isnd=self.isnd, iwnd=self.iwnd, iinit=self.iinit
            )
        )

    def options(self) -> dict[str, object]:
        """The selector words the option-struct ABI carries, CM1-named."""

        return {
            "irst": self.irst,
            "imoist": self.imoist,
            "testcase": self.testcase,
            "ibalance": self.ibalance,
            "irandp": self.irandp,
            "random_seed": self.random_seed,
            "hurr_vg_m_per_s": self.hurr_vg_m_per_s,
            "hurr_rotate_deg": self.hurr_rotate_deg,
            "ric_m": self.ric_m,
            "rjc_m": self.rjc_m,
            "zc_m": self.zc_m,
            "bhrad_m": self.bhrad_m,
            "bvrad_m": self.bvrad_m,
            "bptpert_k": self.bptpert_k,
            "maintain_rh": self.maintain_rh,
            "nbub": self.nbub,
            "bubble_spacing_y_m": self.bubble_spacing_y_m,
            "zdep_m": self.zdep_m,
            "random_line_amplitude_k": self.random_line_amplitude_k,
            # Carried only when CM1's stream is selected, so every selection
            # made before it existed keeps its words.
            **({"random_generator": 1} if self.random_generator == 1 else {}),
            # The same rule for the vortex's environment (iinit=7 only).
            **(
                {"fcor_s_inv": self.fcor_s_inv, "periodic_x": self.periodic_x, "periodic_y": self.periodic_y}
                if self.iinit == 7 else {}
            ),
            # And for base.F's own words: the moist benchmark's (isnd=4)
            # and the skin temperature isnd=11 and 21 read.
            **({"thec_mb_k": self.thec_mb_k, "qt_mb": self.qt_mb} if self.isnd == 4 else {}),
            **({"tsk0_k": self.tsk0_k} if self.isnd in (11, 21) else {}),
            **({"imirror": self.imirror, "jmirror": self.jmirror} if self.ibalance == 2 else {}),
            **({"eqtset": self.eqtset} if self.isnd == 10 else {}),
        }

    def record(self) -> dict[str, object]:
        """Every selector by value, for the run receipt."""

        payload: dict[str, object] = {"isnd": self.isnd, "iwnd": self.iwnd}
        payload.update({"iinit": self.iinit})
        payload.update(self.options())
        return payload


SINGLE_SOUNDING_SELECTION = InitializerSelection()

@dataclass(frozen=True, slots=True)
class TranslatingFrame:
    """CM1 ``base.F:2661-2674``: the constant velocity of the model frame.

    ``imove=1`` moves the whole domain at ``(umove, vmove)``, and base.F
    implements it by subtracting the constant from the base-state winds once,
    at ``base.F:2667-2668``, after every ``isnd`` and ``iwnd`` branch has run.
    ``init3d.F:210-219`` then copies ``u0``/``v0`` into ``ua``/``u3d`` and
    ``va``/``v3d``, so the initial condition inherits the same offset from the
    same subtraction.  Ground-relative winds are ``u_move_m_s + u``
    (``init3d.F:32-33``, ``README.namelist:772-780``).

    The change of variable is exact only without Coriolis.  ``solve1.F:605-609``
    and ``solve1.F:623-624`` are the only places CM1 adds the constant back,
    both under ``icor=1``; CM1's own supercell deck runs ``icor=0``
    (``run/namelist.input:59``), and this runtime's momentum tendencies carry
    no rotation term at all.
    """

    u_move_m_s: float = 0.0
    v_move_m_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("u_move_m_s", "v_move_m_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a float in m/s")
            value = float(value)
            object.__setattr__(self, name, value)
            if not math.isfinite(value):
                raise ValueError(
                    f"{name}={value!r} is not finite; base.F:2667 subtracts it from "
                    "every base-state wind word, so u0/v0 and the ua/va copied from "
                    "them would be non-finite at every level and the first large "
                    "step would integrate a domain of NaN"
                )
            if abs(value) > MAX_FRAME_SPEED_M_S:
                raise ValueError(
                    f"{name}={value!r} m/s exceeds the {MAX_FRAME_SPEED_M_S} m/s this "
                    "lane admits; the frame velocity is a storm motion, and "
                    "base.F:2667 would replace the declared hodograph with a uniform "
                    "frame-relative jet through the whole depth"
                )
            if cm1_real(value) != value:
                raise ValueError(
                    f"{name}={value!r} m/s is not a CM1 `real`; umove and vmove are "
                    "binary32 (base.F:2661-2674) and this frame crosses to Rust as "
                    f"binary32, so base.F:2667 would subtract {cm1_real(value)!r} m/s "
                    "from every base-state wind word while cm1.F:3512 integrated the "
                    "declared value into domainlocx, and the published domain "
                    "location would drift from the carriers it describes"
                )

    @property
    def translates(self) -> bool:
        """CM1's ``imove`` as a predicate: is there anything to subtract?"""

        return self.u_move_m_s != 0.0 or self.v_move_m_s != 0.0

    @property
    def imove(self) -> int:
        return int(self.translates)

    def record(self) -> dict[str, float]:
        return {"u_move_m_s": self.u_move_m_s, "v_move_m_s": self.v_move_m_s}


FIXED_FRAME = TranslatingFrame()


class IdealizedSource:
    """Keep the opaque Rust owner alive through explicitly registered uploads.

    Host views are valid only inside this context.  Close synchronizes streams
    used for uploads before freeing Rust memory, so asynchronous H2D cannot
    outlive the source.  Existing device buffers remain valid after close.
    """

    def __init__(
        self,
        library: str | Path,
        sounding_bytes: bytes,
        geometry: CartesianGridGeometry,
        *,
        iinit: int | None = None,
        origin_xy_m: tuple[float, float] = (0.0, 0.0),
        bubble_center_xy_m: tuple[float, float] | None = None,
        selection: InitializerSelection = SINGLE_SOUNDING_SELECTION,
        moving_frame: TranslatingFrame = FIXED_FRAME,
        owner_factory: Any = RustIdealizedCarrierMap,
    ) -> None:
        if not isinstance(geometry, CartesianGridGeometry):
            raise TypeError("geometry must be CartesianGridGeometry")
        if not isinstance(selection, InitializerSelection):
            raise TypeError("selection must be InitializerSelection")
        if not isinstance(moving_frame, TranslatingFrame):
            raise TypeError("moving_frame must be TranslatingFrame")
        self.geometry = geometry
        self.selection = selection
        # init3d.F's arm is a property of the selection.  The separate argument
        # is kept so the callers that predate it keep working, and a caller
        # that passes both a selection and a different iinit is refused rather
        # than silently given one of the two.
        if iinit is None:
            iinit = selection.iinit
        elif iinit != selection.iinit:
            raise ValueError(
                f"iinit={iinit} contradicts the selection's iinit={selection.iinit}; "
                "the state would carry one perturbation and the receipt name another"
            )
        self.moving_frame = moving_frame
        self.library = Path(library)
        self.sounding_sha256 = hashlib.sha256(sounding_bytes).hexdigest()
        if bubble_center_xy_m is None:
            bubble_center_xy_m = (
                origin_xy_m[0] + geometry.nx * geometry.dx_m * 0.5,
                origin_xy_m[1] + geometry.ny * geometry.dy_m * 0.5,
            )
        self.initializer_geometry = IdealizedGeometry(
            geometry.shape_zyx,
            geometry.spacing_m,
            origin_xy_m,
            bubble_center_xy_m,
        )
        self._owner = owner_factory(
            library,
            sounding_bytes,
            iinit=iinit,
            geometry=self.initializer_geometry,
            **_selector_keywords(owner_factory, selection),
            **_frame_keywords(owner_factory, moving_frame),
            **_levels_keywords(owner_factory, geometry),
        )
        self._streams: dict[int, object] = {}
        self._closed = False
        self._profiles: dict[str, object] = {}

    def __enter__(self) -> "IdealizedSource":
        if self._closed:
            raise RuntimeError("idealized source is already closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            for stream in self._streams.values():
                stream.synchronize()  # type: ignore[attr-defined]
            self._streams.clear()
            self._profiles.clear()
            self._owner.close()
            self._closed = True

    @property
    def groups(self) -> Mapping[str, Mapping[str, Any]]:
        if self._closed:
            raise RuntimeError("idealized source is closed")
        return self._owner.groups

    def _physical(self, group: str, name: str, value: Any) -> Any:
        g = self.geometry
        if group in ("terrain_static", "base_wind"):
            return value
        if group == "moist":
            expected = (6, *g.padded_shape("mass", halo_xy=3, halo_z=1))
            if tuple(value.shape) != expected:
                raise ValueError(f"source {group}.{name} shape {value.shape} differs from {expected}")
            return value[(slice(None), *g.physical_slices("mass", halo_xy=3, halo_z=1))]
        if group == "base_condensate":
            # base.F's qc0/qi0 on the padded mass layout (extrapbcs ghosts,
            # base.F:2048-2049, 2077-2078), like terrain_open's pi0/th0/qv0.
            expected = g.padded_shape("mass", halo_xy=3, halo_z=1)
            if tuple(value.shape) != expected:
                raise ValueError(f"source {group}.{name} shape {value.shape} differs from {expected}")
            return value[g.physical_slices("mass", halo_xy=3, halo_z=1)]
        if group != "terrain_open":
            raise ValueError(f"no source physical-layout rule for {group}.{name}")
        if value.ndim < 3:
            # Native surface/radiation carriers do not define a physical
            # volume crop.  They must be requested with their native layout.
            raise ValueError(f"source {group}.{name} requires native layout")
        stagger = "u" if name in _U else "v" if name in _V else "w" if name in _W else "mass"
        halo_xy = 1 if name in _PRESSURE else 3
        # Rust terrain-base rf0/rrf0 and the initialized rf have CM1 levels
        # 0:nk+1.  The
        # prognostic w family has 0:nk+2 instead.  Both expose physical
        # faces 1:nk+1, but their stored upper halo extents are different.
        expected = (
            (g.nz + 2, g.ny + 6, g.nx + 6)
            if name in ("rf0", "rrf0", "rf")
            else g.padded_shape(stagger, halo_xy=halo_xy, halo_z=1)
        )
        if tuple(value.shape) != expected:
            raise ValueError(f"source {group}.{name} shape {value.shape} differs from {expected}")
        return value[g.physical_slices(stagger, halo_xy=halo_xy, halo_z=1)]

    def field(self, source: str, *, layout: Layout = "native") -> Any:
        """Return immutable Rust words, with no science transformation."""

        if layout not in ("native", "physical", "profile"):
            raise ValueError("layout must be native, physical, or profile")
        try:
            group, name = source.split(".")
            value = self.groups[group][name]
        except (ValueError, KeyError) as error:
            raise ValueError(f"unknown Rust idealized source carrier {source!r}") from error
        if layout == "native":
            return value
        physical = self._physical(group, name, value)
        if layout == "physical":
            return physical
        if source in self._profiles:
            return self._profiles[source]
        if physical.ndim != 3:
            raise ValueError(f"source {source} is not a scalar vertical-profile candidate")
        import numpy as np

        # One horizontal plane at a time bounds temporary comparison memory.
        # uint32 comparison preserves signed zero and payload bits.
        words = physical.view(np.uint32)
        for level in range(physical.shape[0]):
            if not np.all(words[level] == words[level, 0, 0]):
                raise ValueError(f"source {source} varies horizontally at level {level}; profile compression refused")
        profile = np.array(physical[:, 0, 0], dtype="<f4", order="C", copy=True)
        profile.flags.writeable = False
        self._profiles[source] = profile
        return profile

    def plan(self, requests: Sequence[SourceRequest], *, alignment: int = 512) -> dict[str, Any]:
        """Exact requested payload and allocator rounding, excluding external state."""

        if type(alignment) is not int or alignment < 1:
            raise ValueError("alignment must be a positive integer")
        names: set[str] = set()
        rows = []
        for request in requests:
            if request.name in names:
                raise ValueError(f"duplicate semantic buffer {request.name}")
            names.add(request.name)
            value = self.field(request.source, layout=request.layout)
            nbytes = int(value.nbytes)
            rows.append(
                {
                    "name": request.name,
                    "source": request.source,
                    "layout": request.layout,
                    "read_only": request.read_only,
                    "shape": list(value.shape),
                    "payload_bytes": nbytes,
                    "allocation_budget_bytes": ((nbytes + alignment - 1) // alignment) * alignment,
                }
            )
        return {
            "geometry": self.geometry.record(),
            "buffers": rows,
            "payload_bytes": sum(row["payload_bytes"] for row in rows),
            "allocation_budget_bytes": sum(row["allocation_budget_bytes"] for row in rows),
            "allocator_alignment_bytes": alignment,
            "unreferenced_pool_reserve_bytes": None,
            "driver_context_and_other_allocations_bytes": None,
            "claim_limit": "requested named buffers only; pool cache and driver headroom are separate",
        }


class DeviceBufferRegistry:
    """Explicit named CUDA allocations under a caller-selected byte budget.

    The budget covers this registry's allocations, not total device usage.
    Callers subtract driver/other workloads and pool reserve before choosing
    it.  No callback or operation is launched by this factory.
    """

    def __init__(self, cupy: Any, stream: Any, *, budget_bytes: int, alignment: int = 512) -> None:
        if type(budget_bytes) is not int or budget_bytes < 1:
            raise ValueError("budget_bytes must be a positive integer")
        if type(alignment) is not int or alignment < 1:
            raise ValueError("alignment must be a positive integer")
        self.cupy = cupy
        self.stream = stream
        self.budget_bytes = budget_bytes
        self.alignment = alignment
        self._buffers: dict[str, Any] = {}
        self._bytes: dict[str, int] = {}
        self._read_only: dict[str, bool] = {}
        self._owners: dict[str, str] = {}

    @property
    def buffers(self) -> Mapping[str, Any]:
        return MappingProxyType(self._buffers)

    @property
    def registered_allocation_bytes(self) -> int:
        return sum(self._bytes.values())

    def _reserve(self, name: str, payload: int) -> int:
        if not isinstance(name, str) or not name or name in self._buffers or name in self._bytes:
            raise ValueError(f"semantic buffer name is empty or already registered: {name!r}")
        rounded = ((payload + self.alignment - 1) // self.alignment) * self.alignment
        if self.registered_allocation_bytes + rounded > self.budget_bytes:
            raise MemoryError(
                f"buffer {name} needs {rounded} bytes; registry uses "
                f"{self.registered_allocation_bytes} of {self.budget_bytes}; "
                "allocation refused before calling the CUDA allocator"
            )
        return rounded

    def upload(self, source: IdealizedSource, requests: Sequence[SourceRequest]) -> Mapping[str, Any]:
        plan = source.plan(requests, alignment=self.alignment)
        new_names = {request.name for request in requests}
        if new_names.intersection(self._buffers):
            raise ValueError("an upload would overwrite an existing semantic buffer")
        if self.registered_allocation_bytes + plan["allocation_budget_bytes"] > self.budget_bytes:
            raise MemoryError("requested upload exceeds registry budget before any CUDA allocation")
        uploaded = {}
        # Register even partial uploads: allocation failures cannot leave a
        # hidden, uncounted owner or free Rust bytes before a queued H2D copy.
        source._streams[id(self.stream)] = self.stream
        with self.stream:
            for request in requests:
                host = source.field(request.source, layout=request.layout)
                rounded = self._reserve(request.name, int(host.nbytes))
                device = self.cupy.asarray(host, dtype=self.cupy.float32, order="C")
                self._buffers[request.name] = device
                self._bytes[request.name] = rounded
                self._owners[request.name] = request.name
                self._read_only[request.name] = request.read_only
                uploaded[request.name] = device
        return MappingProxyType(uploaded)

    def allocate(self, name: str, shape: tuple[int, ...], *, zero: bool = False, dtype: Any = None) -> Any:
        import numpy as np

        dtype = self.cupy.float32 if dtype is None else dtype
        itemsize = int(np.dtype(dtype).itemsize)
        rounded = self._reserve(name, shape_bytes(shape, itemsize=itemsize))
        with self.stream:
            allocate = self.cupy.zeros if zero else self.cupy.empty
            value = allocate(shape, dtype=dtype)
        self._buffers[name] = value
        self._bytes[name] = rounded
        self._owners[name] = name
        self._read_only[name] = False
        return value

    def adopt(self, name: str, value: Any, *, read_only: bool = False) -> Any:
        """Account for an already-created exposed CUDA workspace allocation.

        This cannot preflight an external allocation.  It refuses unknown
        ownership and mutable aliases instead of inventing a view-sized total.
        Public component allocations that expose no arrays remain a separate
        reservation in the caller's device memory plan.
        """

        allocation = cuda_array_allocation(value)
        if allocation is None:
            raise TypeError("adopt requires CUDA allocation ownership metadata")
        if type(read_only) is not bool:
            raise TypeError("read_only must be bool")
        for existing, buffer in self._buffers.items():
            known = cuda_array_allocation(buffer)
            if known is not None and (known.device_id, known.allocation_pointer) == (
                allocation.device_id, allocation.allocation_pointer
            ):
                if read_only and self._read_only[existing] and buffer is value:
                    return self.alias_read_only(name, existing)
                raise ValueError("adopt would alias an existing mutable or different-view allocation")
        rounded = self._reserve(name, allocation.allocation_bytes)
        self._buffers[name] = value
        self._bytes[name] = rounded
        self._owners[name] = name
        self._read_only[name] = read_only
        return value

    def alias_read_only(self, name: str, target: str) -> Any:
        """Add another name for the exact same declared immutable object.

        This does not deduplicate different source arrays merely because their
        initial values happen to match.  A/current buffers cannot use this API.
        """

        if target not in self._buffers or not self._read_only[target]:
            raise ValueError("aliases require an existing declared read-only buffer")
        if not isinstance(name, str) or not name or name in self._buffers:
            raise ValueError("alias name must be new and nonempty")
        self._buffers[name] = self._buffers[target]
        self._owners[name] = self._owners[target]
        self._read_only[name] = True
        return self._buffers[name]

    def alias_read_only_view(self, name: str, target: str, shape: tuple[int, ...]) -> Any:
        """Expose a prefix of the exact immutable words, with explicit shape."""
        if target not in self._buffers or not self._read_only[target]:
            raise ValueError("views require an existing declared read-only buffer")
        if name in self._buffers or not isinstance(name, str) or not name:
            raise ValueError("view name must be new and nonempty")
        payload = shape_bytes(shape)
        value = self._buffers[target]
        if payload > int(value.nbytes):
            raise ValueError("read-only view extends beyond its source words")
        view = value.reshape(-1)[:payload // 4].reshape(shape)
        self._buffers[name], self._owners[name], self._read_only[name] = view, self._owners[target], True
        return view

    def alias_read_only_broadcast(self, name: str, target: str, shape: tuple[int, ...]) -> Any:
        """Broadcast an explicitly verified immutable vertical profile."""
        if target not in self._buffers or not self._read_only[target]:
            raise ValueError("broadcast requires an existing declared read-only profile")
        if name in self._buffers or not isinstance(name, str) or not name:
            raise ValueError("view name must be new and nonempty")
        shape_bytes(shape)
        value = self._buffers[target]
        if len(shape) != 3 or tuple(value.shape) != (shape[0],):
            raise ValueError("broadcast profile length differs from the requested vertical extent")
        view = self.cupy.broadcast_to(value[:, None, None], shape)
        self._buffers[name], self._owners[name], self._read_only[name] = view, self._owners[target], True
        return view

    def read_only(self, name: str) -> bool:
        """Whether this named allocation is a read-only source upload or alias.

        A read-only buffer is reproduced by re-uploading the same source; a
        mutable one is state a consumer may have written and is the only kind
        a checkpoint has to carry.
        """

        if name not in self._read_only:
            raise KeyError(f"unregistered semantic buffer {name!r}")
        return self._read_only[name]

    def release(self, names: Sequence[str]) -> None:
        """Release named registry references after preceding stream work ends."""

        if len(set(names)) != len(names):
            raise ValueError("release names must be unique")
        for name in names:
            if name not in self._buffers:
                raise KeyError(name)
        cuda_graphs.synchronize(self.stream)
        for name in names:
            owner = self._owners.pop(name)
            del self._buffers[name], self._read_only[name]
            if owner not in self._owners.values():
                self._bytes.pop(owner)

    def snapshot(self, *, label: str) -> dict[str, Any]:
        return snapshot_cuda_allocations(self._buffers, label=label)


__all__ = (
    "FIXED_FRAME",
    "MAX_FRAME_SPEED_M_S",
    "SINGLE_SOUNDING_SELECTION",
    "DeviceBufferRegistry",
    "IdealizedSource",
    "InitializerSelection",
    "SourceRequest",
    "TranslatingFrame",
    "base_profile_requests",
    "initial_prognostic_requests",
)
