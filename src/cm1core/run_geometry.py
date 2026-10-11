"""Will the storm this configuration initializes still be inside this domain?

Every number here is a host computation on the declared configuration and the
declared ``input_sounding``.  Nothing in this module opens a device, allocates
device memory, or reads model output: it answers, before a GPU is taken, the
one question a lateral-boundary-bounded idealized run has to answer about its
own geometry.

The question is not academic.  CM1's ``iinit=1`` perturbation
(``init3d.F:450-482``) is a warm bubble centred at ``ric``/``rjc``, which
``init3d.F:456-457`` sets to the domain centre ``centerx``/``centery``
(``param.F:7408``).  A bubble that has not yet organised does not move at a
mature supercell's motion; it advects with the wind through its own depth.
Declare a translating frame (``base.F:2663-2669``) at the mature motion and
the unorganised perturbation moves through the domain at the difference,
which for a strongly sheared profile is 10-15 m/s -- 40-55 km in the hour a
supercell needs to organise.  On a domain narrower than twice that, the
perturbation reaches the radiative lateral boundary (``bc.F``) before it
becomes a storm, and every later domain maximum is a boundary artifact.

Three quantities decide it, and all three are computable here:

``bubble_steering_wind``
    The environmental wind averaged over the initial perturbation with the
    perturbation's own weight.  ``init3d.F:468-482`` gives every point the
    weight ``cos(0.5*pi*beta)**2``; on the bubble axis ``beta`` is
    ``|z-zc|/bvrad``, so that weight is a function of height alone and the
    advecting wind of the initial condition is its weighted mean.

``bunkers_storm_motion``
    The mature supercell's motion.  This is not CM1: it is the
    internal-dynamics method of Bunkers et al. (2000, Wea. Forecasting 15,
    61-79), the same method the storm-motion line of a sounding record is
    computed with, transcribed here so the frame velocity and the domain can
    be checked against each other from the sounding the run will actually
    read.

``StormTrackModel``
    How the motion gets from the first to the second.  A supercell is not
    born moving at its mature motion; it organises, then splits, then the
    members deviate.  The model is three declared times and a linear ramp
    between them, and both its parameters and its uncertainty band are named
    constants rather than buried numbers.

The result is a track in *frame* coordinates -- the coordinates the domain is
fixed in -- and the model time, if any, at which the tracked member's required
clearance from a lateral boundary is lost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence


RUN_GEOMETRY_SCHEMA = "gpuwm-cm1.run-geometry/v1"

#: ``init3d.F:456-461`` -- the six words of the official ``iinit=1`` warm
#: bubble, inside the ``init3d.F:450-482`` block.  The port's Rust
#: initializer carries the same values in
#: ``rust/cm1-state/src/idealized_initialization.rs``
#: (``WarmBubbleControls::official``), so these are the numbers the run uses.
BUBBLE_CENTER_Z_AGL_M = 1400.0
BUBBLE_HORIZONTAL_RADIUS_M = 10000.0
BUBBLE_VERTICAL_RADIUS_M = 1400.0

#: Bunkers et al. (2000): the mean wind layer, the two shear-anchor layers and
#: the magnitude of the deviation from the mean wind, in m/s.
BUNKERS_MEAN_WIND_TOP_M = 6000.0
BUNKERS_SHEAR_LOWER_TOP_M = 500.0
BUNKERS_SHEAR_UPPER_BOTTOM_M = 5500.0
BUNKERS_SHEAR_UPPER_TOP_M = 6000.0
BUNKERS_DEVIATION_M_S = 7.5

#: Layer means are a uniform-in-height quadrature.  A sounding's own levels
#: are unevenly spaced, so a level-count mean would weight the dense part of
#: the profile; the midpoint rule on a uniform height grid does not.
LAYER_MEAN_SAMPLES = 2048

#: The lateral clearance the tracked member must keep, as a multiple of the
#: bubble's own horizontal radius plus the advection stencil's halo.  The
#: bubble radius is the scale of the initial perturbation and, once the storm
#: is mature, the scale of the updraft-and-hook core that the run exists to
#: resolve; the halo is the three cells the fifth-order operators read.
STENCIL_HALO_CELLS = 3

#: How far the modelled steering wind can be wrong, in m/s.  The one
#: measurement available is this project's own: a 100 m run whose tracked
#: cloud drifted 3.5 m/s further upwind in the zonal component than
#: ``bubble_steering_wind`` gives for its sounding.  Rounded up, and applied
#: for the length of the unorganised phase only -- after the storm organises
#: its motion is anchored to the mature motion, which is a far better
#: constrained quantity than the advection of an unorganised plume.
STEERING_UNCERTAINTY_M_S = 4.0

#: One-based member selectors.  ``RIGHT`` is the Northern-Hemisphere
#: right-moving (cyclonic) member a supercell frame is normally chosen for.
MEMBER_RIGHT = "right"
MEMBER_LEFT = "left"
MEMBER_MEAN = "mean"
MEMBERS: tuple[str, ...] = (MEMBER_RIGHT, MEMBER_LEFT, MEMBER_MEAN)
_MEMBER_SIGN: Mapping[str, float] = MappingProxyType(
    {MEMBER_RIGHT: 1.0, MEMBER_LEFT: -1.0, MEMBER_MEAN: 0.0}
)


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number, not {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, not {result!r}")
    return result


@dataclass(frozen=True, slots=True)
class Hodograph:
    """The base-state wind profile the run will actually integrate.

    ``heights_m`` is strictly ascending and begins at zero, exactly as
    ``base.F:605`` and ``base.F:617-618`` leave ``zsnd``/``usnd``/``vsnd``
    after the surface record is prepended and the surface wind extrapolated.
    ``wind_at`` is ``base.F:685-702``'s own linear interpolation in height.
    """

    heights_m: tuple[float, ...]
    u_m_s: tuple[float, ...]
    v_m_s: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (len(self.heights_m) == len(self.u_m_s) == len(self.v_m_s)):
            raise ValueError("hodograph height, u and v columns must be the same length")
        if len(self.heights_m) < 3:
            raise ValueError(
                "a hodograph needs at least three levels: base.F:617-618 extrapolates "
                "the surface wind from the first two data rows, so a two-row profile "
                "has no surface value and no interior to interpolate"
            )
        for lower, upper in zip(self.heights_m, self.heights_m[1:]):
            if not upper > lower:
                raise ValueError(
                    f"hodograph heights must strictly ascend; {upper} follows {lower}"
                )

    @property
    def top_m(self) -> float:
        return self.heights_m[-1]

    def wind_at(self, height_m: float) -> tuple[float, float]:
        """``base.F:685-702``: linear in height between bracketing levels."""

        z = _finite("height_m", height_m)
        if z <= self.heights_m[0]:
            return self.u_m_s[0], self.v_m_s[0]
        if z >= self.heights_m[-1]:
            return self.u_m_s[-1], self.v_m_s[-1]
        low, high = 0, len(self.heights_m) - 1
        while high - low > 1:
            middle = (low + high) // 2
            if self.heights_m[middle] <= z:
                low = middle
            else:
                high = middle
        span = self.heights_m[high] - self.heights_m[low]
        fraction = (z - self.heights_m[low]) / span
        return (
            self.u_m_s[low] + (self.u_m_s[high] - self.u_m_s[low]) * fraction,
            self.v_m_s[low] + (self.v_m_s[high] - self.v_m_s[low]) * fraction,
        )

    def layer_mean(
        self, bottom_m: float, top_m: float, *, samples: int = LAYER_MEAN_SAMPLES
    ) -> tuple[float, float]:
        """Midpoint-rule mean wind over one height layer."""

        bottom = _finite("bottom_m", bottom_m)
        top = _finite("top_m", top_m)
        if not top > bottom:
            raise ValueError(f"layer top {top} must lie above layer bottom {bottom}")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("layer mean sample count must be a positive integer")
        span = top - bottom
        sum_u = sum_v = 0.0
        for index in range(samples):
            u, v = self.wind_at(bottom + span * (index + 0.5) / samples)
            sum_u += u
            sum_v += v
        return sum_u / samples, sum_v / samples


def hodograph_from_input_sounding(source: bytes | str) -> Hodograph:
    """Read ``base.F``'s ``isnd=7`` wind columns, with CM1's surface record.

    ``base.F:543-561`` reads a three-word header and then five-word records of
    ``z, theta, qv, u, v``; ``base.F:605`` prepends ``zsnd(1)=0`` and
    ``base.F:617-618`` extrapolates the surface wind from the first two data
    rows, because the header carries no wind.  Both are done here, so the
    profile this module reasons about is the profile the model integrates.
    """

    if isinstance(source, bytes):
        try:
            text = source.decode("utf-8")
        except UnicodeError as error:
            raise ValueError(f"input_sounding is not UTF-8 text: {error}") from None
    elif isinstance(source, str):
        text = source
    else:
        raise TypeError("input_sounding source must be bytes or str")
    records: list[tuple[float, float, float]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        words = line.split()
        if not words:
            continue
        if number == 1 or (not records and len(words) == 3):
            # base.F:546 -- the one-line surface header carries no wind.
            continue
        if len(words) < 5:
            raise ValueError(
                f"input_sounding line {number} has {len(words)} words; base.F:553 "
                "reads five (z, theta, qv, u, v) from every record after the header"
            )
        try:
            values = (float(words[0]), float(words[3]), float(words[4]))
        except ValueError:
            raise ValueError(
                f"input_sounding line {number} does not read as z, theta, qv, u, v"
            ) from None
        records.append(values)
    if len(records) < 2:
        raise ValueError(
            "base.F:569-576 stops CM1 with fewer than two wind levels; this profile "
            f"has {len(records)}"
        )
    heights = [0.0] + [row[0] for row in records]
    us = [0.0] + [row[1] for row in records]
    vs = [0.0] + [row[2] for row in records]
    # base.F:617-618, verbatim: simple extrapolation from the first two data
    # rows onto z=0.  Indices 1 and 2 here are CM1's usnd(2) and usnd(3).
    span = heights[2] - heights[1]
    if not span > 0.0:
        raise ValueError("the first two input_sounding records share a height")
    us[0] = us[1] - (heights[1] - heights[0]) * (us[2] - us[1]) / span
    vs[0] = vs[1] - (heights[1] - heights[0]) * (vs[2] - vs[1]) / span
    return Hodograph(tuple(heights), tuple(us), tuple(vs))


#: ``base.F:2286-2640``: the ``iwnd`` branches CM1 r21.1 implements, in source
#: order.  ``iwnd=7`` has no arm (the chain has no ``else``), so it is not here;
#: ``rust/cm1-state/src/base_state_wind.rs`` carries the same list.
IMPLEMENTED_IWND: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11)

#: The height the ground row of an analytic hodograph is evaluated at: the
#: branch's limit from above, a millimetre up.
GROUND_LIMIT_M = 1.0e-3

#: ``base.F:2467-2510``: the ``iwnd=6`` constant wind by ``testcase``; any other
#: testcase takes the ``else`` arm's 10 m/s westerly.
_IWND6_TESTCASE_WIND: Mapping[int, tuple[float, float]] = MappingProxyType({
    1: (1.0, 0.0), 2: (15.0, 0.0), 4: (7.0, -5.5), 9: (8.0, 0.0),
    11: (2.0, 0.0), 12: (20.0, 0.0), 14: (10.0, 0.0),
})


def analytic_base_wind(
    iwnd: int,
    height_m: float,
    *,
    testcase: int = 0,
    hurr_vg_m_s: float = 40.0,
    hurr_rotate_deg: float = 0.0,
) -> tuple[float, float]:
    """``base.F:2263-2650``: CM1's closed-form base-state wind at one height.

    ``base.F:2263`` runs this block for every ``isnd`` except 7, so it is the
    wind of the analytic sounding (``isnd=5``) *and* of the thermodynamics-only
    file lane (``isnd=17``, ``param.F:849-851``).  Each branch is a pure
    function of the face height (``zu``/``zv``, ``base.F:2294,2313``); on the
    flat Cartesian lane both equal the scalar height ``zh(k)``.  The literals
    are ``base.F``'s own, transcribed branch for branch as
    ``rust/cm1-state/src/base_state_wind.rs`` transcribes them for the
    initializer; this copy is evaluated in double precision because it prices
    a track, it does not initialize a state.
    """

    if isinstance(iwnd, bool) or not isinstance(iwnd, int):
        raise TypeError("iwnd must be an integer")
    if iwnd not in IMPLEMENTED_IWND:
        raise ValueError(
            f"base.F:2286-2640 implements iwnd in {IMPLEMENTED_IWND}; got iwnd={iwnd}"
        )
    z = _finite("height_m", height_m)
    pi = math.pi
    if iwnd == 0:
        return 0.0, 0.0
    if iwnd == 1:  # base.F:2286-2316, Rotunno, Klemp and Weisman (1988)
        # base.F:2297-2302: zu < 0 takes uconst1, 0 < zu < 2500 the ramp, and
        # the else arm everything else -- including zu = 0 exactly.
        u = 10.0 * z / 2500.0 if 0.0 < z < 2500.0 else (0.0 if z < 0.0 else 10.0)
        return u, 0.0
    if iwnd == 2:  # base.F:2321-2357, Weisman-Klemp supercell (quarter circle)
        if z <= 2000.0:
            angle = 90.0 * (z / 2000.0) * (pi / 180.0)
            return 7.0 - 7.0 * math.cos(angle), 7.0 * math.sin(angle)
        if z <= 6000.0:
            return 7.0 + (z - 2000.0) * (31.0 - 7.0) / 4000.0, 7.0
        return 31.0, 7.0
    if iwnd == 3:  # base.F:2366-2396, multicell
        low, high = -40.0 / pi, 40.0 / pi + 40.0
        u = low + z * (high - low) / 7500.0 if z <= 7500.0 else high
        return u, 40.0 / pi
    if iwnd == 4:  # base.F:2400-2419, Weisman and Klemp (1982) multicell
        return 35.0 * math.tanh(z / 3000.0), 0.0
    if iwnd == 5:  # base.F:2425-2461, Dornbrack et al. (2005)
        if z < 4000.0:
            return 15.0, 0.0
        if z < 6000.0:
            alpha = 0.25 * pi * (1.0 + math.cos(pi * (z - 4000.0) / 2000.0))
            return 15.0 * math.sin(alpha), 15.0 * math.cos(alpha)
        return 0.0, 15.0
    if iwnd == 6:  # base.F:2467-2510, constant wind by testcase
        return _IWND6_TESTCASE_WIND.get(int(testcase), (10.0, 0.0))
    if iwnd == 8:  # base.F:2525-2552, hurricane boundary layer; param.F:8973
        angle = _finite("hurr_rotate_deg", hurr_rotate_deg) * (pi / 180.0)
        vg = _finite("hurr_vg_m_s", hurr_vg_m_s)
        return -vg * math.sin(-angle), vg * math.cos(-angle)
    if iwnd == 9:  # base.F:2557-2589, Siebesma et al. (2003)
        if z <= 700.0:
            return -8.75, 0.0
        if z < 3000.0:
            return (-4.61 + 8.75) * (z - 700.0) / 2300.0 - 8.75, 0.0
        return -4.61, 0.0
    if iwnd == 10:  # base.F:2593-2613, Ackerman et al. (2009)
        return 3.0 + 4.3 * z / 1000.0, -9.0 + 5.6 * z / 1000.0
    # iwnd == 11: base.F:2617-2640, RICO (vanZanten et al. 2011)
    if z <= 0.0:
        return -9.9, -3.8
    if z < 4000.0:
        return (-1.9 + 9.9) * z / 4000.0 - 9.9, -3.8
    return -1.9, -3.8


def hodograph_from_analytic(
    iwnd: int,
    scalar_heights_m: Sequence[float],
    *,
    testcase: int = 0,
    hurr_vg_m_s: float = 40.0,
    hurr_rotate_deg: float = 0.0,
) -> Hodograph:
    """The analytic lanes' wind on the model's own scalar levels.

    ``base.F:2263-2650`` evaluates the branch at every ``zh(k)`` and nowhere
    else, so the profile this module reasons about is those rows, in the same
    order.  The one row added is the ground, which a hodograph begins at and
    CM1 stores no level for.  It takes the branch's limit from above
    (:data:`GROUND_LIMIT_M`), so a branch whose ``else`` arm happens to catch
    ``zu = 0`` exactly (``iwnd=1``'s top-of-layer 10 m/s) does not put that
    value at the ground.
    """

    heights = [float(z) for z in scalar_heights_m]
    if not heights or heights[0] <= 0.0:
        raise ValueError("scalar heights must be positive and ascend from the lowest level")
    options = {
        "testcase": testcase, "hurr_vg_m_s": hurr_vg_m_s, "hurr_rotate_deg": hurr_rotate_deg,
    }
    ground = analytic_base_wind(iwnd, min(GROUND_LIMIT_M, 0.5 * heights[0]), **options)
    winds = [ground] + [analytic_base_wind(iwnd, z, **options) for z in heights]
    return Hodograph(
        tuple([0.0] + heights), tuple(w[0] for w in winds), tuple(w[1] for w in winds)
    )


def bubble_steering_wind(
    hodograph: Hodograph,
    *,
    center_z_agl_m: float = BUBBLE_CENTER_Z_AGL_M,
    vertical_radius_m: float = BUBBLE_VERTICAL_RADIUS_M,
    samples: int = LAYER_MEAN_SAMPLES,
) -> tuple[float, float]:
    """The wind that advects the initial perturbation, with its own weight.

    ``init3d.F:468-482`` gives each point the amplitude
    ``cos(0.5*pi*beta)**2`` inside ``beta<1``.  On the bubble axis ``beta``
    reduces to ``|z-zc|/bvrad``, so the perturbation's vertical weight is that
    cosine-squared profile and the wind it advects with is its weighted mean.
    Below ground the weight is dropped rather than reflected: there is no air
    there and CM1 evaluates no point there.
    """

    zc = _finite("center_z_agl_m", center_z_agl_m)
    bvrad = _finite("vertical_radius_m", vertical_radius_m)
    if bvrad <= 0.0:
        raise ValueError("the bubble vertical radius must be positive")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("steering-wind sample count must be a positive integer")
    bottom = max(0.0, zc - bvrad)
    top = zc + bvrad
    if not top > bottom:
        raise ValueError("the bubble spans no air above ground")
    span = top - bottom
    sum_u = sum_v = weight_sum = 0.0
    for index in range(samples):
        z = bottom + span * (index + 0.5) / samples
        beta = abs(z - zc) / bvrad
        if beta >= 1.0:
            continue
        weight = math.cos(0.5 * math.pi * beta) ** 2
        u, v = hodograph.wind_at(z)
        sum_u += weight * u
        sum_v += weight * v
        weight_sum += weight
    if weight_sum <= 0.0:
        raise ValueError("the bubble weight integrates to zero over the column")
    return sum_u / weight_sum, sum_v / weight_sum


@dataclass(frozen=True, slots=True)
class StormMotion:
    """Bunkers et al. (2000) internal-dynamics motion, from one hodograph."""

    mean_u_m_s: float
    mean_v_m_s: float
    deviation_u_m_s: float
    deviation_v_m_s: float

    @property
    def right_mover_m_s(self) -> tuple[float, float]:
        return (
            self.mean_u_m_s + self.deviation_u_m_s,
            self.mean_v_m_s + self.deviation_v_m_s,
        )

    @property
    def left_mover_m_s(self) -> tuple[float, float]:
        return (
            self.mean_u_m_s - self.deviation_u_m_s,
            self.mean_v_m_s - self.deviation_v_m_s,
        )

    def member_m_s(self, member: str) -> tuple[float, float]:
        sign = _MEMBER_SIGN[member]
        return (
            self.mean_u_m_s + sign * self.deviation_u_m_s,
            self.mean_v_m_s + sign * self.deviation_v_m_s,
        )

    def record(self) -> dict[str, Any]:
        return {
            "method": "Bunkers et al. (2000) internal-dynamics, 7.5 m/s deviation",
            "mean_wind_m_s": [self.mean_u_m_s, self.mean_v_m_s],
            "deviation_m_s": [self.deviation_u_m_s, self.deviation_v_m_s],
            "right_mover_m_s": list(self.right_mover_m_s),
            "left_mover_m_s": list(self.left_mover_m_s),
        }


def bunkers_storm_motion(hodograph: Hodograph) -> StormMotion:
    """Mean wind plus a 7.5 m/s deviation normal to the 0-6 km shear vector.

    ``V_mean`` is the 0-6 km mean wind; ``V_shear`` is the 5.5-6 km mean minus
    the 0-0.5 km mean; the right mover is ``V_mean + 7.5*(shear_v,-shear_u)``
    normalized.  This is the published method, not a CM1 source transcription,
    and it is labelled as such in every record this module emits.
    """

    if not isinstance(hodograph, Hodograph):
        raise TypeError("hodograph must be a Hodograph")
    if hodograph.top_m < BUNKERS_MEAN_WIND_TOP_M:
        raise ValueError(
            f"the profile tops out at {hodograph.top_m} m, below the "
            f"{BUNKERS_MEAN_WIND_TOP_M} m this motion method integrates to"
        )
    mean_u, mean_v = hodograph.layer_mean(0.0, BUNKERS_MEAN_WIND_TOP_M)
    low_u, low_v = hodograph.layer_mean(0.0, BUNKERS_SHEAR_LOWER_TOP_M)
    high_u, high_v = hodograph.layer_mean(
        BUNKERS_SHEAR_UPPER_BOTTOM_M, BUNKERS_SHEAR_UPPER_TOP_M
    )
    shear_u, shear_v = high_u - low_u, high_v - low_v
    magnitude = math.hypot(shear_u, shear_v)
    if magnitude <= 0.0:
        # No shear vector means no defined deviation direction.  An unsheared
        # profile has no supercell and no split, and saying so is better than
        # dividing by zero or inventing a direction.
        return StormMotion(mean_u, mean_v, 0.0, 0.0)
    scale = BUNKERS_DEVIATION_M_S / magnitude
    return StormMotion(mean_u, mean_v, scale * shear_v, -scale * shear_u)


@dataclass(frozen=True, slots=True)
class StormTrackModel:
    """How the tracked motion gets from the steering wind to the mature one.

    ``organization_s`` is how long the perturbation advects unorganised at the
    steering wind; ``split_s`` is when the cell has split; ``deviation_s`` is
    when the members have reached their full Bunkers deviation.  Motion is
    linear in time between those knots and constant outside them.

    The defaults are this project's own measurements, not literature values:
    the 1 km fixed-domain run in this repository's lineage split at T+46 min,
    and the two high-resolution runs that produced no storm had about 28
    minutes of usable domain and an updraft that never organised, which is the
    lower end of the stated 30-45 minute organisation time.
    """

    organization_s: float = 1800.0
    split_s: float = 2760.0
    deviation_s: float = 3600.0

    def __post_init__(self) -> None:
        values = (
            _finite("organization_s", self.organization_s),
            _finite("split_s", self.split_s),
            _finite("deviation_s", self.deviation_s),
        )
        if values[0] < 0.0:
            raise ValueError("organization_s must not be negative")
        if not values[1] > values[0]:
            raise ValueError("split_s must lie after organization_s")
        if not values[2] > values[1]:
            raise ValueError("deviation_s must lie after split_s")

    def record(self) -> dict[str, Any]:
        return {
            "organization_s": self.organization_s,
            "split_s": self.split_s,
            "deviation_s": self.deviation_s,
        }


DEFAULT_TRACK_MODEL = StormTrackModel()

#: The stated 30-45 minute organisation time as an uncertainty band.  The
#: geometry is evaluated at both ends and reported at its worst, so a domain
#: is admitted only if it holds the storm for either.
ORGANIZATION_BAND_S: tuple[float, float] = (1800.0, 2700.0)


def track_velocity(
    model: StormTrackModel,
    steering_m_s: tuple[float, float],
    motion: StormMotion,
    member: str,
    time_s: float,
) -> tuple[float, float]:
    """Ground-relative velocity of one member at one model time."""

    if member not in _MEMBER_SIGN:
        raise ValueError(f"unknown track member {member!r}; expected one of {MEMBERS}")
    t = _finite("time_s", time_s)
    if t < 0.0:
        raise ValueError("model time must not be negative")
    steer_u, steer_v = float(steering_m_s[0]), float(steering_m_s[1])
    mean_u, mean_v = motion.mean_u_m_s, motion.mean_v_m_s
    sign = _MEMBER_SIGN[member]
    dev_u, dev_v = sign * motion.deviation_u_m_s, sign * motion.deviation_v_m_s
    if t <= model.organization_s:
        return steer_u, steer_v
    if t <= model.split_s:
        fraction = (t - model.organization_s) / (model.split_s - model.organization_s)
        return (
            steer_u + (mean_u - steer_u) * fraction,
            steer_v + (mean_v - steer_v) * fraction,
        )
    if t <= model.deviation_s:
        fraction = (t - model.split_s) / (model.deviation_s - model.split_s)
        return mean_u + dev_u * fraction, mean_v + dev_v * fraction
    return mean_u + dev_u, mean_v + dev_v


def track_displacement(
    model: StormTrackModel,
    steering_m_s: tuple[float, float],
    motion: StormMotion,
    member: str,
    time_s: float,
) -> tuple[float, float]:
    """Ground displacement from the initial position, exactly integrated.

    Every segment of ``track_velocity`` is constant or linear in time, so the
    trapezoid of each segment is its exact integral; there is no quadrature
    error in this number.
    """

    t = _finite("time_s", time_s)
    if t < 0.0:
        raise ValueError("model time must not be negative")
    knots = (0.0, model.organization_s, model.split_s, model.deviation_s)
    x = y = 0.0
    for lower, upper in zip(knots, knots[1:]):
        if t <= lower:
            break
        end = min(t, upper)
        u0, v0 = track_velocity(model, steering_m_s, motion, member, lower)
        u1, v1 = track_velocity(model, steering_m_s, motion, member, end)
        x += 0.5 * (u0 + u1) * (end - lower)
        y += 0.5 * (v0 + v1) * (end - lower)
    if t > model.deviation_s:
        u, v = track_velocity(model, steering_m_s, motion, member, t)
        x += u * (t - model.deviation_s)
        y += v * (t - model.deviation_s)
    return x, y


@dataclass(frozen=True, slots=True)
class MemberGeometry:
    """One member's track inside one fixed domain, in frame coordinates."""

    member: str
    start_xy_m: tuple[float, float]
    end_xy_m: tuple[float, float]
    minimum_xy_m: tuple[float, float]
    maximum_xy_m: tuple[float, float]
    minimum_clearance_m: float
    clearance_at_end_m: float
    exit_time_s: float | None

    @property
    def stays_inside(self) -> bool:
        return self.exit_time_s is None

    def record(self) -> dict[str, Any]:
        return {
            "member": self.member,
            "start_xy_m": list(self.start_xy_m),
            "end_xy_m": list(self.end_xy_m),
            "minimum_xy_m": list(self.minimum_xy_m),
            "maximum_xy_m": list(self.maximum_xy_m),
            "minimum_clearance_m": self.minimum_clearance_m,
            "clearance_at_end_m": self.clearance_at_end_m,
            "exit_time_s": self.exit_time_s,
            "stays_inside": self.stays_inside,
        }


@dataclass(frozen=True, slots=True)
class RunGeometry:
    """The complete host answer for one configuration."""

    domain_extent_m: tuple[float, float]
    bubble_center_xy_m: tuple[float, float]
    frame_m_s: tuple[float, float]
    steering_m_s: tuple[float, float]
    motion: StormMotion
    required_clearance_m: float
    steering_uncertainty_allowance_m: float
    admission_clearance_m: float
    duration_s: float
    organization_band_s: tuple[float, float]
    members: Mapping[str, MemberGeometry]
    recommended_bubble_center_xy_m: tuple[float, float]
    recommended_frame_m_s: tuple[float, float]
    required_domain_extent_m: tuple[float, float]

    @property
    def tracked(self) -> MemberGeometry:
        return self.members[MEMBER_RIGHT]

    def recommendation(self) -> dict[str, Any]:
        """The one change that makes this run fit, with its numbers.

        A refusal whose remedy is itself refused leaves the operator with
        nothing to do, so the remedy is decided here in the order that costs
        least: move the bubble if the declared frame's own excursion already
        fits the declared grid; otherwise change the frame as well; otherwise
        say how wide the grid has to be under the best frame available.
        """

        width, depth = self.domain_extent_m
        radius = BUBBLE_HORIZONTAL_RADIUS_M
        centre = self.recommended_bubble_center_xy_m
        fits_here = (
            self.required_domain_extent_m[0] <= width
            and self.required_domain_extent_m[1] <= depth
            and radius <= centre[0] <= width - radius
            and radius <= centre[1] <= depth - radius
        )
        if fits_here:
            return {
                "action": "move-the-bubble",
                "bubble_center_xy_m": list(centre),
                "frame_m_s": list(self.frame_m_s),
                "required_domain_extent_m": list(self.required_domain_extent_m),
            }
        frame = self.recommended_frame_m_s
        best = _recommended_placement(
            steering_m_s=self.steering_m_s,
            motion=self.motion,
            frame_m_s=frame,
            duration_s=self.duration_s,
            domain_extent_m=self.domain_extent_m,
            admission_clearance_m=self.admission_clearance_m,
            organization_band_s=self.organization_band_s,
        )
        fits_reframed = (
            best["required_domain_extent_m"][0] <= width
            and best["required_domain_extent_m"][1] <= depth
            and radius <= best["bubble_center_xy_m"][0] <= width - radius
            and radius <= best["bubble_center_xy_m"][1] <= depth - radius
        )
        return {
            "action": "change-the-frame" if fits_reframed else "widen-the-domain",
            "bubble_center_xy_m": list(best["bubble_center_xy_m"]),
            "frame_m_s": list(frame),
            "required_domain_extent_m": list(best["required_domain_extent_m"]),
        }

    def record(self) -> dict[str, Any]:
        return {
            "schema": RUN_GEOMETRY_SCHEMA,
            "authority": (
                "CM1 r21.1 init3d.F:450-482 warm bubble; init3d.F:456-457 ric/rjc; "
                "param.F:7408 centerx/centery; base.F:2663-2669 translating frame; "
                "base.F:605,617-618,685-702 sounding winds. The storm motion is "
                "Bunkers et al. (2000), a published method and not CM1 source."
            ),
            "domain_extent_m": list(self.domain_extent_m),
            "bubble_center_xy_m": list(self.bubble_center_xy_m),
            "frame_m_s": list(self.frame_m_s),
            "bubble_steering_wind_m_s": list(self.steering_m_s),
            "storm_motion": self.motion.record(),
            "required_clearance_m": self.required_clearance_m,
            "steering_uncertainty_allowance_m": self.steering_uncertainty_allowance_m,
            "steering_uncertainty_m_s": STEERING_UNCERTAINTY_M_S,
            "admission_clearance_m": self.admission_clearance_m,
            "duration_s": self.duration_s,
            "organization_band_s": list(self.organization_band_s),
            "members": {name: item.record() for name, item in self.members.items()},
            "recommended_bubble_center_xy_m": list(self.recommended_bubble_center_xy_m),
            "recommended_frame_m_s": list(self.recommended_frame_m_s),
            "required_domain_extent_m": list(self.required_domain_extent_m),
            "recommendation": self.recommendation(),
        }


def required_clearance_m(dx_m: float, dy_m: float) -> float:
    """Lateral clearance the tracked member must keep from every boundary.

    The bubble's own horizontal radius (``init3d.F:459``) plus the fifth-order
    advection halo.  The radius is the scale of the initial perturbation and,
    once the storm is mature, of the updraft-and-hook core the run exists to
    resolve; inside the halo the operators are already reading boundary rows.
    """

    dx = _finite("dx_m", dx_m)
    dy = _finite("dy_m", dy_m)
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("grid spacings must be positive")
    return BUBBLE_HORIZONTAL_RADIUS_M + STENCIL_HALO_CELLS * max(dx, dy)


def _candidate_times(
    duration_s: float,
    model: StormTrackModel,
    steering_m_s: tuple[float, float],
    motion: StormMotion,
    member: str,
    frame_m_s: tuple[float, float],
) -> tuple[float, ...]:
    """Every model time at which a frame-relative extremum can occur.

    ``track_velocity`` is piecewise linear in time, so the frame-relative
    position is piecewise quadratic and its extrema over a segment lie either
    at a segment end or where the frame-relative velocity crosses zero.  Both
    are computed here exactly.  Nothing is sampled: a sweep would both cost
    more and miss an extremum that falls between samples.
    """

    knots = [0.0, model.organization_s, model.split_s, model.deviation_s, duration_s]
    times = {0.0, duration_s}
    for knot in knots:
        if 0.0 < knot < duration_s:
            times.add(knot)
    ordered = sorted(times)
    for lower, upper in zip(ordered, ordered[1:]):
        u0, v0 = track_velocity(model, steering_m_s, motion, member, lower)
        u1, v1 = track_velocity(model, steering_m_s, motion, member, upper)
        span = upper - lower
        for start, end, frame in ((u0, u1, frame_m_s[0]), (v0, v1, frame_m_s[1])):
            slope = end - start
            if slope == 0.0:
                continue
            # Relative velocity start-frame + slope*(t-lower)/span crosses zero.
            offset = -(start - frame) * span / slope
            if 0.0 < offset < span:
                times.add(lower + offset)
    return tuple(sorted(times))


def _evaluate_member(
    *,
    member: str,
    model: StormTrackModel,
    steering_m_s: tuple[float, float],
    motion: StormMotion,
    frame_m_s: tuple[float, float],
    start_xy_m: tuple[float, float],
    domain_extent_m: tuple[float, float],
    clearance_m: float,
    duration_s: float,
    open_sides: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> MemberGeometry:
    frame_u, frame_v = frame_m_s
    x0, y0 = start_xy_m
    width, depth = domain_extent_m

    def edge_clearance(x: float, y: float) -> float:
        return _open_side_clearance(x, y, width, depth, open_sides)

    def clearance_at(t: float) -> tuple[float, float, float]:
        dx, dy = track_displacement(model, steering_m_s, motion, member, t)
        # base.F:2667 subtracts the frame velocity from every base-state wind,
        # so a ground displacement becomes a frame displacement by removing
        # the frame's own travel over the same time.
        x = x0 + dx - frame_u * t
        y = y0 + dy - frame_v * t
        return x, y, edge_clearance(x, y)

    min_x = min_y = math.inf
    max_x = max_y = -math.inf
    worst_clearance = math.inf
    exit_time: float | None = None
    previous: float | None = None
    end_xy = (x0, y0)
    for t in _candidate_times(
        duration_s, model, steering_m_s, motion, member, frame_m_s
    ):
        x, y, clearance = clearance_at(t)
        min_x, max_x = min(min_x, x), max(max_x, x)
        min_y, max_y = min(min_y, y), max(max_y, y)
        worst_clearance = min(worst_clearance, clearance)
        if exit_time is None and clearance < clearance_m:
            # The clearance is piecewise quadratic and its extrema are the
            # candidate times, so the first violating candidate brackets the
            # crossing with the previous one.  Bisect to a tenth of a second,
            # which is finer than any large step this lane runs.
            low = 0.0 if previous is None else previous
            high = t
            for _ in range(48):
                if high - low <= 0.1:
                    break
                middle = 0.5 * (low + high)
                if clearance_at(middle)[2] < clearance_m:
                    high = middle
                else:
                    low = middle
            exit_time = high
        previous = t
        end_xy = (x, y)
    return MemberGeometry(
        member=member,
        start_xy_m=(x0, y0),
        end_xy_m=end_xy,
        minimum_xy_m=(min_x, min_y),
        maximum_xy_m=(max_x, max_y),
        minimum_clearance_m=worst_clearance,
        clearance_at_end_m=edge_clearance(end_xy[0], end_xy[1]),
        exit_time_s=exit_time,
    )


def _open_side_clearance(
    x: float,
    y: float,
    width: float,
    depth: float,
    open_sides: tuple[bool, bool, bool, bool],
) -> float:
    """Distance to the nearest *open* lateral boundary (west, east, south, north).

    A periodic side (``wbc``/``ebc``/``sbc``/``nbc`` = 1) is no boundary for a
    storm to reach: what leaves one edge re-enters the other.  A rigid wall
    (= 3) and an open radiative side (= 2) both are, and both count.  With no
    side counted the distance is infinite.
    """

    west, east, south, north = open_sides
    distances = [
        value
        for value, counted in ((x, west), (width - x, east), (y, south), (depth - y, north))
        if counted
    ]
    return min(distances) if distances else math.inf


def _model_variants(
    model: StormTrackModel, band: tuple[float, float]
) -> tuple[StormTrackModel, ...]:
    """The track models the geometry is evaluated over.

    The organisation time is the one parameter of the track model with a
    stated range rather than a single value, so the geometry is evaluated at
    both ends of that range and at the nominal, and reported at its worst.
    """

    variants: list[StormTrackModel] = []
    for organization in sorted({band[0], band[1], model.organization_s}):
        if not 0.0 <= organization < model.split_s:
            continue
        variants.append(StormTrackModel(organization, model.split_s, model.deviation_s))
    return tuple(variants) if variants else (model,)


def _track_bounds(
    *,
    model: StormTrackModel,
    steering_m_s: tuple[float, float],
    motion: StormMotion,
    frame_m_s: tuple[float, float],
    member: str,
    duration_s: float,
    band: tuple[float, float] = ORGANIZATION_BAND_S,
) -> tuple[float, float, float, float]:
    """Frame-relative displacement bounds of one member's track.

    The bounds are the union over every admitted organisation time, so the
    bubble placement and domain extent this module recommends hold for the
    whole stated band rather than for its nominal value alone.
    """

    frame_u, frame_v = frame_m_s
    min_x = min_y = math.inf
    max_x = max_y = -math.inf
    for variant in _model_variants(model, band):
        for t in _candidate_times(
            duration_s, variant, steering_m_s, motion, member, frame_m_s
        ):
            dx, dy = track_displacement(variant, steering_m_s, motion, member, t)
            x, y = dx - frame_u * t, dy - frame_v * t
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
    return min_x, max_x, min_y, max_y


def minimax_frame_velocity(
    *,
    steering_m_s: tuple[float, float],
    motion: StormMotion,
    member: str = MEMBER_RIGHT,
    duration_s: float,
    model: StormTrackModel = DEFAULT_TRACK_MODEL,
) -> tuple[float, float]:
    """The frame velocity that minimizes the tracked member's excursion.

    The track is a fixed function of time; subtracting a constant frame
    velocity shifts every sample by ``-frame*t``.  The excursion along each
    axis is the width of the sample set, and the frame that minimizes it is
    the Chebyshev centre of the point set ``(t, displacement)`` -- which for a
    one-dimensional family of lines through the origin is found by bisection
    on the frame speed.  Ten iterations of golden-section over the interval
    between the steering wind and the mature motion gets it to well under
    0.01 m/s, which is far finer than the model that produced the track.
    """

    duration = _finite("duration_s", duration_s)
    if duration <= 0.0:
        raise ValueError("duration_s must be positive")
    target = motion.member_m_s(member)

    def width(fraction: float) -> tuple[float, float]:
        frame = (
            steering_m_s[0] + fraction * (target[0] - steering_m_s[0]),
            steering_m_s[1] + fraction * (target[1] - steering_m_s[1]),
        )
        min_x, max_x, min_y, max_y = _track_bounds(
            model=model,
            steering_m_s=steering_m_s,
            motion=motion,
            frame_m_s=frame,
            member=member,
            duration_s=duration,
        )
        return max_x - min_x, max_y - min_y

    # The excursion in each axis is convex in the frame fraction, and the two
    # axes share one parameter here because the frame is constrained to the
    # segment between the two physical velocities.  Minimize their maximum.
    low, high = -0.5, 1.5
    phi = 0.5 * (math.sqrt(5.0) - 1.0)
    a = high - phi * (high - low)
    b = low + phi * (high - low)
    fa, fb = max(width(a)), max(width(b))
    for _ in range(60):
        if fa < fb:
            high, b, fb = b, a, fa
            a = high - phi * (high - low)
            fa = max(width(a))
        else:
            low, a, fa = a, b, fb
            b = low + phi * (high - low)
            fb = max(width(b))
    fraction = 0.5 * (low + high)
    return (
        steering_m_s[0] + fraction * (target[0] - steering_m_s[0]),
        steering_m_s[1] + fraction * (target[1] - steering_m_s[1]),
    )


def _recommended_placement(
    *,
    steering_m_s: tuple[float, float],
    motion: StormMotion,
    frame_m_s: tuple[float, float],
    duration_s: float,
    domain_extent_m: tuple[float, float],
    admission_clearance_m: float,
    organization_band_s: tuple[float, float],
    model: StormTrackModel = DEFAULT_TRACK_MODEL,
) -> dict[str, Any]:
    """Centre the tracked member's excursion in the domain, and price it."""

    min_x, max_x, min_y, max_y = _track_bounds(
        model=model,
        steering_m_s=steering_m_s,
        motion=motion,
        frame_m_s=frame_m_s,
        member=MEMBER_RIGHT,
        duration_s=duration_s,
        band=organization_band_s,
    )
    width, depth = domain_extent_m
    return {
        "bubble_center_xy_m": (
            0.5 * width - 0.5 * (min_x + max_x),
            0.5 * depth - 0.5 * (min_y + max_y),
        ),
        "required_domain_extent_m": (
            (max_x - min_x) + 2.0 * admission_clearance_m,
            (max_y - min_y) + 2.0 * admission_clearance_m,
        ),
    }


def evaluate_run_geometry(
    *,
    hodograph: Hodograph,
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
    duration_s: float,
    frame_m_s: tuple[float, float] = (0.0, 0.0),
    bubble_center_xy_m: tuple[float, float] | None = None,
    model: StormTrackModel = DEFAULT_TRACK_MODEL,
    organization_band_s: tuple[float, float] = ORGANIZATION_BAND_S,
    members: Sequence[str] = MEMBERS,
    open_sides: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> RunGeometry:
    """Track every member of the declared initial condition through the run.

    ``bubble_center_xy_m`` defaults to ``param.F:7408``'s domain centre, which
    is what ``init3d.F:456-457`` assigns to ``ric``/``rjc``.  The domain is the
    half-open rectangle ``[0, nx*dx) x [0, ny*dy)`` the port's initializer
    builds from ``origin_xy_m=(0,0)``.  ``open_sides`` (west, east, south,
    north) names the sides a storm can be lost at; a periodic side is not one
    of them, and the default -- every side counted -- is the lane's own
    all-open deck.
    """

    for name, value in (("nx", nx), ("ny", ny)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    dx = _finite("dx_m", dx_m)
    dy = _finite("dy_m", dy_m)
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("grid spacings must be positive")
    duration = _finite("duration_s", duration_s)
    if duration <= 0.0:
        raise ValueError("duration_s must be positive")
    width, depth = nx * dx, ny * dy
    if bubble_center_xy_m is None:
        bubble_center_xy_m = (0.5 * width, 0.5 * depth)
    center = (
        _finite("bubble_center_x_m", bubble_center_xy_m[0]),
        _finite("bubble_center_y_m", bubble_center_xy_m[1]),
    )
    frame = (
        _finite("frame u", frame_m_s[0]),
        _finite("frame v", frame_m_s[1]),
    )
    steering = bubble_steering_wind(hodograph)
    motion = bunkers_storm_motion(hodograph)
    clearance = required_clearance_m(dx, dy)
    band = (
        _finite("organization band low", organization_band_s[0]),
        _finite("organization band high", organization_band_s[1]),
    )
    # The allowance is the steering error times the nominal unorganised
    # phase, not the band maximum: the band is already evaluated end to end
    # below and reported at its worst, so multiplying the two worst cases
    # together would refuse domains that hold the storm comfortably.
    allowance = STEERING_UNCERTAINTY_M_S * model.organization_s
    admission = clearance + allowance
    sides = tuple(bool(side) for side in open_sides)
    if len(sides) != 4:
        raise ValueError("open_sides names four sides: west, east, south, north")
    variants = _model_variants(model, band)
    evaluated: dict[str, MemberGeometry] = {}
    for member in members:
        worst: MemberGeometry | None = None
        for variant in variants:
            item = _evaluate_member(
                member=member,
                model=variant,
                steering_m_s=steering,
                motion=motion,
                frame_m_s=frame,
                start_xy_m=center,
                domain_extent_m=(width, depth),
                clearance_m=admission,
                duration_s=duration,
                open_sides=sides,
            )
            if worst is None or item.minimum_clearance_m < worst.minimum_clearance_m:
                worst = item
        assert worst is not None
        evaluated[member] = worst
    # What would have worked: the frame that minimizes the tracked member's
    # excursion, the bubble centre that puts that excursion's bounding box in
    # the middle of the domain, and the domain that holds it with clearance.
    recommended_frame = minimax_frame_velocity(
        steering_m_s=steering,
        motion=motion,
        member=MEMBER_RIGHT,
        duration_s=duration,
        model=model,
    )
    placement = _recommended_placement(
        steering_m_s=steering,
        motion=motion,
        frame_m_s=frame,
        duration_s=duration,
        domain_extent_m=(width, depth),
        admission_clearance_m=admission,
        organization_band_s=band,
        model=model,
    )
    recommended_center = placement["bubble_center_xy_m"]
    required_extent = placement["required_domain_extent_m"]
    return RunGeometry(
        domain_extent_m=(width, depth),
        bubble_center_xy_m=center,
        frame_m_s=frame,
        steering_m_s=steering,
        motion=motion,
        required_clearance_m=clearance,
        steering_uncertainty_allowance_m=allowance,
        admission_clearance_m=admission,
        duration_s=duration,
        organization_band_s=band,
        members=MappingProxyType(evaluated),
        recommended_bubble_center_xy_m=recommended_center,
        recommended_frame_m_s=recommended_frame,
        required_domain_extent_m=required_extent,
    )


__all__ = [
    "BUBBLE_CENTER_Z_AGL_M",
    "BUBBLE_HORIZONTAL_RADIUS_M",
    "BUBBLE_VERTICAL_RADIUS_M",
    "DEFAULT_TRACK_MODEL",
    "Hodograph",
    "MEMBERS",
    "MEMBER_LEFT",
    "MEMBER_MEAN",
    "MEMBER_RIGHT",
    "MemberGeometry",
    "ORGANIZATION_BAND_S",
    "RUN_GEOMETRY_SCHEMA",
    "RunGeometry",
    "STEERING_UNCERTAINTY_M_S",
    "StormMotion",
    "StormTrackModel",
    "bubble_steering_wind",
    "bunkers_storm_motion",
    "evaluate_run_geometry",
    "hodograph_from_input_sounding",
    "minimax_frame_velocity",
    "required_clearance_m",
    "track_displacement",
    "track_velocity",
]
