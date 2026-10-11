"""Every ArWen microphysics scheme GM1 cannot yet select, and the one thing
that blocks each.

This is not a wish list.  ArWen's public dispatch
(``gpuwm/core/microphysics.py`` ``apply``) admits ``mp_physics`` 1, 6, 8, 9,
10, 16, 18, 28 and 50, and GM1 binds four of those nine.  The other five are
implemented, compiled and reachable inside ArWen; what stops each here is a
single, named, local piece of GM1 work, recorded so that the next reader does
not have to re-derive it and so that ``gpuwm-cm1 doctor`` can say it out loud
rather than letting a refusal read as "unsupported".

Two blockers do all the work:

``six-mass-species-post-physics``
    ``cuda_gm1_tilestream_post_wsm6``'s EOS/``qa`` publication kernel copies a
    compile-time constant six mass rows
    (``gm1_tiled_post_wsm6_eos_qa_f32_v1``: ``for(unsigned s=0;s<6u;++s)``),
    and its binding checks ``qa`` is exactly ``(6, nk, ny, nx)``.  CM1's own
    ``ptype`` tables are not all six long -- ``ptype=1`` is three
    (``src/param.F:3171``) and ``ptype=50`` is four mass rows before its
    number and rime rows (``src/param.F:4102-4108``) -- so a component with a
    different mass prefix has no publication path.  Making that loop bound a
    kernel parameter is the whole fix; the copy has no arithmetic, so six
    stays bit-identical.

``column-vertical-velocity-carrier`` -- built
    ``gpuwm/core/milbrandt2.py:254`` reads ``state.w`` -- WRF's ``grid%w_2``
    lower full-level slice.  GM1 now stages it: ``ColumnComponentSpec``
    declares ``extra_interface_carriers`` and
    ``cuda_cartesian_finish`` copies the dycore's ``w3d``, which is the same
    quantity on exactly ArWen's ``(nz+1, ny, nx)`` stagger.  Classic and
    aerosol-aware Thompson both needed it and both are now bound; Milbrandt-Yau
    still needs the six-mass-species change as well.

Nothing here is a scientific refusal.  Each entry names GM1 work, not a
missing capability.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .errors import refuse


COLUMN_COMPONENT_BLOCKER_SCHEMA = "gm1.column-component-blocker/v1"


class ColumnComponentBlocker:
    """One unselectable component: what it is, and the one thing in the way."""

    __slots__ = ("component_id", "scheme_name", "mp_physics", "cm1_ptype",
                 "blocker", "detail")

    def __init__(
        self,
        *,
        component_id: str,
        scheme_name: str,
        mp_physics: int,
        cm1_ptype: int | None,
        blocker: str,
        detail: str,
    ) -> None:
        self.component_id = component_id
        self.scheme_name = scheme_name
        self.mp_physics = mp_physics
        self.cm1_ptype = cm1_ptype
        self.blocker = blocker
        self.detail = detail

    def record(self) -> dict[str, object]:
        return {
            "schema": COLUMN_COMPONENT_BLOCKER_SCHEMA,
            "component_id": self.component_id,
            "scheme_name": self.scheme_name,
            "arwen_mp_physics": self.mp_physics,
            "cm1_ptype": self.cm1_ptype,
            "blocker": self.blocker,
            "detail": self.detail,
        }


_BLOCKERS: tuple[ColumnComponentBlocker, ...] = (
    ColumnComponentBlocker(
        component_id="kessler-mp1",
        scheme_name="Kessler",
        mp_physics=1,
        cm1_ptype=1,
        blocker="six-mass-species-post-physics",
        detail=(
            "CM1 ptype=1 carries three mass scalars, qv,qc,qr "
            "(src/param.F:3171-3183); the sealed post-physics EOS/qa kernel "
            "publishes exactly six.  GM1 also already carries a CM1-transcribed "
            "ptype=1 operator family of its own (cuda_gm1_kessler_driver.py and "
            "siblings), which is the only path on which GM1 could show bit-level "
            "agreement with native CM1 microphysics, so this one is worth "
            "unblocking first"
        ),
    ),
    ColumnComponentBlocker(
        component_id="milbrandt2-mp9",
        scheme_name="Milbrandt-Yau two-moment",
        mp_physics=9,
        cm1_ptype=None,
        blocker="six-mass-species-post-physics",
        detail=(
            "Milbrandt-Yau carries a seventh mass row, hail (gpuwm/core/"
            "milbrandt2.py reads state.qh beside qc,qr,qi,qs,qg), so its mass "
            "prefix is seven rather than six.  Its other requirement, state.w "
            "at :254, is already satisfied: ColumnComponentSpec declares "
            "extra_interface_carriers and the tile inventory stages the "
            "dycore's w3d.  CM1 r21.1 has no Milbrandt ptype, so there is no "
            "CM1 differential for it either way"
        ),
    ),
    ColumnComponentBlocker(
        component_id="nssl2-mp18",
        scheme_name="NSSL two-moment",
        mp_physics=18,
        cm1_ptype=26,
        blocker="public-seam-requires-a-physics-driver",
        detail=(
            "gpuwm.core.microphysics.apply refuses mp_physics=18 without a "
            "DomainState PhysicsDriver and a persistent production binding "
            "(NSSL2ProductionBinding), neither of which the CM1-owned column "
            "adapter constructs: the scheme is reached through "
            "apply_nssl2_production(state, cfg, dt, hooks, ...), a different "
            "entry point with its own hook contract.  Binding it is a seam "
            "extension, not a species-table change.  CM1's own ptype=26/27 "
            "axis is dynamic (src/param.F:3736-3810 builds it from lhab and "
            "the hail/CCN switches), so the axis transcription is real work too"
        ),
    ),
    ColumnComponentBlocker(
        component_id="wdm6-mp16",
        scheme_name="WDM6",
        mp_physics=16,
        cm1_ptype=None,
        blocker="arwen-wdm6-requires-an-xland-land-sea-mask",
        detail=(
            "measured, not inferred: dispatching a WDM6 column through the "
            "CM1-owned adapter refuses inside ArWen at "
            "gpuwm/core/wdm6.py:156 -- \"mp_physics=16 (WDM6) needs WRF's XLAND "
            "land/sea mask for its per-column autoconversion threshold "
            "(module_mp_wdm6.F:607-614) and the physics driver carries none; "
            "refusing to substitute an all-land or all-water mask\".  ArWen "
            "builds XLAND in gpuwm.core.physics.initialize_physics from a "
            "landmask, which the CM1-owned column state does not construct and "
            "which CM1's idealized Cartesian lane has no field for at all.  "
            "The nine-scalar axis, the carrier map and the shared WSM6 "
            "rimed-ice constants are all transcribed and tested "
            "(cm1core.arwen_scheme_components.WDM6_SPEC); what is missing is a "
            "surface field, and giving the scheme one this lane does not model "
            "would silently pick its autoconversion threshold"
        ),
    ),
    ColumnComponentBlocker(
        component_id="p3-mp50",
        scheme_name="P3 one-category",
        mp_physics=50,
        cm1_ptype=50,
        blocker="six-mass-species-post-physics",
        detail=(
            "CM1 ptype=50's axis is qv,qc,qr,qi,qni,qnr,qir,qib with nqs1=nqs2=4 "
            "(src/param.F:4102-4128), so its mass prefix is four and two of its "
            "remaining rows are neither kg/kg nor #/kg -- qir is a rime mass and "
            "qib a rime volume in m^-3 kg^-1.  P3 also wants two component-owned "
            "volumes, th_old and qv_old (gpuwm/core/p3.py:1729), which the "
            "existing component_owned_carriers mechanism already allocates"
        ),
    ),
)

COLUMN_COMPONENT_BLOCKERS: Mapping[str, ColumnComponentBlocker] = MappingProxyType(
    {blocker.component_id: blocker for blocker in _BLOCKERS}
)


def blocked_component_records() -> list[dict[str, object]]:
    """Every blocked component, in a stable order, for a receipt or doctor."""

    return [blocker.record() for blocker in _BLOCKERS]


def refuse_blocked_component(component_id: str, *, option: str) -> None:
    """Refuse a known-but-blocked component by naming its one blocker.

    A caller reaches this only for an id that is in
    :data:`COLUMN_COMPONENT_BLOCKERS`; anything else is refused by the generic
    unsupported-component path, which says a different and correct thing.
    """

    blocker = COLUMN_COMPONENT_BLOCKERS[component_id]
    refuse(
        code="GM1-COLUMN-COMPONENT-BLOCKED",
        gate="column-component",
        reason=(
            f"{blocker.scheme_name} (ArWen mp_physics={blocker.mp_physics}) is "
            f"compiled and reachable inside the admitted ArWen estate, but GM1 "
            f"cannot select it yet: {blocker.blocker}.  {blocker.detail}.  "
            "Binding it without that work would stage a scalar axis GM1 has no "
            "publication path for, so the run would either refuse mid-step or "
            "publish a frame whose species inventory does not match the scheme "
            "that produced it"
        ),
        remedy=(
            "select a bound component, or land the named blocker: "
            "src/cm1core/column_component_blockers.py carries the exact change"
        ),
        option=option,
        value=component_id,
    )


__all__ = [
    "COLUMN_COMPONENT_BLOCKERS",
    "COLUMN_COMPONENT_BLOCKER_SCHEMA",
    "ColumnComponentBlocker",
    "blocked_component_records",
    "refuse_blocked_component",
]
