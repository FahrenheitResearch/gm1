"""The flat acoustic RK stage with the run's lateral step: sound.F's lateral calls in sound.F's positions.

:func:`cm1core.cuda_gm1_child_psolver3_open.launch_cuda_gm1_child_psolver3_open_stage` is a sealed operator (the
real child's operator manifest pins its bytes), and its flat (``terrain_flag = .false.``) stage was never handed
the run's lateral family: on stretched levels ``irbc = 2`` rebuilt no phase speeds on the small steps,
``roflux = 1`` rescaled nothing, and under ``irbc = 1`` or ``2`` the Durran-Klemp histories of
solve2.F:1258-1271 overwrote the Klemp-Wilhelmson phase speeds.  :func:`launch_cuda_gm1_flat_acoustic_stage` is
that launcher with the run's :class:`cm1core.cuda_boundary_families.LateralBoundaryStep`:

* ``irbc = 2`` rebuilds the Klemp-Wilhelmson phase speeds from ``u3d``/``v3d`` at the top of every small step
  (sound.F:260-268, bc.F:1948-2053);
* ``roflux = 1`` rescales the outward normal mass flux after the open-side update and before the pressure
  gradient (sound.F:283-302, bc.F:2180-2391), with ``rho0*u3d`` rounded to binary32 first as bc.F:2254 forms it;
* the Durran-Klemp histories are produced for ``irbc = 4`` only.

The small-step loop is derived from the sealed one by the two insertions above (``_LATERAL_EDITS``, each
anchored on exactly one place of the sealed text), so every other statement is the sealed loop's own.  The
sealed launcher still does the validation, the compiled images, the workspace and the receipt; for the time of
one launch it runs the derived loop.  With no lateral step (the all-open, ``irbc = 4`` family) the sealed
launcher runs untouched.  ``ssopenbc`` runs on both axes in one launch where CM1 runs ``restrict_openbc_we``
between the west/east and the south/north updates (sound.F:274-302); the two orders write the same words,
because ``restrict_openbc_we`` reads and writes ``u3d`` alone and the south/north update ``v3d`` alone.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import textwrap
from typing import Any, Iterator

from . import cuda_gm1_child_psolver3_open as _sealed

#: (anchor in the sealed loop, the anchor with the lateral call put in its place).
_LATERAL_EDITS: tuple[tuple[str, str], ...] = (
    (
        "def _enqueue_flat_stage(\n",
        "def _enqueue_flat_stage_lateral(\n",
    ),
    (
        "    obc: int,\n) -> Any:\n",
        "    obc: int,\n    lateral: Any,\n) -> Any:\n",
    ),
    (
        "                nest_substep.begin(nrk=nrk, n=n, dts=float(dts))\n"
        "            if ablations.open_consumption:\n",
        "                nest_substep.begin(nrk=nrk, n=n, dts=float(dts))\n"
        "            # sound.F:260-268: irbc=2 rebuilds the Klemp-Wilhelmson phase\n"
        "            # speeds from u3d/v3d at the top of every small step.\n"
        "            if lateral.selection.irbc == 2:\n"
        "                _bc.enqueue_radbc(\n"
        "                    a, ni=ni, nj=nj, nk=nk, ng=CM1_OPEN_NGXY, selection=lateral.selection,\n"
        "                    compiled=lateral.compiled, stream=stream, momentum=\"3d\",\n"
        "                )\n"
        "            if ablations.open_consumption:\n",
    ),
    (
        "                        a[\"u3d\"], a[\"v3d\"], *dims, dts, rdx, rdy, u32(obc),\n"
        "                    ),\n"
        "                    stream=stream,\n"
        "                )\n",
        "                        a[\"u3d\"], a[\"v3d\"], *dims, dts, rdx, rdy, u32(obc),\n"
        "                    ),\n"
        "                    stream=stream,\n"
        "                )\n"
        "                # sound.F:283-302, in CM1's own position: after the open\n"
        "                # normal-velocity step and before the pressure gradient, so the\n"
        "                # rescaled face is what this small step's divergence sees.\n"
        "                if lateral.selection.roflux == 1:\n"
        "                    _bc.enqueue_restrict_outward_flux(\n"
        "                        a, m, lateral.workspace, ni=ni, nj=nj, nk=nk, ng=CM1_OPEN_NGXY,\n"
        "                        selection=lateral.selection, compiled=lateral.compiled, stream=stream,\n"
        "                        apply_compiled=lateral.roflux_cm1_order,\n"
        "                    )\n",
    ),
)


def lateral_stage_source() -> str:
    """The derived small-step loop's text: the sealed loop with ``_LATERAL_EDITS`` applied."""

    text = textwrap.dedent(inspect.getsource(_sealed._enqueue_flat_stage))
    for old, new in _LATERAL_EDITS:
        if text.count(old) != 1:
            raise RuntimeError(
                f"the sealed flat acoustic loop does not hold this anchor exactly once: {old[:60]!r}"
            )
        text = text.replace(old, new)
    return text


def _derive() -> Any:
    from . import cuda_boundary_families

    namespace = dict(vars(_sealed))
    namespace["_bc"] = cuda_boundary_families
    code = compile(lateral_stage_source(), f"<{_sealed.__name__}._enqueue_flat_stage with the lateral step>", "exec")
    exec(code, namespace)  # noqa: S102 - the sealed module's own text, with the edits above
    return namespace["_enqueue_flat_stage_lateral"]


_enqueue_flat_stage_lateral = _derive()


@contextlib.contextmanager
def _flat_loop_with(lateral: Any) -> Iterator[None]:
    sealed_loop = _sealed._enqueue_flat_stage
    _sealed._enqueue_flat_stage = functools.partial(_enqueue_flat_stage_lateral, lateral=lateral)
    try:
        yield
    finally:
        _sealed._enqueue_flat_stage = sealed_loop


def launch_cuda_gm1_flat_acoustic_stage(
    arrays: Any, metrics: Any, *, lateral: Any | None = None, **kwargs: Any
) -> Any:
    """:func:`~cm1core.cuda_gm1_child_psolver3_open.launch_cuda_gm1_child_psolver3_open_stage`, handed ``lateral``.

    ``lateral`` is the run's :class:`~cm1core.cuda_boundary_families.LateralBoundaryStep`, or ``None`` for the
    all-open, ``irbc = 4`` family, for which the sealed launcher runs as it always has.  Only the flat stage
    takes a lateral step; the sides it names must be the ``lateral_sides`` the stage is handed.
    """

    if lateral is None:
        return _sealed.launch_cuda_gm1_child_psolver3_open_stage(arrays, metrics, **kwargs)
    vertical = kwargs.get("vertical")
    if vertical is None or not getattr(vertical, "flat_column", None):
        raise ValueError(
            "the terrain child acoustic stage is sealed to the all-open Durran-Klemp family; "
            "only the flat stage takes a lateral step"
        )
    sides = tuple(int(side) for side in kwargs.get("lateral_sides", (2, 2, 2, 2)))
    if tuple(int(side) for side in lateral.selection.sides) != sides:
        raise ValueError(
            f"the lateral step names sides {lateral.selection.sides} but the stage was handed {sides}; "
            "one run has one family"
        )
    ablations = kwargs.get("ablations") or _sealed.TerrainOpenSoundAblations()
    # solve2.F:1258-1271 produces the Durran-Klemp histories for irbc=4 only;
    # irbc=1 and irbc=2 build radbc* from the Klemp-Wilhelmson formula
    # (bc.F:1948-2053), which a production would overwrite.
    kwargs["ablations"] = dataclasses.replace(
        ablations,
        radiation_history_production=ablations.radiation_history_production and lateral.selection.irbc == 4,
    )
    with _flat_loop_with(lateral):
        launch = _sealed.launch_cuda_gm1_child_psolver3_open_stage(arrays, metrics, **kwargs)
    launch.receipt["lateral_step"] = {
        **lateral.record(),
        "small_step_radbc": lateral.selection.irbc == 2,
        # The rescale acts on open sides only; with none, nothing is enqueued.
        "restrict_outward_flux": lateral.selection.roflux == 1 and lateral.selection.any_open,
        "durran_klemp_production": lateral.selection.irbc == 4,
        "anchors": "sound.F:260-268, 283-302; bc.F:1948-2053, 2180-2391; solve2.F:1258-1271",
        "small_step_loop": "derived from the sealed flat loop (cuda_gm1_flat_lateral_acoustic._LATERAL_EDITS)",
    }
    return launch


__all__ = ["launch_cuda_gm1_flat_acoustic_stage", "lateral_stage_source"]
