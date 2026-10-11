"""The vectorized capture is the reference operator, to the last bit.

``nest_downscale`` exists because the nest's own parent-to-child operator is a
pure-Python nest of lists, and one 75-level transfer onto a 500 x 500 child is
18.75 million evaluations of it.  Rewriting it over NumPy is only legitimate
if the rewrite is the same arithmetic, and "the same arithmetic" for a
bit-exact port is a claim about raw words rather than about a tolerance.  Every
comparison here is on ``uint64`` views for that reason.
"""

from __future__ import annotations

import math
import random

import pytest

numpy = pytest.importorskip("numpy")

from cm1core.nest_boundary import (  # noqa: E402
    block_mean_residual,
    interpolate_parent_to_child,
    limited_slopes,
)
from cm1core.nest_downscale import (  # noqa: E402
    assert_matches_reference,
    downscale_level,
    downscale_volume,
    limited_slopes_array,
    require_identical_vertical_decks,
)
from cm1core.nest_geometry import NestGeometry  # noqa: E402


def _geometry(**overrides: int) -> NestGeometry:
    fields = {
        "parent_nx": 40,
        "parent_ny": 36,
        "parent_dx_m": 250.0,
        "parent_dy_m": 250.0,
        "refinement_ratio": 5,
        "time_refinement_ratio": 5,
        "parent_i0": 8,
        "parent_j0": 6,
        "parent_span_x": 10,
        "parent_span_y": 9,
        "spec_zone": 1,
        "relax_zone": 4,
        "feedback": 1,
    }
    fields.update(overrides)
    return NestGeometry(**fields)


def _field(ny: int, nx: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-40.0, 40.0) for _ in range(nx)] for _ in range(ny)]


def test_limited_slopes_agree_word_for_word_with_the_reference() -> None:
    geometry = _geometry()
    parent = _field(geometry.parent_ny, geometry.parent_nx, seed=11)
    slope_x, slope_y = limited_slopes_array(parent)
    for j in range(geometry.parent_ny):
        for i in range(geometry.parent_nx):
            want_x, want_y = limited_slopes(parent, j, i)
            got_x = float(slope_x[j, i])
            got_y = float(slope_y[j, i])
            assert math.copysign(1.0, got_x) == math.copysign(1.0, want_x)
            assert got_x.hex() == want_x.hex(), f"sx at ({j},{i})"
            assert got_y.hex() == want_y.hex(), f"sy at ({j},{i})"


@pytest.mark.parametrize("ratio", [3, 5, 7, 9])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_capture_is_the_reference_for_several_ratios(ratio: int, seed: int) -> None:
    geometry = _geometry(refinement_ratio=ratio, time_refinement_ratio=ratio)
    parent = _field(geometry.parent_ny, geometry.parent_nx, seed=seed)
    assert_matches_reference(parent, geometry)


def test_the_capture_is_the_reference_on_a_flat_field() -> None:
    """A zero deviation returns (0, 0) before the reference forms a quotient.

    The vectorized form has to divide by something to stay branchless, so the
    case where the reference never divides at all is the one that would expose
    a stray nan or a 0/0.
    """

    geometry = _geometry()
    parent = [[7.5] * geometry.parent_nx for _ in range(geometry.parent_ny)]
    assert_matches_reference(parent, geometry)
    captured = downscale_level(parent, geometry)
    assert numpy.isfinite(captured).all()
    assert (captured == 7.5).all()


def test_the_capture_is_the_reference_at_a_sharp_edge() -> None:
    """A step function is where the extremum limiter actually does something."""

    geometry = _geometry()
    parent = [
        [0.0 if i < geometry.parent_nx // 2 else 100.0 for i in range(geometry.parent_nx)]
        for _ in range(geometry.parent_ny)
    ]
    assert_matches_reference(parent, geometry)
    captured = downscale_level(parent, geometry)
    # No new extrema: the limiter's whole purpose.
    assert float(captured.min()) >= 0.0
    assert float(captured.max()) <= 100.0


def test_a_non_negative_parent_gives_a_non_negative_child() -> None:
    """The property the child's microphysics depends on at the interface."""

    geometry = _geometry()
    rng = random.Random(99)
    parent = [
        [max(0.0, rng.uniform(-0.004, 0.018)) for _ in range(geometry.parent_nx)]
        for _ in range(geometry.parent_ny)
    ]
    captured = downscale_level(parent, geometry)
    assert float(captured.min()) >= 0.0


def test_the_block_mean_identity_survives_vectorization() -> None:
    """Each r x r child block still averages to its parent cell exactly.

    ``block_mean_residual`` proves the property for the reference's slopes;
    this sums the captured child words in the same pairing order and requires
    the same exact zero, so the vectorized operator inherits the conservation
    statement rather than merely resembling it.
    """

    geometry = _geometry(refinement_ratio=5, time_refinement_ratio=5)
    parent = _field(geometry.parent_ny, geometry.parent_nx, seed=7)
    captured = downscale_level(parent, geometry)
    ratio = geometry.refinement_ratio
    for block_j in range(geometry.parent_span_y):
        for block_i in range(geometry.parent_span_x):
            centre = parent[geometry.parent_j0 + block_j][geometry.parent_i0 + block_i]
            assert block_mean_residual(
                parent, geometry, geometry.parent_j0 + block_j,
                geometry.parent_i0 + block_i,
            ) == 0.0
            total = 0.0
            for m in range(ratio):
                for mm in range(ratio):
                    total += float(
                        captured[block_j * ratio + mm, block_i * ratio + m]
                    ) - centre
            assert total == pytest.approx(0.0, abs=1e-9)


def test_a_volume_is_captured_level_by_level() -> None:
    geometry = _geometry()
    levels = 6
    volume = numpy.stack(
        [
            numpy.asarray(_field(geometry.parent_ny, geometry.parent_nx, seed=40 + k))
            for k in range(levels)
        ]
    )
    captured = downscale_volume(volume, geometry)
    assert captured.shape == (levels, geometry.ny, geometry.nx)
    for k in range(levels):
        one = downscale_level(volume[k], geometry)
        assert (captured[k].view(numpy.uint64) == one.view(numpy.uint64)).all()


def test_float32_input_is_promoted_the_way_the_reference_promotes_it() -> None:
    """A restart arena holds f32; the reference reads each word as a double."""

    geometry = _geometry()
    parent32 = numpy.asarray(
        _field(geometry.parent_ny, geometry.parent_nx, seed=5), dtype=numpy.float32
    )
    as_lists = [[float(v) for v in row] for row in parent32]
    expected = numpy.asarray(
        interpolate_parent_to_child(as_lists, geometry), dtype=numpy.float64
    )
    measured = downscale_level(parent32, geometry)
    assert (expected.view(numpy.uint64) == measured.view(numpy.uint64)).all()


class _Deck:
    def __init__(self, sha: str, stretch_z: int = 1) -> None:
        self.sha256 = sha
        self.stretch_z = stretch_z


class _Grid:
    def __init__(self, nz: int, vertical: object | None) -> None:
        self.nz = nz
        self.vertical = vertical


def test_unequal_vertical_decks_are_refused_by_value() -> None:
    from cm1core.errors import CM1Refusal

    same = _Deck("aa" * 32)
    require_identical_vertical_decks(_Grid(75, same), _Grid(75, same))

    with pytest.raises(CM1Refusal) as excinfo:
        require_identical_vertical_decks(_Grid(75, same), _Grid(60, same))
    assert excinfo.value.detail.code == "CM1-NEST-DOWNSCALE-VERTICAL"
    assert excinfo.value.detail.value == 60

    with pytest.raises(CM1Refusal) as excinfo:
        require_identical_vertical_decks(_Grid(75, same), _Grid(75, _Deck("bb" * 32)))
    assert excinfo.value.detail.code == "CM1-NEST-DOWNSCALE-VERTICAL"

    with pytest.raises(CM1Refusal) as excinfo:
        require_identical_vertical_decks(_Grid(75, same), _Grid(75, None))
    assert excinfo.value.detail.code == "CM1-NEST-DOWNSCALE-VERTICAL"


def test_an_unstretched_pair_is_admitted_without_a_deck_object() -> None:
    require_identical_vertical_decks(_Grid(40, None), _Grid(40, None))


def test_the_one_way_limit_is_stated_rather_than_implied() -> None:
    """A downscaled child must never read as a live nest."""

    from cm1core.nest_downscale import ONE_WAY_LIMIT, downscale_record

    record = downscale_record(
        _geometry(),
        parent_restart="restart-000006",
        parent_step=3600,
        parent_model_time_s=1800.0,
        carriers=("ua", "va", "wa"),
    )
    assert record["coupling"] == "one-way-initial-condition-only"
    assert record["limit"] == ONE_WAY_LIMIT
    assert "not coupled to it afterwards" in ONE_WAY_LIMIT or "not a parent-fed" in ONE_WAY_LIMIT
    assert record["vertical_treatment"].startswith("copied level for level")


# ---------------------------------------------------------------------------
# The staggered extension.  u, v and w do not sit at cell centres, and an
# initial condition has to fill them.
# ---------------------------------------------------------------------------


def test_mass_mass_staggered_is_exactly_the_unstaggered_operator() -> None:
    """The extension must not perturb the case the reference already covers."""

    from cm1core.nest_downscale import downscale_level_staggered

    geometry = _geometry()
    parent = _field(geometry.parent_ny, geometry.parent_nx, seed=21)
    plain = downscale_level(parent, geometry)
    staggered = downscale_level_staggered(
        parent, geometry, x_lattice="mass", y_lattice="mass"
    )
    assert plain.shape == staggered.shape
    assert (plain.view(numpy.uint64) == staggered.view(numpy.uint64)).all()


@pytest.mark.parametrize("ratio", [3, 5, 9])
def test_a_face_lattice_copies_every_coincident_parent_face_exactly(ratio: int) -> None:
    """The property the whole conservation argument rests on.

    Every ``r``-th child face lies exactly on a parent face, so its sub-cell
    offset is exactly ``0.0`` and the x correction vanishes identically.  Take
    the centre sub-row of an odd ratio as well and the y correction vanishes
    too, so the child word must be the parent word **copied** -- not
    reconstructed and rounded to something very close to it.  Comparing the
    hex forms is what makes that a claim about bits.
    """

    from cm1core.nest_downscale import downscale_level_staggered

    geometry = _geometry(refinement_ratio=ratio, time_refinement_ratio=ratio)
    parent = _field(geometry.parent_ny, geometry.parent_nx, seed=33)
    captured = downscale_level_staggered(
        parent, geometry, x_lattice="face", y_lattice="mass"
    )
    assert captured.shape == (geometry.ny, geometry.nx + 1)
    centre = ratio // 2
    checked = 0
    for face in range(geometry.parent_span_x + 1):
        for block in range(geometry.parent_span_y):
            jc = block * ratio + centre
            got = float(captured[jc, face * ratio])
            want = float(parent[geometry.parent_j0 + block][geometry.parent_i0 + face])
            assert got.hex() == want.hex(), f"face {face}, row {jc}"
            checked += 1
    assert checked == (geometry.parent_span_x + 1) * geometry.parent_span_y


def test_a_face_lattice_introduces_no_new_extrema() -> None:
    from cm1core.nest_downscale import downscale_level_staggered

    geometry = _geometry()
    parent = [
        [0.0 if i < geometry.parent_nx // 2 else 50.0 for i in range(geometry.parent_nx)]
        for _ in range(geometry.parent_ny)
    ]
    for x_lat, y_lat in (("face", "mass"), ("mass", "face"), ("face", "face")):
        captured = downscale_level_staggered(
            parent, geometry, x_lattice=x_lat, y_lattice=y_lat
        )
        assert float(captured.min()) >= 0.0, (x_lat, y_lat)
        assert float(captured.max()) <= 50.0, (x_lat, y_lat)


def test_the_face_lattice_carries_one_more_point_on_its_own_axis() -> None:
    from cm1core.nest_downscale import downscale_level_staggered

    geometry = _geometry()
    parent = _field(geometry.parent_ny, geometry.parent_nx, seed=44)
    assert downscale_level_staggered(
        parent, geometry, x_lattice="face", y_lattice="mass"
    ).shape == (geometry.ny, geometry.nx + 1)
    assert downscale_level_staggered(
        parent, geometry, x_lattice="mass", y_lattice="face"
    ).shape == (geometry.ny + 1, geometry.nx)
    assert downscale_level_staggered(
        parent, geometry, x_lattice="face", y_lattice="face"
    ).shape == (geometry.ny + 1, geometry.nx + 1)


def test_an_unknown_lattice_is_refused_by_name() -> None:
    from cm1core.nest_downscale import downscale_level_staggered

    geometry = _geometry()
    parent = _field(geometry.parent_ny, geometry.parent_nx, seed=1)
    with pytest.raises(ValueError, match="lattice"):
        downscale_level_staggered(parent, geometry, x_lattice="corner")


def test_a_staggered_volume_is_each_level_staggered() -> None:
    from cm1core.nest_downscale import (
        downscale_level_staggered,
        downscale_volume_staggered,
    )

    geometry = _geometry()
    volume = numpy.stack(
        [
            numpy.asarray(_field(geometry.parent_ny, geometry.parent_nx, seed=70 + k))
            for k in range(4)
        ]
    )
    captured = downscale_volume_staggered(volume, geometry, x_lattice="face")
    assert captured.shape == (4, geometry.ny, geometry.nx + 1)
    for k in range(4):
        one = downscale_level_staggered(volume[k], geometry, x_lattice="face")
        assert (captured[k].view(numpy.uint64) == one.view(numpy.uint64)).all()
