"""Asset skips for tests whose own file bytes a sealed receipt pins.

Eight ``cm1-*-source-exact-v1`` validation receipts record the byte size and
sha256 of the test file that checks them, and that file's
``test_validation_receipt_binds_cpu_only_fail_closed_port`` re-verifies the
pin.  Decorating the authority-pin test inside such a file with
``requires_cm1_pristine`` would change the pinned bytes and fail the receipt
test, so the marker is applied here, by node id, at collection time instead.

Register from ``tests/conftest.py`` with::

    from receipt_pinned_asset_skips import pytest_collection_modifyitems  # noqa: F401,E402
"""

from __future__ import annotations

from authority_trees import requires_cm1_pristine, requires_native_reference

#: Tests that read ``upstream/cm1-21.1/pristine`` and live in a receipt-pinned file.
PRISTINE_NODE_IDS = frozenset(
    {
        "tests/test_dns_temperature_diffusion.py::test_authority_and_complete_bounded_source_blocks_are_pinned",
        "tests/test_idealized_radiation.py::test_authority_and_all_source_dispatch_anchors_are_pinned",
        "tests/test_large_scale_nudging.py::test_authority_and_bounded_source_order_are_pinned",
        "tests/test_prescribed_surface_flux.py::test_authority_and_complete_bounded_source_blocks_are_pinned",
        "tests/test_scalar_artificial_diffusion.py::test_authority_and_bounded_scalar_branches_are_pinned",
        "tests/test_sgs_startup_ramp.py::test_authority_and_complete_source_block_are_pinned",
        "tests/test_surface_exchange_coefficients.py::test_authority_and_complete_bounded_source_blocks_are_pinned",
        "tests/test_surface_initialization.py::test_authority_and_exact_pre_landuse_boundary_are_pinned",
    }
)

#: Tests that read ``work/native-reference/cm1r21.1`` (``CM1_AUTHORITY_ROOT``).
#: The first two live in files whose bytes cuda-advection5-v1 and
#: cuda-scalar-moist-rk-v1 pin; the third calls the oracle verifier, which reads
#: the same tree.  All three pass when the tree is supplied.
NATIVE_REFERENCE_NODE_IDS = frozenset(
    {
        "tests/test_cuda_advection5.py::test_authority_sources_are_exact_cm1_r21_1_bytes",
        "tests/test_cuda_scalar_moist_rk.py::test_authority_sources_are_exact_cm1_r21_1_bytes",
        "tests/test_cuda_scalar_moist_rk_evidence.py::test_native_scalar_moist_rk_oracle_and_negative_controls_are_sealed",
    }
)


def pytest_collection_modifyitems(config, items) -> None:  # noqa: ANN001
    for item in items:
        # Match on the file name and test name, so the id is the same whatever
        # directory pytest was invoked from.
        key = f"tests/{item.path.name}::{item.name}"
        if key in PRISTINE_NODE_IDS:
            item.add_marker(requires_cm1_pristine)
        if key in NATIVE_REFERENCE_NODE_IDS:
            item.add_marker(requires_native_reference)
