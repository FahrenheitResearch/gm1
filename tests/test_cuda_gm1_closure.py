from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from cm1core.cuda_gm1_closure import (
    GM1_CLOSURE_SCHEMA,
    GM1_CLOSURE_SOURCE_ORDER,
)


ROOT = Path(__file__).resolve().parents[1]


def test_closure_import_is_cuda_cold_and_gm1_branded() -> None:
    command = (
        "import sys; import cm1core.cuda_gm1_closure as module; "
        "assert module.GM1_CLOSURE_SCHEMA.startswith('gm1.'); "
        "assert not any(name == 'cupy' or name.startswith('cupy.') "
        "for name in sys.modules)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert GM1_CLOSURE_SCHEMA == "gm1.cuda-les-closure/v1"


def test_closure_source_order_covers_every_large_step_cadence() -> None:
    assert GM1_CLOSURE_SOURCE_ORDER == (
        "post-step:calcnm",
        "post-step:calcdef",
        "post-step:tkekm-tkebc-gettau-state",
        "solve1:rayleigh-held",
        "solve1:sgs-momentum-stress",
        "solve1:theta-and-six-moisture-scalar-turbulence",
        "solve1:eqtset2-idiss",
        "solve1:tke-diffusion-held",
        "solve2:nrk1-tke-advection-rk",
        "solve2:nrk2-tke-advection-rk",
        "solve2:nrk3-tke-advection-rk",
        "solve3:tkea-commit",
    )


def test_forecast_has_no_zero_closure_product_path() -> None:
    source = (ROOT / "src/cm1core/cuda_gm1_forecast.py").read_text(encoding="utf-8")
    assert "builtin_closure = launch_cuda_gm1_closure_solve1" in source
    assert "closure_tke = launch_cuda_gm1_closure_stage" in source
    assert "closure_commit = finish_cuda_gm1_closure_step" in source
    assert "closure_refresh = refresh_cuda_gm1_closure_for_next_step" in source
    assert "No zero-closure product run is accepted" in source

