from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_import_does_not_load_arwen_or_cuda() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    script = (
        "import sys, cm1core, cm1core.state, cm1core.zero_step; "
        "assert 'gpuwm' not in sys.modules; "
        "assert 'cupy' not in sys.modules; "
        "print(cm1core.MATURITY)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "port-in-progress"
