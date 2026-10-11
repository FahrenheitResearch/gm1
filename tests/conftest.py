from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

# No test reads or writes the developer's own measured GPU rates: every run
# the suite makes (and the door learns from) goes to a cache of its own.
import atexit  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

if "GPUWM_CM1_CACHE_DIR" not in os.environ:
    _TEST_CACHE = tempfile.mkdtemp(prefix="gpuwm-cm1-test-cache-")
    os.environ["GPUWM_CM1_CACHE_DIR"] = _TEST_CACHE
    atexit.register(shutil.rmtree, _TEST_CACHE, ignore_errors=True)

# thompson-aerosol-mp28 is not offered by the distribution (the Cartesian
# door refuses it unless this developer switch is on); the suite keeps
# exercising its code path.  test_unlisted_component.py clears it.
os.environ.setdefault("GPUWM_CM1_UNLISTED_MP28", "1")

# Under WSL every run GM1 starts also starts a hidden Windows-side holder
# (cm1run.wsl_keepalive); the suite's runs start none.
os.environ.pop("WSL_DISTRO_NAME", None)

from authority_trees import link_supplied_assets  # noqa: E402
from receipt_pinned_asset_skips import pytest_collection_modifyitems  # noqa: F401,E402
# In the public source snapshot (no evidence/), a test that reads a tree kept in
# the development repository skips and names it; with evidence/ present this is inert.
from private_tree_skips import (  # noqa: F401,E402
    pytest_make_collect_report,
    pytest_runtest_makereport,
    pytest_runtest_setup,
)

# Before collection, so skipif markers evaluated at import see the links.
link_supplied_assets()
