#!/usr/bin/env python3
"""Print the t=0 environment parameters of one configuration's base state.

This is the block every run already prints and records -- `base.F:3143-3189`
over `getcape.F:14-594`, plus the bulk shear, Bunkers et al. (2000) storm
motion and storm-relative helicity CM1 has no equivalent for -- run on its own
so a sounding can be examined before a GPU is asked for.  Nothing here opens a
device; the whole computation is the Rust initializer's base-state column.

    tools/report_sounding_parameters.py CONFIG.toml --sounding input_sounding

With ``--json`` the canonical record goes to stdout exactly as it appears in
the run receipt under ``initial_sounding_parameters``.

Unsupported source tool: run it from a clone of the repository with gpuwm-cm1 installed
(pip install gpuwm-cm1). It uses the installed package when there is one and this
checkout's src/ otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
# The installed gpuwm-cm1 when there is one (a repository clone carries no built native
# tools); this checkout's src/ otherwise.
if importlib.util.find_spec("cm1core") is None:
    sys.path.insert(0, str(ROOT / "src"))

from cm1core.cartesian_door import (  # noqa: E402
    require_initializer_selection,
    require_sounding_report,
    resolve_initializer_library,
    resolve_sounding,
)
from cm1core.config import load_config  # noqa: E402
from cm1core.errors import CM1Refusal  # noqa: E402
from cm1core.sounding_report import format_sounding_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="the run's TOML document")
    parser.add_argument(
        "--sounding",
        type=Path,
        default=None,
        help="the input_sounding this run would read; omitted on an analytic lane",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=None,
        help="the cm1-state initializer library; resolved from the package when omitted",
    )
    parser.add_argument("--json", action="store_true", help="emit the receipt record")
    arguments = parser.parse_args(argv)

    try:
        config = load_config(arguments.config)
        selection, _ = require_initializer_selection(config)
        library = resolve_initializer_library(arguments.library)
        if selection.reads_input_sounding:
            _, sounding_bytes = resolve_sounding(arguments.sounding)
        else:
            sounding_bytes = b""
        report = require_sounding_report(
            config,
            library=library,
            sounding_bytes=sounding_bytes,
            selection=selection,
        )
    except CM1Refusal as refusal:
        print(str(refusal), file=sys.stderr)
        return 2
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(format_sounding_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
