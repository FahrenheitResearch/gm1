#!/usr/bin/env python3
"""A stand-in for ``nvidia-smi`` that answers the estimator's two queries.

The fixture is the JSON file named by ``FAKE_NVIDIA_SMI_FIXTURE``: a list of
``gpus`` (each a mapping of nvidia-smi field names to the values nvidia-smi
would print with ``nounits``), a list of ``apps`` and, optionally,
``unsupported_fields`` (fields an old driver rejects) or ``fail`` (an exit
code and message, as a broken driver answers).
"""

import json
import os
import sys


def main(argv: list[str]) -> int:
    fixture = json.loads(open(os.environ["FAKE_NVIDIA_SMI_FIXTURE"], encoding="utf-8").read())
    if "fail" in fixture:
        print(fixture["fail"]["message"], file=sys.stderr)
        return int(fixture["fail"]["code"])
    query = next((arg for arg in argv if arg.startswith("--query-")), None)
    if query is None:
        print("fake nvidia-smi answers only --query-gpu and --query-compute-apps", file=sys.stderr)
        return 2
    kind, _, fields = query[len("--query-"):].partition("=")
    names = fields.split(",")
    for name in names:
        if name in fixture.get("unsupported_fields", ()):
            print(f'Field "{name}" is not a valid field to query.', file=sys.stdout)
            return 2
    rows = fixture.get("gpus" if kind == "gpu" else "apps", [])
    for row in rows:
        print(", ".join(str(row.get(name, "[N/A]")) for name in names))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
