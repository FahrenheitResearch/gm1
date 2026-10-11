"""Write the GrADS / parcel reference namelists: the conformance namelist with each variant's edits.

usage: python namelists.py OUT_DIR   (run from the repository root)
"""

import json
import re
import sys
from pathlib import Path

REF = Path("tests/data/cm1-ref-24x24")
out = Path(sys.argv[1])
source = (REF / "namelist.input").read_text()
for name, edits in json.loads((REF / "grads" / "variants.json").read_text()).items():
    text = source
    for key, value in edits.items():
        text, n = re.subn(rf"^(\s*{key}\s*=\s*)([^,\n]*)(,?)", lambda m: f"{m.group(1)}{value}{m.group(3)}",
                          text, count=1, flags=re.M)
        if n == 0:  # a key the conformance namelist leaves at its default (outunits)
            text, n = re.subn(r"^( nz\s*=.*\n)", lambda m: m.group(1) + f" {key} = {value},\n", text,
                              count=1, flags=re.M)
        assert n == 1, (name, key)
    (out / name).mkdir(parents=True, exist_ok=True)
    (out / name / "namelist.input").write_text(text)
