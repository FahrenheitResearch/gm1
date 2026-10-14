"""The native renderer's products for a run folder, in plain words (no numpy: the command line's help reads it).

:mod:`cm1core.render_restart` draws them; ``gpuwm-cm1 render --help``, the GUI and the docs list them from here.
"""

from __future__ import annotations

import functools
from pathlib import Path

#: Every product the native renderer draws from a restart, in the order help and the GUI list them.
PRODUCTS: tuple[str, ...] = (
    "reflectivity", "reflectivity-level", "updraft", "vorticity", "cold-pool", "scalar",
    "supercell", "vortex-composite", "storm-fusion", "updraft-helicity",
    "swath-wind", "swath-vorticity", "swath-pressure", "swath-helicity",
)
#: The products whose picture is one horizontal level (``--level K`` or ``--height M``).
LEVEL_PRODUCTS = ("reflectivity-level", "vorticity", "cold-pool", "scalar", "supercell", "vortex-composite",
                  "storm-fusion")
#: The environment planes the envelope route draws; a restart carries none of them.
ENVIRONMENT_PRODUCTS = ("cape", "cin", "lcl", "lfc")
#: Plain words for each product, for help, the GUI and the receipt.
PRODUCT_WORDS = {
    "reflectivity": "Composite reflectivity (column maximum, NWS colours)",
    "reflectivity-level": "Reflectivity at one height (default 1 km)",
    "updraft": "Strongest updraft in each column",
    "vorticity": "Vertical vorticity at one level (default the lowest)",
    "cold-pool": "Potential temperature perturbation at one level (the cold pool)",
    "scalar": "Any restart field at one level (--field)",
    "supercell": "Three panels: reflectivity, updraft, vorticity",
    "vortex-composite": "Reflectivity, updraft and spin blended in one picture",
    "storm-fusion": "Reflectivity, updraft and spin, a second blend",
    "updraft-helicity": "2-5 km updraft helicity (CM1's sum)",
    "swath-wind": "Strongest wind at the lowest level since the start",
    "swath-vorticity": "Strongest spin at the lowest level since the start",
    "swath-pressure": "Deepest pressure drop at the lowest level since the start",
    "swath-helicity": "Strongest updraft helicity since the start",
}


#: The products that draw reflectivity, which a restart has only where GM1 computes it on the CPU.
REFLECTIVITY_PRODUCTS = ("reflectivity", "reflectivity-level", "supercell", "vortex-composite", "storm-fusion")
#: The swath each swath product draws (CM1's names, as a restart's receipt lists the swaths it carried).
SWATH_NAMES = {"swath-wind": "sws", "swath-vorticity": "svs", "swath-pressure": "sps", "swath-helicity": "shs"}
#: The microphysics whose reflectivity GM1 computes on the CPU from a restart (quicklook.reflectivity.MORRISON),
#: the ones that compute their own inside the model and write it to the history frames only
#: (quicklook.reflectivity.OWN_REFLECTIVITY), and their plain names (quicklook.reflectivity.PLAIN_SCHEME); kept
#: here, without numpy, for the GUI's server, and held equal to those by a test.
CPU_REFLECTIVITY = ("morrison-mp10",)
MODEL_REFLECTIVITY = ("wsm6-mp6",)
SCHEME_WORDS = {"morrison-mp10": "Morrison", "wsm6-mp6": "WSM6", "thompson-mp8": "Thompson", "kessler": "Kessler"}
#: The receipt published beside every restart.
PUBLICATION = "restart-publication.json"


def cannot_draw(directory: Path | str) -> dict[str, str]:
    """The products one published restart cannot draw, each with the reason in a sentence, from its receipt alone
    (the envelope is not opened): the GUI greys these out instead of offering a picture that would be refused.
    What it cannot tell from the receipt it leaves on offer; the command still refuses then, with its reason."""

    import os

    path = Path(directory) / PUBLICATION
    try:
        stat = path.stat()
    except OSError:
        return {}
    # The GUI asks on every Results refresh; a receipt is read again only when it changes.
    return dict(_cannot_draw(os.fspath(path), stat.st_mtime_ns, stat.st_size))


@functools.lru_cache(maxsize=256)
def _cannot_draw(path: str, _mtime_ns: int, _size: int) -> dict[str, str]:
    import json

    try:
        with open(path, encoding="utf-8") as stream:
            receipt = json.load(stream)
    except (OSError, ValueError):
        return {}
    if not isinstance(receipt, dict):
        return {}
    config = receipt.get("config") if isinstance(receipt.get("config"), dict) else {}
    out: dict[str, str] = {}
    component = str(config.get("microphysics") or "")
    if component and component not in CPU_REFLECTIVITY:
        scheme = SCHEME_WORDS.get(component, component)
        why = (f"{scheme} writes its reflectivity to the history frames only; a restart carries none."
               if component in MODEL_REFLECTIVITY else
               f"GM1 has no CPU reflectivity formula for {scheme}, so a restart has no reflectivity.")
        for product in REFLECTIVITY_PRODUCTS:
            out[product] = why
    swaths = receipt.get("swaths") if isinstance(receipt.get("swaths"), dict) else {}
    carried = swaths.get("carried")
    if isinstance(carried, list):
        grid = config.get("grid") if isinstance(config.get("grid"), dict) else {}
        vertical = grid.get("vertical") if isinstance(grid.get("vertical"), dict) else {}
        try:
            stretched = int(vertical.get("stretch_z") or 0) != 0
        except (TypeError, ValueError):
            stretched = False
        for product, name in SWATH_NAMES.items():
            if name not in carried:
                out[product] = ("GM1 keeps no vorticity or helicity swath on stretched levels."
                                if stretched and name in ("svs", "shs") else "This run kept no such swath.")
    return out


__all__ = ["CPU_REFLECTIVITY", "ENVIRONMENT_PRODUCTS", "LEVEL_PRODUCTS", "MODEL_REFLECTIVITY", "PRODUCTS",
           "PRODUCT_WORDS", "REFLECTIVITY_PRODUCTS", "SCHEME_WORDS", "SWATH_NAMES", "cannot_draw"]
