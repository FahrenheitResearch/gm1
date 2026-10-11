"""Reflectivity for the quick-looks: the scheme's own operator or CM1's calc_refl10cm, nothing else.

In this order (picture rules):

1. the frame's own ``dbz`` carrier, when the microphysics computes reflectivity inside the model;
2. for Morrison (ptype = 5), CM1 r21.1 morrison.F ``calc_refl10cm`` on the CPU, from the frame's rain, snow and
   graupel/hail mass and number, density and temperature.  The port lives with the cm1out writer
   (``cm1core.cm1out.reflectivity``); until that module is in the build, a development checkout's
   copy of the same port is used.  The operator runs only on the columns that
   hold rain, snow or graupel above CM1's own 1e-12 floor; every other column is below 5 dBZ and is not drawn.
3. otherwise no reflectivity picture, and the reason says why.

Nothing here smooths: the 1 km field is interpolated linearly in Z (mm^6 m^-3) between the two scalar levels that
bracket 1 km, and the pictures sample it bilinearly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .frame import QuickFrame

MORRISON = ("morrison-mp10",)
#: The schemes GM1 writes their own reflectivity for (``gm1_runtime.REFLECTIVITY_COMPONENTS``, not imported here:
#: the worker stays clear of the runtime's imports; a test holds the two equal).
OWN_REFLECTIVITY = ("wsm6-mp6",)
R1 = np.float32(1.0e-12)
FLOOR_DBZ = np.float32(-35.0)
CHUNK_COLUMNS = 12000
P00, RD, CP = 1.0e5, 287.04, 1005.7
MASS = ("qr", "qs", "qg")
NUMBER = ("ncr", "ncs", "ncg")
PLAIN_SCHEME = {"morrison-mp10": "Morrison", "wsm6-mp6": "WSM6", "thompson-mp8": "Thompson",
                "kessler": "Kessler"}


@dataclass
class Reflectivity:
    dbz: np.ndarray | None          # (nz, ny, nx) dBZ, or None
    operator: str                   # plain words: where the numbers come from
    reason: str | None = None       # why there is none

    @property
    def available(self) -> bool:
        return self.dbz is not None


@lru_cache(maxsize=1)
def _calc_function() -> tuple[Callable[..., Any] | None, str]:
    """calc_refl10cm from the cm1out writer or, in a source checkout, from tools/morrison_dbz.py."""

    try:
        from ..cm1out import reflectivity as module  # type: ignore[attr-defined]
    except ImportError:
        module = None
    if module is not None and callable(getattr(module, "calc_refl10cm", None)):
        return module.calc_refl10cm, "cm1core.cm1out.reflectivity"
    tool = Path(__file__).resolve().parents[3] / "tools" / "morrison_dbz.py"
    if tool.is_file():
        spec = importlib.util.spec_from_file_location("_gm1_morrison_dbz_port", tool)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.calc_refl10cm, "tools/morrison_dbz.py"
    return None, ""


def _temperature(qf: QuickFrame, sj: slice, si: slice) -> np.ndarray:
    """T = theta * pi (CM1 mp_driver.F:108) over a crop, from whatever the frame carries."""

    f32 = np.float32
    if qf.has("th"):
        theta = np.array(qf.view("th")[:, sj, si], dtype=f32)
    else:
        th0 = qf.base_column("th0")
        if th0 is None or not qf.has("th3d"):
            raise ValueError("temperature needs th, or th3d with the base-state th0")
        theta = np.array(qf.view("th3d")[:, sj, si], dtype=f32) + th0.astype(f32)[:, None, None]
    if qf.has("pi"):
        pi = np.array(qf.view("pi")[:, sj, si], dtype=f32)
    elif qf.has("pp3d") and qf.base_column("pi0") is not None:
        pi = np.array(qf.view("pp3d")[:, sj, si], dtype=f32) + qf.base_column("pi0").astype(f32)[:, None, None]
    elif qf.has("prs"):
        pi = ((np.array(qf.view("prs")[:, sj, si], dtype=np.float64) / P00) ** (RD / CP)).astype(f32)
    else:
        raise ValueError("the Exner function needs pi, pp3d with pi0, or prs")
    return (theta * pi).astype(f32)


def morrison_volume(qf: QuickFrame, calc: Callable[..., Any], crop: tuple[slice, slice] | None = None) -> np.ndarray:
    """calc_refl10cm on the columns holding rain, snow or graupel/hail; the floor elsewhere."""

    sj, si = crop if crop is not None else (slice(None), slice(None))
    ihail = (qf.config.get("dense_ice") or {}).get("ihail")
    if ihail not in (0, 1):
        raise ValueError("the frame does not record ihail (config.dense_ice.ihail), which sets the hail density")
    names = qf.species_names("q3d")
    numbers = qf.species_names("nc3d")
    q = qf.view("q3d")
    n = qf.view("nc3d")
    mass = [np.array(q[names.index(s), :, sj, si], dtype=np.float32) for s in MASS]
    nz = mass[0].shape[0]
    have = np.zeros(mass[0].shape[1:], bool)
    for m in mass:
        have |= (m > R1).any(axis=0)
    out = np.full(mass[0].shape, FLOOR_DBZ, np.float32)
    cols = np.flatnonzero(have.ravel())
    if cols.size == 0:
        return out
    num = [np.array(n[numbers.index(s), :, sj, si], dtype=np.float32) for s in NUMBER]
    rho = np.array(qf.view("rho")[:, sj, si], dtype=np.float32)
    t = _temperature(qf, sj, si)
    flat = [a.reshape(nz, -1) for a in (*mass, t, *num, rho)]
    of = out.reshape(nz, -1)
    for c0 in range(0, cols.size, CHUNK_COLUMNS):
        cc = cols[c0:c0 + CHUNK_COLUMNS]
        a = [f[:, cc] for f in flat]
        of[:, cc] = calc(None, a[0], a[1], a[2], a[3], None, a[4], a[5], a[6], a[7], ihail=int(ihail))
    return out


def volume(qf: QuickFrame) -> Reflectivity:
    component = str(qf.config.get("microphysics") or "")
    scheme = PLAIN_SCHEME.get(component, component or "this microphysics")
    if qf.has("dbz") and tuple(qf.frame.carriers["dbz"].shape) == (qf.nz, qf.ny, qf.nx):
        return Reflectivity(qf.get("dbz"), f"{scheme}'s own reflectivity, computed in the model")
    if component in MORRISON:
        calc, where = _calc_function()
        if calc is not None:
            return Reflectivity(morrison_volume(qf, calc), "CM1's calc_refl10cm (Morrison, 10 cm) on the CPU")
        return Reflectivity(None, "", "This build has no CPU reflectivity operator for Morrison yet; it comes with "
                                      "the cm1out writer's reflectivity step.")
    if component in OWN_REFLECTIVITY:
        if qf.t <= 0.0:
            return Reflectivity(None, "", f"The starting frame holds no reflectivity: {scheme} writes its own from "
                                          "its first step on, when the namelist sets output_dbz = 1.")
        return Reflectivity(None, "", f"This run did not ask {scheme} for its reflectivity (output_dbz = 0), so no "
                                      "reflectivity picture is drawn.")
    return Reflectivity(None, "", f"{scheme} has no reflectivity formula in GM1, and CM1's is not ported for it, so "
                                  "no reflectivity picture is drawn.")


def possible(qf: QuickFrame) -> bool:
    """Whether :func:`volume` can give reflectivity for this frame, without computing it."""

    if qf.has("dbz") and tuple(qf.frame.carriers["dbz"].shape) == (qf.nz, qf.ny, qf.nx):
        return True
    return str(qf.config.get("microphysics") or "") in MORRISON and _calc_function()[0] is not None


def at_height(dbz3: np.ndarray, zh: np.ndarray, z_m: float) -> tuple[np.ndarray, tuple[float, float]]:
    """dBZ at ``z_m`` above ground, linear in Z between the bracketing scalar levels, and those two heights."""

    k = int(np.searchsorted(zh, z_m)) - 1
    k = max(0, min(k, len(zh) - 2))
    w = float(np.clip((z_m - zh[k]) / (zh[k + 1] - zh[k]), 0.0, 1.0))
    za = np.power(10.0, dbz3[k] / 10.0, dtype=np.float64)
    zb = np.power(10.0, dbz3[k + 1] / 10.0, dtype=np.float64)
    return (10.0 * np.log10((1 - w) * za + w * zb)).astype(np.float32), (float(zh[k]), float(zh[k + 1]))


__all__ = ["Reflectivity", "at_height", "morrison_volume", "possible", "volume"]
