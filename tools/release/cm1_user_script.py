#!/usr/bin/env python3
"""A typical CM1-user xarray analysis, run unchanged on a real CM1 cm1out_*.nc and on a GM1 cm1out file.

Uses only CM1 conventions: dims xh/yh/zh/xf/yf/zf/time, km coordinates, CM1 variable names. The release smoke
(smoke_cm1_user.sh) runs it on the netCDF cm1out a fresh install writes; the same script runs unchanged on
real CM1 r21.1 output.
"""
import sys
import numpy as np
import xarray as xr

path = sys.argv[1]
ds = xr.open_dataset(path)          # a CM1 user would use open_mfdataset over cm1out_0000*.nc
print(f"== {path}")
print("dims:", dict(ds.sizes))
print("coord units:", {c: ds[c].attrs.get("units") for c in ("xh", "yh", "zh", "xf", "yf", "zf", "time")})
t = ds.time.values
print("times (s):", t.tolist())
last = ds.isel(time=-1)
# 1. max updraft and height of max, the first thing everyone prints
w = last.w
k, j, i = np.unravel_index(int(w.argmax()), w.shape)
print(f"wmax = {float(w.max()):.2f} m/s at zf = {float(ds.zf[k]):.2f} km")
# 2. w at 5 km via nearest level (km coordinates, CM1 default outunits=1)
w5 = last.w.sel(zf=5.0, method="nearest")
print(f"w at zf~5 km: max {float(w5.max()):.2f}, min {float(w5.min()):.2f}")
# 3. destagger u, v to scalar points (xf->xh, yf->yh), ground-relative with umove/vmove
uc = 0.5 * (last.u.isel(xf=slice(1, None)).values + last.u.isel(xf=slice(0, -1)).values) + float(last.umove)
vc = 0.5 * (last.v.isel(yf=slice(1, None)).values + last.v.isel(yf=slice(0, -1)).values) + float(last.vmove)
print(f"max ground-relative wind speed at lowest level: {float(np.hypot(uc[0], vc[0]).max()):.2f} m/s")
# 4. theta perturbation from total theta minus the far-field column (CM1 users do this)
thp = last.th - last.th.isel(xh=0, yh=0)
print(f"lowest-level theta' min (cold pool): {float(thp.isel(zh=0).min()):.2f} K")
# 5. total condensate path
qcond = sum(last[q] for q in ("qc", "qr", "qi", "qs", "qg") if q in last)
print(f"max total condensate: {float(qcond.max())*1000:.3f} g/kg")
# 6. surface rain accumulation
print(f"max accumulated rain: {float(last.rain.max()):.3f} {ds.rain.attrs.get('units')}")
# 7. time series of max w (what cm1out_stats.nc' wmax gives a CM1 user)
print("wmax(t):", [round(float(ds.w.isel(time=n).max()), 2) for n in range(ds.sizes["time"])])
