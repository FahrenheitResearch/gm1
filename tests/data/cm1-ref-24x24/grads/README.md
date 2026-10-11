# CM1 r21.1 GrADS and parcel reference output

Real CM1 r21.1 output for the `gpuwm-cm1 cm1out` GrADS and parcel tests
(`tests/test_cm1out_grads.py`, `tests/test_cm1out_pdata.py`), made on
2026-09-23 with the same build as `../README.md` (gfortran 15.2.0,
`-O2 -fopenmp`, `-DNETCDF -DNCFPLUS`), `OMP_NUM_THREADS=4`.  Each run took
0.3-3.9 s.

## The runs

Each is `../namelist.input` with the edits in `variants.json`
(`make/namelists.py` writes them):

| Run | Edits | What it holds |
|---|---|---|
| `g1` | `output_format = 1`, `output_filetype = 1`, `timax = 120` | every time in `cm1out_s.dat` ...; 3 times |
| `g2` | `output_format = 1`, `output_filetype = 2`, `timax = 60` | `cm1out_00000N_s.dat` ..., `options template` |
| `g3` | as `g2` with `output_filetype = 3` | CM1's per-process names (`cm1out_000000_00000N_s.dat`) and its joined-file `dset` |
| `gn` | as `g2` with `imove = 0` | the metadata without the domain's motion |
| `gs` | as `g2` with `../variants.json`'s `stretchg` edits (`stretch_z = 1`) | `zdef ... levels` with the stretched heights |
| `gu` | as `g2` with `outunits = 2`, `iorigin = 1` | coordinates in metres from the corner |
| `pg` | `nx = ny = 120`, `iprcl = 1`, `nparcels = 36000`, `output_format = 1`, `timax = 60` | `cm1out_pdata.dat` and `cm1out_parcel.ctl` |
| `pn` | as `pg` with `output_format = 2` | `cm1out_pdata.nc` |

## The files kept here

A collection script copied each run's descriptors (`*.ctl`), its
`cm1out_metadata.dat` and `cm1out_stats.dat`, and wrote:

* `reference.json`: the sha256 and size of every `.dat` CM1 wrote; the sha256
  of every t = 0 history record of `g1` (one per variable, from
  `cm1out_s.dat` ... `cm1out_w.dat`); a check that all 46 of those records
  are word-identical to the netCDF run's `../cm1out_000001.nc` (true for all
  46); and a check that `cm1out_pdata.dat` is `cm1out_pdata.nc` transposed
  (`(time, npvals, nparcels)` records against `(time, xh)` variables: true);
* `pg/cm1out_pdata.header.json`: the layout of CM1's `cm1out_pdata.nc`
  (dimensions, variables with their attributes, global attributes, the
  first words of each coordinate).

The `.dat` history files themselves (up to 45 MB) were not kept.
