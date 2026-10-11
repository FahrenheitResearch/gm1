# CM1 r21.1 reference output at 24 x 24 x 40

Real CM1 r21.1 output for the `gpuwm-cm1 cm1out` conformance tests
(`tests/test_cm1out.py`, `tests/test_cm1out_gpu.py`).

## The CM1 build

A CM1 r21.1 tree whose 91 `src/*.F` files match the release's published
content digests (`tools/authority/cm1r21.1-content-sha256.tsv`, checked
2026-09-23), built with gfortran 15.2.0, `-O2 -fopenmp`, `OUTPUTOPT =
-DNETCDF -DNCFPLUS` (netCDF-4 classic model, deflate level 1 with shuffle)
and run with `OMP_NUM_THREADS=4`.

## The two runs

`namelist.input` is CM1's distributed `run/config_files/supercell/namelist.input`
with only these changes:

```
nx = 24, ny = 24, timax = 120.0, tapfrq = 60.0, output_format = 2, output_filetype = 2
```

It wrote `cm1out_000001.nc` (t = 0), `cm1out_000002.nc` (60 s),
`cm1out_000003.nc` (120 s) and `cm1out_stats.nc` (statfrq = 60 s).

`stats-run.namelist.input` is the same with `timax = 2400.0`,
`tapfrq = statfrq = 300.0` and `output_basestate`, `output_thpert`,
`output_prspert`, `output_pipert` and `output_rho` set to 1, so that each
history file holds every field `statpack.F` reads.

## The files kept here

| File | What | Source |
|---|---|---|
| `namelist.input` | the conformance namelist | as above |
| `cm1out_000001.nc` | CM1's own t = 0 history file, unmodified (md5 477555a88846210fa40d740c28dcd9e4) | first run |
| `cm1out_000002.header.json` | the layout of CM1's `cm1out_000002.nc` (md5 03c9f1d17171f079eb81f92807723fa7): dimensions, variables and their attributes, the 71 global attributes with types and values, every 1-D variable as float32 words | `tools/cm1out_reference.py header` |
| `cm1out_stats.header.json` | the same for CM1's `cm1out_stats.nc` (md5 0db607c9f1bda0bcea3b9359fdbeb5bd), including every statistic at 0, 60 and 120 s as float32 words | `tools/cm1out_reference.py header` |
| `stats-run.namelist.input` | the statistics namelist | as above |
| `stats-run-t1500.npz` | the fields `statpack.F` reads at 1500 s, from the second run's `cm1out_000006.nc` (md5 d19be5a0c1b59805def4a63dc6e23699) | `tools/cm1out_reference.py stats-state` |
| `variants.json` | fourteen CM1 runs of `namelist.input` with the edits it records (`timax = tapfrq = statfrq = 60`, then: `basic` with thpert, pipert and rho on; `output_filetype` 1 and 3; `outunits = 2` with `iorigin = 1`; `imove = 0`; `icor = 1`; `dx = dy = 333.33`; `sgsmodel = 2`; every budget with Morrison (`budget`), with WSM6 and Coriolis (`budgetw`, where CM1 forces `eqtset = 1`) and with even-order advection and `idiff = 1, difforder = 2` (`budgetd`, where CM1 turns idiff off); the base state and derived fields; two `stretch_z = 1` grids, `stretchg` being one GM1 runs): CM1's history variables in order with dims, long_name and units, plus the words the tests need (the dx coordinates, the stretched vertical coordinates and Fornberg attributes, a 40 x 4 x 4 block of `pi`, `pipert`, `prs` and the `pi0` column) | `tools/cm1out_reference.py variants` |
| `stats-run-t1500.cm1-stats.json` | CM1's own `cm1out_stats.nc` record at 1500 s (md5 of the file 05d26670a3c9ce658764be62e0533df6), as float32 words | `tools/cm1out_reference.py stats-state` |

A first description of this reference said its `xh` runs from -23 to 23 km.
CM1's own file does not: at nx = 24, dx = 1 km
and `iorigin = 2` it runs from -11.5 to 11.5 km (`xf` from -12 to 12), and
the tests hold the converter to CM1's file.

## GrADS and parcels

`grads/` holds CM1's own GrADS descriptors and records and its parcel
files for variants of the same namelist (`grads/README.md`).
