# CM1 namelist fixtures for the importer tests

Byte copies of CM1's own files, used by `tests/test_cm1import_*.py`. `tests/`
is not in the sdist or the wheel. The public source snapshot carries the three
CM1 r21.1 files below, which `NOTICE` lists under the CM1 r21.1 release grant
it quotes, and leaves out the r22.0 and r18 files, whose terms `NOTICE` does
not quote; the tests that read those two skip there.

| File | Source | SHA-256 |
|---|---|---|
| `r21.1-supercell.namelist.input` | cm1r21.1 `run/config_files/supercell/namelist.input` | `3854f731efe6a6a9d56d6aa3db198434ccaba85bc5584b5e8ce1c5edfd3b56a4` |
| `r21.1-squall_line.namelist.input` | cm1r21.1 `run/config_files/squall_line/namelist.input` | `2f75792a569cf497d76d9df2186692f81a296028e85c162c65cd19df8dfdf5b8` |
| `r21.1-run.namelist.input` | cm1r21.1 `run/namelist.input` | `3cf17aa1ac513544d4695994013fd30ea89c2bb0f7d926e41e348635882a6ca5` |
| `r22.0-supercell.namelist.input` | cm1r22.0 `run/config_files/supercell/namelist.input` (development repository only) | `d8e64ca3de41ed855c7fc486e1a87adb1bda4d15ea260bd2705e631f735011bb` |
| `r18-run.namelist.input` | cm1r18 `run/namelist.input` (development repository only) | `47b9c7dd27c77f452a3eb606066289302d7a644b119f13ea8e21f7aaa9f14ef3` |

The three r21.1 digests are the ones `tools/authority/cm1r21.1-content-sha256.tsv`
records for the pinned tarball; the tests check them against it.

`r21.1-run.harness-translation.toml` is the reference translation of the
top-level `run/namelist.input`. It was produced by the importer's translator
before it moved into `cm1import`, with the same case settings as the supercell
and squall_line references (`physics.component = "morrison-mp10"`,
`run.lane = "cartesian"`, `runtime.warmup_steps = 0`,
`runtime.memory_budget_mib = 8192`, `runtime.capacity_profile = "default"`).
