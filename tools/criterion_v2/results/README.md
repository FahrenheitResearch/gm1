# Recorded criterion v2 results

These are the files the scoring wrote, kept as the record behind `docs/GM1-TORNADO-VERDICTS-V2.md`.

| Folder | What it holds |
|---|---|
| `synthetic-v1-diagnosis/` | The analytic-vortex study that showed why the first rule (v1) depended on the grid. |
| `synthetic-v2-prefreeze/` | The analytic vortices and negative controls v2 was frozen with: 105 case and grid rows, each with its expected and its scored outcome (`sweep.tsv`). |
| `scoring-v2/` | The pre-registered scoring of the development runs (E100V, E100W, E100N and the rest). |
| `scoring-v2-pf4m/` | The scoring of E100R2-PF-4m, the 4 m run with public data, and its independent verification (`verify/`). |

In the two scoring folders, each scored source has a folder named by its label with three files:

* `samples.tsv`: one row per sample (a frame or a window record) with the strongest candidate's centre,
  radius of maximum wind, winds, the conditions it met and its class (Q qualifies, U cannot be judged on
  this grid, N does not qualify).
* `candidates.tsv`: every candidate vortex of every sample.
* `verdict.json`: the outcome, caption, episodes and counts, the hashes of the rule and scorer files, and
  how the source was read.

The three E100R2-PF-4m window sources keep the two tables gzipped (`gzip -n`).
`tornado-verdicts-v2.json` and `tornado-verdicts-v2-pf4m.json` collect the sources per run, and
`verdict-table.md` is the same as a table. Beside them are each source's command record (`*.run.json`),
the reasons a source was not scored (`*.not-scored.json`) and the logs.

`python3 tools/criterion_v2/check_recorded.py` runs the frozen rule on the tables again and compares the
result with every `verdict.json`.

## What was changed for publication

Only names of machines and folders. Paths on the machines that held the runs were replaced by
placeholders, and the names of those machines were taken out of free-text notes. No number, time, hash,
class or verdict was changed, and the tables (`samples.tsv`, `candidates.tsv`, plain or gzipped) are
byte for byte what the scorer wrote.

| Placeholder | Stands for |
|---|---|
| `RUNS`, `RUNS-2M` | the folders that held the development runs' output |
| `RUN` | the E100R2-PF-4m run folder; `RUN/S3` is its 4 m nest stage |
| `SCORING` | the folder the scoring wrote into (these results are its copy) |
| `CHECKOUT` | the clean checkout of the repository the scorer ran from |

The independent verification's folder was named after its place in the work and is now `verify/`.

## The word "clean" in the E100R2-PF-4m files

The source `E100R2-PF-4m-window` is described in these files as "the whole clean window", and the notes speak
of a "clean limit" at 3452.4375 s. Clean there means only that a record could be read. It does not mean the
nest's time step limit held: that was passed at 3312 s. The sources `E100R2-PF-4m-window-3312on` and
`E100R2-PF-4m-window` therefore hold winds (up to 163.3 m/s) that must not be quoted, and the row for the
whole window in `scoring-v2-pf4m/verdict-table.md` carries no note saying so. Only
`E100R2-PF-4m-window-pre3312` backs the published numbers. The files are kept as they were written.

## What is not here

* The model output the scorer read. Only E100R2-PF-4m has public output, as netCDF on Hugging Face; see
  "How to check this yourself, and what cannot be checked yet" in the verdicts page.
* The run's status notes and the short control rerun that a few entries name
  (`reference_stream`, "status notes"). They are named so the record stays complete, not so they can be
  followed.
* Commit hashes name commits of the private development repository.
