#!/bin/bash
# criterion v2 scoring of E100R2-PF-4m from a clean clone of the development repository at 344e389 (CPU only, nice 19, ionice idle).
# One scorer at a time: main + forkserver + 5 workers = 7 processes, plus the 1-process catalogue test = 8.
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
C=CHECKOUT
R=SCORING
S=$C/tools/criterion_v2/scoring-sources-v2-pf4m.json
mkdir -p $R/logs
cd $C
span() { nice -n 19 ionice -c3 python3 tools/criterion_v2/window_segments.py score --spec $S --span $1 --runs $R --workers $2 > $R/logs/$1.log 2>&1; echo "$(date -u +%FT%TZ) $1 rc=$?" >> $R/driver.log; }
v1w() { nice -n 19 ionice -c3 python3 tools/criterion_v2/window_segments.py v1-window --spec $S --span $1 --runs $R > $R/logs/$1.v1.log 2>&1; echo "$(date -u +%FT%TZ) v1 $1 rc=$?" >> $R/driver.log; }
echo "$(date -u +%FT%TZ) start $(git rev-parse HEAD)" >> $R/driver.log
nice -n 19 ionice -c3 python3 tools/criterion_v2/window_segments.py check --spec $S --verify-sha 8 > $R/logs/check.log 2>&1; echo "$(date -u +%FT%TZ) check rc=$?" >> $R/driver.log
span E100R2-PF-4m-window-pre3312 5
span E100R2-PF-4m-window-3312on 5
span E100R2-PF-4m-window 5
nice -n 19 ionice -c3 python3 tools/criterion_v2/score_runs.py --manifest $S --out $R --only E100R2-PF-4m-child --jobs 1 >> $R/driver.log 2>&1
echo "$(date -u +%FT%TZ) child rc=$?" >> $R/driver.log
v1w E100R2-PF-4m-window-pre3312 &
v1w E100R2-PF-4m-window-3312on &
v1w E100R2-PF-4m-window &
wait
echo "$(date -u +%FT%TZ) done" >> $R/driver.log
