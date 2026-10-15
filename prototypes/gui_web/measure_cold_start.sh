#!/usr/bin/env bash
# Cold start to the first native window: launch time (epoch, in FORSTER_GUI_T0) to the
# moment the window's page has connected to the server (it prints "ready N s after
# launch" and quits). Usage: measure_cold_start.sh N COMMAND...  (run from the repo root)
#   prototypes/gui_web/measure_cold_start.sh 5 prototypes/gui_web/dist/FORSTER-web.app/Contents/MacOS/FORSTER-web
#   prototypes/gui_web/measure_cold_start.sh 5 uv run --with-requirements prototypes/gui_web/requirements.txt python prototypes/gui_web/main.py
set -euo pipefail
n=$1
shift
for _ in $(seq "$n"); do
  t0=$(perl -MTime::HiRes=time -e 'printf "%.4f", time')
  FORSTER_GUI_T0=$t0 perl -e 'alarm shift; exec @ARGV' 60 "$@" --exit-when-ready 2>&1 | grep -o "ready [0-9.]* s" || echo "no window within 60 s"
done
