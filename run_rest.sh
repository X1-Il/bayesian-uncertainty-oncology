#!/usr/bin/env bash
# Run the remaining study stages sequentially, cheapest first.
#
# E3/E4/E5/E6 reuse the saved ensemble (--load-checkpoints), so they are minutes
# each. E1 and E2 train a fresh ensemble per sweep point and dominate the budget,
# so they go last: if the machine is interrupted, the cheap results survive.
#
#   bash run_rest.sh
set -u

# Launched from PowerShell's Start-Process, bash inherits a PATH without the Git
# coreutils directory, so dirname/date/tail are all missing. Put them back before
# using any of them -- `cd "$(dirname "$0")"` silently no-ops otherwise, leaving
# the script running against whatever directory it happened to inherit.
export PATH="/usr/bin:/bin:$PATH"
cd "$(dirname "$0")" || { echo "cannot locate project directory"; exit 1; }
PY=.venv/Scripts/python.exe

STAGES="${STAGES:-e3 e4 e5 e6 e1 e2}"
for stage in $STAGES; do
  case "$stage" in
    e1|e2) flags="--stage $stage" ;;              # train their own models
    *)     flags="--stage $stage --load-checkpoints" ;;
  esac
  # --full is not optional here: main.py defaults to --quick, and a stage run
  # that silently used the smoke-test budget would look like a successful study.
  echo "===== $stage ($(date +%H:%M:%S)) ====="
  $PY -u main.py --full --seed 0 $flags >> run_rest.log 2>&1
  code=$?
  if [ $code -ne 0 ]; then
    echo "!! stage $stage failed (exit $code)"
    tail -20 run_rest.log
    exit $code
  fi
  echo "-- $stage done ($(date +%H:%M:%S))"
done

echo "===== all stages complete ====="
$PY summarize.py > RESULTS.md && echo "wrote RESULTS.md"
$PY paper/make_tables.py
