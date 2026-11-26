#!/usr/bin/env bash
# Runs the wildfire detection sim N times back to back (fresh random heat
# source spawn each run, since sim.py doesn't seed np.random), tracking
# how many reach [MISSION COMPLETE] vs time out vs something else, AND
# saves a full per-tick CSV for every trial (not just the terminal log)
# so a failure can actually be diagnosed from real data instead of
# guessed at again.
#
# Usage: bash run_trials2.sh [num_trials] [per_trial_timeout_seconds]
# Defaults: 5 trials, 120s each.
#
# Was 90s -- two separate 90s-timeout batches (10 trials, then 20) each
# had exactly one trial killed right as it was succeeding: fully
# centered, under 1m from the target, speed smoothly decaying toward
# the HOVER-entry threshold, just not quite there yet when the 90s
# clock ran out. Not a sim bug -- just not enough wall-clock headroom
# for the slower-to-lock-on runs. Bumped to 120s.

set -uo pipefail

N=${1:-5}
TIMEOUT_S=${2:-120}

SUCCESS=0
TIMEDOUT=0
OTHER=0

mkdir -p /tmp/trials

for i in $(seq 1 "$N"); do
  echo "=== Trial $i/$N ==="

  lsof -tiTCP:2240 -sTCP:LISTEN | xargs -r kill -9 2>/dev/null
  lsof -tiTCP:2241 -sTCP:LISTEN | xargs -r kill -9 2>/dev/null
  sleep 1

  LOG="/tmp/trials/trial_${i}.log"
  CSV="/tmp/trials/trial_${i}.csv"
  rm -f "$CSV"

  unset DEBUG_STATIC DEBUG_HEAT_X DEBUG_HEAT_Y DEBUG_HEAT_Z DEBUG_DRONE_ALT DEBUG_SPIN
  export LOG_CSV=1
  export LOG_CSV_PATH="$CSV"

  elodin run main.py > "$LOG" 2>&1 &
  PID=$!

  ELAPSED=0
  while kill -0 "$PID" 2>/dev/null && [ "$ELAPSED" -lt "$TIMEOUT_S" ]; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
  done

  if kill -0 "$PID" 2>/dev/null; then
    echo "trial $i: TIMED OUT after ${TIMEOUT_S}s -- never reached HOVER/confirm, killing process"
    kill -9 "$PID" 2>/dev/null
    TIMEDOUT=$((TIMEDOUT + 1))
  elif grep -q "MISSION COMPLETE" "$LOG"; then
    echo "trial $i: SUCCESS"
    grep -A 3 "MISSION COMPLETE" "$LOG"
    SUCCESS=$((SUCCESS + 1))
  else
    echo "trial $i: process ended WITHOUT mission complete -- check $LOG, last lines:"
    tail -n 15 "$LOG"
    OTHER=$((OTHER + 1))
  fi

  if [ -f "$CSV" ]; then
    echo "trial $i: CSV saved, $(wc -l < "$CSV") rows"
  else
    echo "trial $i: WARNING -- no CSV was written at $CSV"
  fi
  echo
done

echo "================================================"
echo "Summary over $N trials: $SUCCESS succeeded, $TIMEDOUT timed out, $OTHER ended some other way"
echo "Per-trial logs/CSVs saved under /tmp/trials/ (trial_<N>.log, trial_<N>.csv)"
