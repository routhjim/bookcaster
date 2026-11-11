#!/bin/bash
# card.sh BUS cmd...: run cmd on one GPU (an XTX by PCI bus, or "igpu") under its lock, taken with flock -n polling
# (never a blocking wait); refuses to start an XTX job while $ASSETS_UE_HOME/GPU_HOLD exists. Prints the wait and
# run seconds. HIP_VISIBLE_DEVICES is set to that device for cmd.
bus=$1; shift
hold=${ASSETS_UE_HOME:-$HOME/ue}/GPU_HOLD
lock=/tmp/gpu-$bus.lock

t0=$(date +%s.%N)
exec 9>>"$lock"
while :; do
  if [ "$bus" != igpu ] && [ -e "$hold" ]; then sleep 5; continue; fi
  flock -n 9 && break
  sleep 2
done
t1=$(date +%s.%N)
export HIP_VISIBLE_DEVICES=$(${ASSETS_HIP_INDEX:-$HOME/bin/hip-index} "$bus")
"$@"; rc=$?
exec 9>&-
printf '[card %s] waited %.1fs ran %.1fs rc=%d\n' "$bus" "$(echo "$t1-$t0" | bc)" "$(echo "$(date +%s.%N)-$t1" | bc)" $rc >&2
exit $rc
