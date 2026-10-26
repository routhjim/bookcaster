#!/bin/bash
# The whole 3D flow of one chapter, unattended (user 10-21: "run it overnight"): bookcaster plan -> assets ->
# approve --auto -> stage -> approve --auto -> video --flow 3d --auto-approve, each as the book's own job, one after
# the other; stops at the first step whose job does not end "done". Only bookcaster commands (the bookcaster-only
# rule). The automated QA gates stand in for the review: blocking findings are NOT auto-approved.
#
#   systemd-run --user --unit=<book>-chain tools/flow3d_chain.sh <book> <chapter> [stop-after-step]
#   (stop-after: plan | assets | stage - the chain ends there, e.g. while staging fixes are still being made)
#   [start-at]: plan | assets | stage | video - the earlier steps are skipped (their results are on disk)
#   log: ~/audiobooks/<book>/work/chain.log
set -u
BOOK=$1; CH=$2; STOP_AFTER=${3:-}; START_AT=${4:-plan}
BC=${BOOKCASTER:-$HOME/.local/bin/bookcaster}
DIR=$HOME/audiobooks/$BOOK
UNIT=bookcaster-$(echo "$BOOK" | sed 's/[^A-Za-z0-9_-]/-/g' | cut -c1-40)
LOG=$DIR/work/chain.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

wait_job() {         # until the book's job unit has ended, then its job.json state
    sleep 20
    while systemctl --user is-active -q "$UNIT"; do sleep 30; done
    python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('state','?'))" "$DIR/work/job.json" 2>/dev/null || echo "?"
}

ORDER="plan assets stage video"
started=0
step() {             # step <name> <bookcaster args...>
    local name=$1; shift
    [ "$name" = "$START_AT" ] && started=1
    [ $started = 1 ] || { say "skip $name (start at $START_AT)"; return 0; }
    say "START $name: bookcaster $*"
    "$BC" "$@" >> "$LOG" 2>&1 || { say "FAILED to start $name"; exit 1; }
    local st; st=$(wait_job)
    say "END $name: $st"
    [ "$st" = "done" ] || { say "STOP: $name ended '$st' (see $DIR/work/job.log)"; exit 1; }
    [ "$name" = "$STOP_AFTER" ] && { say "stopping after $name (as asked)"; exit 0; }
    return 0
}

say "chain for $BOOK chapter $CH"
step plan   plan "$BOOK" --chapter "$CH"
step assets assets "$BOOK" --chapter "$CH"
say "approve --auto (assets)"; "$BC" approve "$BOOK" --auto >> "$LOG" 2>&1
step stage  stage "$BOOK" --chapter "$CH"
say "approve --auto (staged props)"; "$BC" approve "$BOOK" --auto >> "$LOG" 2>&1
step video  video "$BOOK" --flow 3d --chapter "$CH" --auto-approve
say "chain finished: $(ls -t "$DIR"/out/*.mp4 2>/dev/null | head -1)"
