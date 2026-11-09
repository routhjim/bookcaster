#!/usr/bin/env bash
# The one test command for merges: runs tools/tests/test_*.py in parallel, each with its interpreter.
#
#   tools/tests/run_suite.sh            # FAST (default for merges): leaves out tests marked slow
#   tools/tests/run_suite.sh full       # FULL: every test but the manual harnesses
#   tools/tests/run_suite.sh fast test_shots_1023 test_rooms_1029   # only the named tests (mode still applies)
#   SUITE_JOBS=8 tools/tests/run_suite.sh   # parallel jobs (default: CPU count)
#
# A test is marked by a one-line comment in its first 10 lines; words may be combined ("# suite: slow pose"):
#   # suite: slow     - takes over ~20 s alone; FULL only
#   # suite: manual   - a harness that needs arguments, a GPU or a person; never run here
#   # suite: pose     - needs ~/tts-ab/.venv-pose/bin/python -I (default: ~/tts_reader/.venv/bin/python)
# Exit 0 when every test run passed, 1 otherwise. CPU only: systemctl, systemd-run and sudo are shimmed on PATH (a
# test can never stop, start or restart a unit such as strata-eval, nor launch a scoped GPU job); is-active answers
# "inactive". Every shimmed call is listed at the end as a side effect the test should stub itself.
set -u
MODE=${1:-fast}
case "$MODE" in fast|full) shift || true ;; -h|--help) sed -n '2,15p' "$0"; exit 0 ;; *) MODE=fast ;; esac
[[ "$MODE" == fast || "$MODE" == full ]] || { echo "usage: $0 [fast|full] [test names...]"; exit 2; }

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
export SUITE_PY_MAIN=${SUITE_PY_MAIN:-$HOME/tts_reader/.venv/bin/python}
export SUITE_PY_POSE=${SUITE_PY_POSE:-$HOME/tts-ab/.venv-pose/bin/python}
JOBS=${SUITE_JOBS:-$(nproc)}
OUT=$(mktemp -d "${TMPDIR:-/tmp}/run_suite.XXXXXX")   # logs only (small)
trap 'rm -rf "$OUT"' EXIT

BIN="$OUT/bin"; mkdir -p "$BIN"
for t in systemctl systemd-run sudo; do
    cat >"$BIN/$t" <<SHIM
#!/usr/bin/env bash
echo "\${SUITE_TEST:-?}: $t \$*" >>"$OUT/sidefx.log"
[[ "$t" == systemctl ]] && { [[ " \$* " == *" is-active "* ]] && exit 3; exit 0; }
exit 1
SHIM
    chmod +x "$BIN/$t"
done
export PATH="$BIN:$PATH"

marks() { head -n 10 "$1" | sed -n 's/^# suite:[[:space:]]*//p' | head -n 1; }

declare -a RUN=() SKIP_SLOW=() SKIP_MANUAL=()
if (($#)); then
    FILES=(); for n in "$@"; do n=${n%.py}; FILES+=("$HERE/${n##*/}.py"); done
else
    FILES=("$HERE"/test_*.py)
fi
for f in "${FILES[@]}"; do
    [[ -f "$f" ]] || { echo "no such test: $f"; exit 2; }
    m=" $(marks "$f") "
    name=$(basename "$f" .py)
    if [[ "$m" == *" manual "* ]]; then SKIP_MANUAL+=("$name")
    elif [[ "$m" == *" slow "* && "$MODE" == fast ]]; then SKIP_SLOW+=("$name")
    else
        py="$SUITE_PY_MAIN"; [[ "$m" == *" pose "* ]] && py="$SUITE_PY_POSE -I"
        RUN+=("$name|$py")
    fi
done

one() {   # name|interpreter -> $OUT/name.{log,rc,sec}
    local name=${1%%|*} py=${1#*|} t0 t1 rc
    t0=$(date +%s.%N)
    export SUITE_TEST=$name
    # shellcheck disable=SC2086
    (cd "$SUITE_ROOT" && timeout 600 $py "tools/tests/$name.py" >"$SUITE_OUT/$name.log" 2>&1 </dev/null); rc=$?
    t1=$(date +%s.%N)
    echo "$rc" >"$SUITE_OUT/$name.rc"
    awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f\n", b-a}' >"$SUITE_OUT/$name.sec"
    [[ $rc == 0 ]] && echo "  ok   $name" || echo "  FAIL $name (exit $rc)"
}
export -f one
export SUITE_ROOT="$ROOT" SUITE_OUT="$OUT"

echo "run_suite: $MODE mode, ${#RUN[@]} tests, $JOBS jobs" \
     "(left out: ${#SKIP_SLOW[@]} slow, ${#SKIP_MANUAL[@]} manual)"
T0=$(date +%s.%N)
printf '%s\n' "${RUN[@]}" | xargs -P "$JOBS" -I{} bash -c 'one "$@"' _ {}
T1=$(date +%s.%N)

FAILED=()
for r in "${RUN[@]}"; do
    name=${r%%|*}
    [[ "$(cat "$OUT/$name.rc" 2>/dev/null)" == 0 ]] || FAILED+=("$name")
done
for name in "${FAILED[@]}"; do
    echo; echo "===== FAIL $name (last 30 lines) ====="
    tail -n 30 "$OUT/$name.log"
done
echo
echo "slowest:"
for r in "${RUN[@]}"; do n=${r%%|*}; printf '%6s s  %s\n' "$(cat "$OUT/$n.sec" 2>/dev/null)" "$n"; done | sort -rn | head -n 8
((${#SKIP_SLOW[@]})) && echo "left out (slow; run 'full'): ${SKIP_SLOW[*]}"
((${#SKIP_MANUAL[@]})) && echo "left out (manual): ${SKIP_MANUAL[*]}"
if [[ -s "$OUT/sidefx.log" ]]; then
    echo "side effects blocked by the shims (stub these in the test):"; sort "$OUT/sidefx.log" | uniq -c | sed 's/^/  /'
fi
[[ -n "${SUITE_TIMES:-}" ]] && for r in "${RUN[@]}"; do n=${r%%|*}; echo "$(cat "$OUT/$n.sec") $n"; done >"$SUITE_TIMES"
TOTAL=$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.1f", b-a}')
if ((${#FAILED[@]})); then
    echo "run_suite $MODE: ${#FAILED[@]} of ${#RUN[@]} FAILED (${FAILED[*]}) in $TOTAL s"; exit 1
fi
echo "run_suite $MODE: all ${#RUN[@]} passed in $TOTAL s"
