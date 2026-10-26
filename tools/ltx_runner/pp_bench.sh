#!/bin/bash
# pp_bench.sh [configs...]: the --pp lane-scheduling bench (10-21). Runs eval/ltx/pp_bench/jobs.json (6 knight ch1
# action jobs, 57-121 frames, inputs under flow3d/stage_thin_1021) per config, then checks every clip is
# byte-identical to the reference (the 10-21 overnight run) and prints the wall time. Needs both XTXs (flocks).
# Baseline 10-21 08:15: 223.9 s / 6 jobs (lanes 2, exclusive card A).
set -u
B=~/tts_reader/eval/ltx/pp_bench
REF=~/audiobooks/knight_forged/work/video/01_chapter_1/flow3d/stage_thin_1021/clips
EMB=~/audiobooks/knight_forged/work/video/01_chapter_1/flow3d/ltx/emb
IC=~/models/ltx-2.5/loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors
declare -A CFG=([base]="--lanes 2" [shared]="--lanes 2 --pp-shared-a" [l3]="--lanes 3" [l3shared]="--lanes 3 --pp-shared-a")
HIP=$(~/bin/hip-index 05),$(~/bin/hip-index 65)
for c in "${@:-base shared l3 l3shared}"; do
  for name in $c; do
    o=$B/out_$name; rm -rf "$o"; mkdir -p "$o"
    HIP_VISIBLE_DEVICES=$HIP flock /tmp/gpu-05.lock flock /tmp/gpu-65.lock timeout 1800 ~/tts-ab/.venv-ltx/bin/python \
      ~/tts_reader/tools/ltx_runner/ltx_split_next.py gen $B/jobs.json --ic-lora "$IC" --fa-amd --blaslt --out "$o" \
      --budget 1700 --pp --split 6 ${CFG[$name]} --emb "$EMB" > "$o.log" 2>&1
    rc=$?; same=0; n=0
    for f in "$o"/*.mp4; do n=$((n+1)); cmp -s "$f" "$REF/$(basename "$f")" && same=$((same+1)); done
    echo "$name rc=$rc $(grep -oE 'done [0-9]+ jobs in [0-9.]+s' "$o.log") identical $same/$n $(grep -oE '\[B\] busy [0-9.]+s of [0-9.]+s wall \([0-9]+%\)' "$o.log")"
  done
done
