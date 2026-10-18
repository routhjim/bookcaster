#!/usr/bin/env bash
# Start/stop the Fish S2 Pro voice stack (and, optionally, ACE-Step music)
# as memory-capped systemd user units.
#
#   tools/fish_stack.sh up [--music]    engine (:8891) + adapter (:5014) [+ ACE :8001]
#   tools/fish_stack.sh up all          + ACE + Strata Flash-Next on the iGPU (:18100) and the
#                                       Claude Code proxy (:18080): the audiobook agent's stack
#   tools/fish_stack.sh down [--music|all]
#   tools/fish_stack.sh status
#
# Memory: Strata pins ~90 GiB of the shared RAM, so it starts LAST, after the
# GPU page pool is flushed (freed GPU pages otherwise sit in the kernel's pool
# and look used); every server start first checks there is room for its load
# peak and refuses instead of tipping the box into an OOM.
#
# Paths and GPU pins come from the environment (defaults are this project's
# reference box: HIP device 0 = the XTX for Fish, 2 = the XTX for ACE):
#   FISH_VENV  venv with fish-speech + torch (ROCm/CUDA)    ~/tts-ab/.venv-fish
#   QWEN_VENV  venv the adapter runs in (Whisper check)      ~/tts-ab/.venv-qwen
#   ACE_VENV / ACE_DIR  ACE-Step 1.5 venv and checkout       ~/tts-ab/.venv-ace, ~/tts-ab/ACE-Step-1.5
#   ACE_SERVE  ACE launcher script                           ~/tts-ab/ab2/ace_serve.py
#   FISH_BUS / ACE_BUS  PCI bus of the GPU for each          05 / 65
#   FISH_GPU / ACE_GPU  HIP_VISIBLE_DEVICES index (overrides the bus lookup)
#   FISH_WARM_REF  a voice clip to encode at start (optional)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
FISH_VENV=${FISH_VENV:-$HOME/tts-ab/.venv-fish}
QWEN_VENV=${QWEN_VENV:-$HOME/tts-ab/.venv-qwen}
ACE_VENV=${ACE_VENV:-$HOME/tts-ab/.venv-ace}
ACE_DIR=${ACE_DIR:-$HOME/tts-ab/ACE-Step-1.5}
ACE_SERVE=${ACE_SERVE:-$HOME/tts-ab/ab2/ace_serve.py}
# HIP numbers GPUs in KFD topology order, which changes with how the docks
# come up at boot (a reboot swapped the iGPU and the second XTX): find each
# server's GPU by its PCI bus instead of a fixed index.
hip_index() {
  local i=0 n p bus
  for n in $(ls -d /sys/class/kfd/kfd/topology/nodes/* | sort -t/ -k8 -n); do
    p=$n/properties
    [ "$(awk '/^simd_count/{print $2}' "$p")" = 0 ] && continue
    bus=$(printf %02x $(( $(awk '/^location_id/{print $2}' "$p") >> 8 )))
    [ "$bus" = "$1" ] && { echo $i; return 0; }
    i=$((i + 1))
  done
  echo "fish_stack: no GPU on PCI bus $1" >&2
  return 1
}
FISH_GPU=${FISH_GPU:-$(hip_index "${FISH_BUS:-05}")}
ACE_GPU=${ACE_GPU:-$(hip_index "${ACE_BUS:-65}")}
FISH_WARM_REF=${FISH_WARM_REF:-$HOME/.local/share/tts_reader/f5_voices/bob.wav}
STRATA_UP=${STRATA_UP:-$HOME/strata-eval/strata-best-up.sh}
FLUSH=${FLUSH:-/usr/local/sbin/gpu-pool-flush}
# GiB that must be free before each start (its load peak plus margin).
FISH_NEED_GB=${FISH_NEED_GB:-12}
ACE_NEED_GB=${ACE_NEED_GB:-16}
STRATA_NEED_GB=${STRATA_NEED_GB:-96}

music=0; strata=0
case "${2:-}" in
  --music) music=1 ;;
  all|--all) music=1; strata=1 ;;
esac

avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }

room_for() {   # need_gb label: flush the GPU page pool if short, refuse if still short
  local need=$1 label=$2
  if [ "$(avail_gb)" -lt "$need" ] && [ -x "$FLUSH" ]; then
    sudo -n "$FLUSH" >/dev/null 2>&1 || true
  fi
  if [ "$(avail_gb)" -lt "$need" ]; then
    echo "not starting $label: $(avail_gb) GiB free, it needs $need" >&2
    return 1
  fi
}

wait_for() {   # url label
  printf 'waiting for %s' "$2"
  for _ in $(seq 360); do
    curl -sf "$1" >/dev/null 2>&1 && { echo " ready"; return 0; }
    printf .; sleep 5
  done
  echo " TIMED OUT (journalctl --user -u ${3:-?})"; return 1
}

up() {
  if ! systemctl --user is-active -q tts-fish; then
    room_for "$FISH_NEED_GB" "the Fish engine" || return 1
    systemctl --user reset-failed tts-fish 2>/dev/null || true
    systemd-run --user --unit=tts-fish -p MemoryMax=28G -p MemorySwapMax=0 \
      -E HIP_VISIBLE_DEVICES="$FISH_GPU" -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -E FISH_WARM_REF="$FISH_WARM_REF" \
      "$FISH_VENV/bin/python" -u "$HERE/fish_engine_server.py"
  fi
  if ! systemctl --user is-active -q tts-fish-adapter; then
    systemctl --user reset-failed tts-fish-adapter 2>/dev/null || true
    systemd-run --user --unit=tts-fish-adapter -p MemoryMax=8G -p MemorySwapMax=0 \
      -E HIP_VISIBLE_DEVICES= -E QWEN_VERIFY_WORKERS="${QWEN_VERIFY_WORKERS:-4}" \
      "$QWEN_VENV/bin/python" -I -u "$HERE/fish_tts_server.py"
  fi
  if [ $music = 1 ] && ! systemctl --user is-active -q tts-ace; then
    room_for "$ACE_NEED_GB" "ACE-Step" || return 1
    systemctl --user reset-failed tts-ace 2>/dev/null || true
    systemd-run --user --unit=tts-ace -p MemoryMax=48G -p MemorySwapMax=0 \
      -p WorkingDirectory="$ACE_DIR" -E HIP_VISIBLE_DEVICES="$ACE_GPU" \
      -E ACESTEP_CONFIG_PATH=acestep-v15-xl-sft -E ACESTEP_LM_BACKEND=pt \
      -E ACESTEP_INIT_LLM=false -E ACESTEP_ROCM_DTYPE=bfloat16 \
      -E ACESTEP_OFFLOAD_TO_CPU=false -E HF_HUB_DISABLE_XET=1 \
      -E TORCH_BLAS_PREFER_HIPBLASLT=1 \
      "$ACE_VENV/bin/python" "$ACE_SERVE"
  fi
  # First start compiles the decode graphs (minutes); later starts reuse them.
  wait_for http://127.0.0.1:8891/health "Fish engine" tts-fish
  wait_for http://127.0.0.1:5014/health "Fish adapter" tts-fish-adapter
  [ $music = 1 ] && wait_for http://127.0.0.1:8001/health "ACE-Step" tts-ace
  if [ $strata = 1 ]; then
    if ! curl -sf -m5 http://127.0.0.1:18100/v1/models >/dev/null 2>&1; then
      # Last: after the voice/music load peaks are over.
      room_for "$STRATA_NEED_GB" "Strata" || return 1
      "$STRATA_UP" >/dev/null
      wait_for http://127.0.0.1:18100/v1/models "Strata Flash-Next" strata-eval
    fi
    systemctl --user start cc-streamproxy
    wait_for http://127.0.0.1:18080/health "Claude Code proxy" cc-streamproxy
  fi
  echo "free now: $(avail_gb) GiB"
  return 0
}

down() {
  local units="tts-fish-adapter tts-fish"
  [ $music = 1 ] && units="$units tts-ace"
  [ $strata = 1 ] && units="cc-streamproxy strata-eval $units"
  for u in $units; do
    systemctl --user stop "$u" 2>/dev/null || true
    systemctl --user reset-failed "$u" 2>/dev/null || true
  done
  # Freed GPU memory lands in the kernel's page pool; hand it back.
  [ -x "$FLUSH" ] && sudo -n "$FLUSH" >/dev/null 2>&1 || true
  echo "free now: $(avail_gb) GiB"
}

status() {
  for u in tts-fish tts-fish-adapter tts-ace strata-eval cc-streamproxy; do
    printf '%-18s %s\n' "$u" "$(systemctl --user is-active "$u" 2>/dev/null || true)"
  done
  curl -s http://127.0.0.1:5014/health 2>/dev/null && echo
  echo "free: $(avail_gb) GiB"
}

case "${1:-status}" in
  up) up ;;
  down) down ;;
  status) status ;;
  *) echo "usage: $0 up|down|status [--music|all]" >&2; exit 2 ;;
esac
