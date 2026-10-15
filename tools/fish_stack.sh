#!/usr/bin/env bash
# Start/stop the Fish S2 Pro voice stack (and, optionally, ACE-Step music)
# as memory-capped systemd user units.
#
#   tools/fish_stack.sh up [--music]    engine (:8891) + adapter (:5014) [+ ACE :8001]
#   tools/fish_stack.sh down [--music]
#   tools/fish_stack.sh status
#
# Paths and GPU pins come from the environment (defaults are this project's
# reference box: HIP device 0 = the XTX for Fish, 2 = the XTX for ACE):
#   FISH_VENV  venv with fish-speech + torch (ROCm/CUDA)    ~/tts-ab/.venv-fish
#   QWEN_VENV  venv the adapter runs in (Whisper check)      ~/tts-ab/.venv-qwen
#   ACE_VENV / ACE_DIR  ACE-Step 1.5 venv and checkout       ~/tts-ab/.venv-ace, ~/tts-ab/ACE-Step-1.5
#   ACE_SERVE  ACE launcher script                           ~/tts-ab/ab2/ace_serve.py
#   FISH_GPU / ACE_GPU  HIP_VISIBLE_DEVICES for each         0 / 2
#   FISH_WARM_REF  a voice clip to encode at start (optional)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
FISH_VENV=${FISH_VENV:-$HOME/tts-ab/.venv-fish}
QWEN_VENV=${QWEN_VENV:-$HOME/tts-ab/.venv-qwen}
ACE_VENV=${ACE_VENV:-$HOME/tts-ab/.venv-ace}
ACE_DIR=${ACE_DIR:-$HOME/tts-ab/ACE-Step-1.5}
ACE_SERVE=${ACE_SERVE:-$HOME/tts-ab/ab2/ace_serve.py}
FISH_GPU=${FISH_GPU:-0}
ACE_GPU=${ACE_GPU:-2}
FISH_WARM_REF=${FISH_WARM_REF:-$HOME/.local/share/tts_reader/f5_voices/bob.wav}

music=0
[ "${2:-}" = "--music" ] && music=1

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
  return 0
}

down() {
  local units="tts-fish-adapter tts-fish"
  [ $music = 1 ] && units="$units tts-ace"
  for u in $units; do
    systemctl --user stop "$u" 2>/dev/null || true
    systemctl --user reset-failed "$u" 2>/dev/null || true
  done
}

status() {
  for u in tts-fish tts-fish-adapter tts-ace; do
    printf '%-18s %s\n' "$u" "$(systemctl --user is-active "$u" 2>/dev/null || true)"
  done
  curl -s http://127.0.0.1:5014/health 2>/dev/null && echo
}

case "${1:-status}" in
  up) up ;;
  down) down ;;
  status) status ;;
  *) echo "usage: $0 up|down|status [--music]" >&2; exit 2 ;;
esac
