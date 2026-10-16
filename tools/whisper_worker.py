"""Whisper transcription worker for tools/qwen_tts_server.py's verification.

CTranslate2 (faster-whisper) and the ROCm torch build abort with
"free(): invalid pointer" when loaded in one process, so the server runs
this as a child process: one JSON request per stdin line
({"path": "<16 kHz mono float32 .npy>"}), one JSON reply per stdout line
({"text": "..."}).
"""

import json
import os
import sys

import numpy as np
from faster_whisper import WhisperModel


def main() -> None:
    model = WhisperModel(os.environ.get("QWEN_VERIFY_WHISPER", "small.en"),
                         device="cpu", compute_type="int8", cpu_threads=8)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            audio = np.load(json.loads(line)["path"]).astype(np.float32)
            segs, _ = model.transcribe(audio, beam_size=1, language="en",
                                       condition_on_previous_text=False)
            reply = {"text": " ".join(s.text for s in segs).strip()}
        except Exception as exc:   # keep serving; the server treats it as unchecked
            reply = {"error": str(exc)}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
