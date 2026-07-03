"""Reference contract for the builderr local-dictation challenge.

Entrants replace the body of transcribe() with their own local engine/router.
The CLI signature and the result.json shape are REQUIRED and checked by the harness:

    python -m solution.transcribe --input clip.wav --mode auto --output result.json

Rules: runs fully local; no outbound network during the scored run (loopback to a
local ASR server is fine); emit the JSON below; no hardcoded phrase fixes.

This skeleton emits a valid contract result. If `faster-whisper` is installed it
runs a real local baseline; otherwise it returns an empty transcript clearly
flagged so the contract still validates (and scores as a blank — replace it!).
"""
from __future__ import annotations
import argparse, json, time


def transcribe(wav_path: str, mode: str = "auto") -> dict:
    t0 = time.time()
    text, model_ids, candidates = "", [], []
    asr_ms = 0.0
    try:
        from faster_whisper import WhisperModel  # local, offline once weights are cached
        a = time.time()
        import os
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights")
        model = WhisperModel(model_path, device="cpu", compute_type="int8")
        segments, info = model.transcribe(wav_path, language=None, task="transcribe")
        text = " ".join(s.text for s in segments).strip()
        asr_ms = (time.time() - a) * 1000
        model_ids = ["faster-whisper-small-int8"]
        candidates = [{"engine": "faster-whisper-small", "text": text}]
    except Exception as e:
        import traceback
        traceback.print_exc()
        candidates = [{"engine": "none", "text": "", "note": f"plug your engine here ({type(e).__name__})"}]

    total_ms = (time.time() - t0) * 1000
    return {
        "text": text,
        "mode_used": mode,
        "language_guess": "unknown",
        "timings_ms": {"total": round(total_ms), "asr": round(asr_ms), "postprocess": 0},
        "raw_candidates": candidates,
        "model_ids": model_ids,
        "local_only": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="auto", choices=["auto", "fast", "hinglish", "verbatim"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = transcribe(args.input, args.mode)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.output}  ({result['timings_ms']['total']}ms, local_only={result['local_only']})")


if __name__ == "__main__":
    main()
