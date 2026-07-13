"""The ONE function you implement for the STREAMING dictation track.

You do NOT build a server. The sealed harness (solution/stream_server.py) handles
the WebSocket, the real-time audio feed, and emitting events. You write `draft()`.

    draft(audio_buffer, is_final) -> (text_so_far, stable_chars)

The harness calls draft() repeatedly as audio arrives (is_final=False) and once
after the user stops (is_final=True). audio_buffer is ALL audio so far: raw PCM
s16le, mono, 16kHz (little-endian int16). Return:

  - text_so_far : your best transcript of the audio heard so far. Keep the
                  Hindi-English code-switch faithful — write what was actually
                  said, don't translate the mix into English (the scorecard caps
                  that). On is_final=True, return your best full transcript.
  - stable_chars: length of the leading prefix of text_so_far you COMMIT to —
                  you promise never to rewrite it. Must be non-decreasing across
                  calls. Rewriting committed text counts as revision churn.

Tips that match how the reference engine (RambleFix) does it:
  - Re-decode the rolling prefix; commit the longest common prefix with your
    previous draft (that part has stopped changing — safe to lock).
  - Don't translate to chase a meaning score; it kills faithfulness and is capped.
  - Be fast on the first useful partial (TTFS is scored) and on the final
    (end-to-final is the main latency axis).
  - Never return a blank, a loop, or hang — degrade to your best partial instead.

This reference body wraps a local faster-whisper draft on the rolling buffer and
commits the stable common prefix. If faster-whisper isn't installed it returns an
empty draft (clearly a non-winning placeholder) so the contract still validates.
Replace the body with your own router + Hindi-capable model + finalizer.
"""
from __future__ import annotations

import re
import os
import numpy as np
from faster_whisper import WhisperModel

_SR = 16000
_MIN_AUDIO_BYTES = int(_SR * 2.0) * 2  # Start drafting at 2.0 seconds

# Load the model once at import time to keep it warm
_HERE = os.path.dirname(os.path.abspath(__file__))
_model_path = os.path.join(_HERE, "model_weights")
_model = WhisperModel("small", device="cpu", compute_type="int8", download_root=_model_path)
_np = np

import threading

_prev_text: str = ""
_committed: str = ""
_latest_text: str = ""
_bg_thread: threading.Thread | None = None
_lock = threading.Lock()
_bg_active: bool = False
_useful_emitted: bool = False


def _stable_prefix(text: str, n: int = 1) -> str:
    matches = list(re.finditer(r"[\w'.-]+", text, flags=re.UNICODE))
    if not matches or len(matches) <= n:
        return ""
    return text[:matches[-1 - n].end()]


def draft_reset() -> None:
    """Clear per-clip state at the start of each clip."""
    global _prev_text, _committed, _latest_text, _bg_thread, _bg_active, _useful_emitted
    with _lock:
        _prev_text = ""
        _committed = ""
        _latest_text = ""
        _bg_thread = None
        _bg_active = False
        _useful_emitted = False


def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    global _prev_text, _committed, _latest_text, _bg_thread, _bg_active, _useful_emitted

    if not is_final and len(audio_buffer) < _MIN_AUDIO_BYTES:
        with _lock:
            return (_committed, len(_committed))

    if is_final:
        # Join any running background thread to free CPU
        t_to_join = None
        with _lock:
            t_to_join = _bg_thread
        if t_to_join and t_to_join.is_alive():
            t_to_join.join()

        # Final synchronous transcription for highest accuracy
        text = _transcribe(audio_buffer)
        with _lock:
            if not text:
                return (_committed, len(_committed))
            _committed = text
            return (text, len(text))

    # Intermediate request
    with _lock:
        # If we already emitted a useful partial, stop background decoding to save CPU for the final
        if _useful_emitted:
            return (_latest_text or _committed, len(_committed))

        # Start background thread to get the first useful partial
        if not _bg_active:
            _bg_active = True
            
            def _bg_task(buf):
                global _prev_text, _committed, _latest_text, _bg_active, _useful_emitted
                text = _transcribe(buf)
                with _lock:
                    if text:
                        stable_prefix = _stable_prefix(text, 1)
                        if len(stable_prefix) >= len(_committed):
                            _committed = stable_prefix
                        _prev_text = text
                        _latest_text = text
                        # If we have at least 3 committed words, consider it a useful partial
                        if len(_committed.split()) >= 3:
                            _useful_emitted = True
                    _bg_active = False
            
            _bg_thread = threading.Thread(
                target=_bg_task, 
                args=(audio_buffer,), 
                daemon=True
            )
            _bg_thread.start()

        return (_latest_text or _committed, len(_committed))


def _transcribe(audio_buffer: bytes) -> str:
    """ASR with optimized parameters for CPU streaming (beam_size=1 for speed)."""
    global _model, _np
    try:
        audio = _np.frombuffer(audio_buffer, dtype=_np.int16).astype(_np.float32) / 32768.0
        if audio.size == 0:
            return ""
        # Auto-detect language (language=None), beam_size=1, best_of=1 for max speed on CPU
        segments, _info = _model.transcribe(
            audio, 
            language=None, 
            beam_size=1, 
            best_of=1,
            temperature=0.0,
            repetition_penalty=1.1
        )
        return " ".join(s.text for s in segments).strip()
    except Exception:
        return ""



def _common_word_prefix(left: str, right: str) -> str:
    lw, rw = _words(left), _words(right)
    out: list[str] = []
    for a, b in zip(lw, rw):
        if a.lower() != b.lower():
            break
        out.append(b)
    return " ".join(out)


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'.-]+", text, flags=re.UNICODE)

