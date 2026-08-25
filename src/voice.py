"""Local voice, everything runs on the Mac, nothing leaves for the cloud.

TTS (text to speech): macOS `say` (native, local, free), AIFF then WAV via `afconvert`. Always
works on macOS.

STT (speech to text): `mlx-whisper` (local Whisper on MLX) IF installed. Otherwise it stays
DEGRADED (the endpoint says so clearly): in this environment `pip install mlx-whisper` did not
complete cleanly, so dictation is left ready-if-installed instead of blocking the rest (decision
noted; TTS is not affected).
"""
import importlib.util
import io
import os
import shutil
import subprocess  # nosec B404 - for `say`/`afconvert` (macOS local TTS): fixed list, no shell
import tempfile
import wave
from pathlib import Path

MAX_TTS_CHARS = 2000            # do not try to read huge responses aloud
# LOCAL Whisper model (MLX). tiny = fast and light; base/small = more accurate (config.env).
WHISPER_MODEL = os.environ.get("WHISPER_MODEL") or "mlx-community/whisper-tiny"


def tts_available() -> bool:
    return shutil.which("say") is not None and shutil.which("afconvert") is not None


def stt_available() -> bool:
    return importlib.util.find_spec("mlx_whisper") is not None


def synthesize(text, voice=None):
    """Text to WAV audio (bytes) via `say` (fully local). None if it could not be done."""
    text = (text or "").strip()
    if not tts_available() or not text:
        return None
    text = text[:MAX_TTS_CHARS]
    try:
        with tempfile.TemporaryDirectory() as d:
            aiff, wav = os.path.join(d, "o.aiff"), os.path.join(d, "o.wav")
            cmd = ["say", "-o", aiff] + (["-v", voice] if voice else []) + [text]
            subprocess.run(cmd, check=True, timeout=30, capture_output=True)  # noqa: S603 - fixed list; `text` goes as argv (no shell), no injection  # nosec B603
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@22050", aiff, wav],  # noqa: S603,S607 - fixed command (no shell); afconvert ships with macOS, via PATH  # nosec B603 B607
                           check=True, timeout=30, capture_output=True)
            return Path(wav).read_bytes()
    except (OSError, subprocess.SubprocessError):
        return None


def _audio_to_array(audio_bytes):
    """Decode ANY browser audio (WebM/Opus from MediaRecorder, WAV, mp4/aac...) to float32 mono at
    16 kHz, using PyAV (bundles ffmpeg EMBEDDED in the wheel, no system ffmpeg needed, fully local).
    If PyAV is missing, fall back to the pure WAV decoder (`wave`). Returns None if it could not be
    done. (Previously we only accepted WAV and the browser sends WebM, giving 'file does not start
    with RIFF'.)"""
    try:
        import av
    except ImportError:
        return _wav_to_array(audio_bytes)         # without PyAV: WAV only
    import numpy as np
    chunks = []
    try:
        with av.open(io.BytesIO(audio_bytes)) as cont:
            res = av.AudioResampler(format="s16", layout="mono", rate=16000)
            for frame in cont.decode(audio=0):
                for rf in res.resample(frame):
                    chunks.append(rf.to_ndarray().reshape(-1))
            try:
                for rf in res.resample(None):     # flush the resampler buffer (audio tail)
                    chunks.append(rf.to_ndarray().reshape(-1))
            except (TypeError, ValueError):
                pass
    except Exception:  # noqa: BLE001 - unsupported container, try pure WAV just in case
        return _wav_to_array(audio_bytes)
    if not chunks:
        return None
    return (np.concatenate(chunks).astype(np.float32) / 32768.0)


def _wav_to_array(wav_bytes):
    """Fallback WITHOUT PyAV: decode a 16-bit PCM WAV to float32 mono 16 kHz (`wave` module + numpy)."""
    import numpy as np
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            if w.getsampwidth() != 2:             # we expect 16-bit PCM
                return None
            ch, fr, raw = w.getnchannels(), w.getframerate(), w.readframes(w.getnframes())
    except (wave.Error, EOFError, OSError):
        return None
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if a.size == 0:
        return None
    if ch == 2:                                   # stereo to mono
        a = a.reshape(-1, 2).mean(axis=1)
    if fr != 16000:                               # Whisper expects 16 kHz, linear resample
        n_out = max(1, int(round(a.shape[0] * 16000 / fr)))
        a = np.interp(np.linspace(0, a.shape[0] - 1, n_out), np.arange(a.shape[0]), a).astype(np.float32)
    return a


def transcribe(wav_bytes, model=None):
    """WAV audio to text via mlx-whisper (LOCAL Whisper on MLX). Returns the text, or None if STT is
    unavailable or failed (the caller treats it as 'degraded'). LOGS the concrete error to help
    diagnose (previously it was swallowed silently)."""
    if not stt_available() or not wav_bytes:
        return None
    try:
        import mlx_whisper
        audio = _audio_to_array(wav_bytes)
        if audio is None or audio.size == 0:
            print(f"[voice] STT: could not decode the audio "
                  f"(bytes={len(wav_bytes)}, header={wav_bytes[:12]!r})", flush=True)
            return None
        r = mlx_whisper.transcribe(audio, path_or_hf_repo=(model or WHISPER_MODEL), language="es")
        return (r.get("text") or "").strip()
    except Exception as e:  # noqa: BLE001 - degraded, never breaks; but we RECORD the error
        print(f"[voice] STT failure: {type(e).__name__}: {e} "
              f"(bytes={len(wav_bytes)}, header={wav_bytes[:12]!r})", flush=True)
        return None
