#!/usr/bin/env python3
# Boots -- local Piper voice shaped to sound like the edge-tts voice (2026-11-18).
#
# edge-tts (Microsoft "Roger", +20 Hz, +8 %) needs the internet. When it fails,
# remote_voice used to fall back to Qwen3-TTS over the HPC VPN: a different,
# female voice, and also network-bound. This is the local fallback: Piper
# en_US-ryan-medium on the Pi's CPU (~0.4 s per short sentence), tuned to
# Roger's cadence. Measured on 3 reference sentences against Roger:
#
#   pitch   Ryan 154 Hz vs Roger 169 Hz -> played PITCH x faster (tape trick)
#           with length_scale stretched by the same factor, so pace is kept
#   pace    LENGTH_SCALE, speech time within ~2 % of Roger's
#   pauses  Roger pauses ~0.5-0.7 s at commas, Piper ~0.2 s. Text is split at
#           clause and sentence ends and silence inserted; a short tag after a
#           comma (", Jan.") stays attached, as Roger runs it on.
#
# Env: ME3_PIPER_FB_VOICE (model path), ME3_PIPER_PITCH, ME3_PIPER_LS,
#      ME3_PIPER_COMMA_S, ME3_PIPER_SENT_S.
import os
import re
import threading

import numpy as np

VOICE = os.environ.get("ME3_PIPER_FB_VOICE", "/home/jan/me3/models/en_US-ryan-medium.onnx")
PITCH = float(os.environ.get("ME3_PIPER_PITCH", "1.08"))
LENGTH_SCALE = float(os.environ.get("ME3_PIPER_LS", "1.00"))
COMMA_S = float(os.environ.get("ME3_PIPER_COMMA_S", "0.55"))
SENT_S = float(os.environ.get("ME3_PIPER_SENT_S", "0.50"))
NOISE_SCALE = 0.667      # Piper defaults for expressive speech
NOISE_W = 0.8
SHORT_TAG_WORDS = 2      # ", Jan." / ", please." stay on the clause before

_lock = threading.Lock()
_voice = {"v": None, "err": None}


def available():
    return os.path.exists(VOICE)


def _load():
    with _lock:
        if _voice["v"] is None and _voice["err"] is None:
            try:
                from piper import PiperVoice
                _voice["v"] = PiperVoice.load(VOICE)
            except Exception as e:      # model missing / onnxruntime problem
                _voice["err"] = e
        if _voice["v"] is None:
            raise RuntimeError(f"piper voice unavailable: {_voice['err']!r}")
        return _voice["v"]


def preload():
    """Load the model in the background so the first fallback isn't slow."""
    if available():
        threading.Thread(target=lambda: _try(_load), daemon=True).start()


def _try(fn):
    try:
        fn()
    except Exception:
        pass


def pieces(text):
    """[(piece, pause_after_s)]: split at clause and sentence ends; a short
    tag after a comma stays attached to what comes before."""
    parts = [p for p in re.split(r"(?<=[,;:.!?])\s+", (text or "").strip()) if p]
    out = []
    for p in parts:
        if out and out[-1][0].rstrip()[-1:] in ",;:" and len(p.split()) <= SHORT_TAG_WORDS:
            out[-1] = (out[-1][0] + " " + p, 0.0)
        else:
            out.append((p, 0.0))
    res = []
    for i, (p, _) in enumerate(out):
        last = i == len(out) - 1
        end = p.rstrip()[-1:]
        res.append((p, 0.0 if last else (SENT_S if end in ".!?" else COMMA_S)))
    return res


def _trim(a, thr=0.01, pad_head=200, pad_tail=400):
    nz = np.where(np.abs(a) > thr)[0]
    if not len(nz):
        return a[:0]
    return a[max(0, nz[0] - pad_head):nz[-1] + pad_tail]


def synthesize(text, out_sr=48000):
    """text -> int16 mono PCM bytes at out_sr, in the Roger-like cadence."""
    import soxr
    from piper import SynthesisConfig
    v = _load()
    sr = v.config.sample_rate
    cfg = SynthesisConfig(length_scale=LENGTH_SCALE * PITCH, noise_scale=NOISE_SCALE,
                          noise_w_scale=NOISE_W)
    parts = []
    for piece, pause in pieces(text):
        for chunk in v.synthesize(piece, syn_config=cfg):
            parts.append(_trim(chunk.audio_float_array))
        if pause:
            parts.append(np.zeros(int(sr * pause), np.float32))
    if not parts:
        return b""
    x = np.concatenate(parts).astype(np.float32)
    # Tape trick: treat the samples as recorded PITCH x faster -> higher pitch;
    # length_scale above was stretched by PITCH so the pace stays the same.
    y = soxr.resample(x, sr * PITCH, out_sr)
    return (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
