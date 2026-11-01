#
# ME3 â€” Local Piper TTS service (Pipecat 1.12 TTSService)
# Runs fully on the Pi (no cloud). Implements the 1.12 abstract `run_tts` generator.
#
import asyncio
import os
import subprocess
import sys
import tempfile
import wave
from io import BytesIO
from typing import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.services.tts_service import TTSService

PIPER_VOICE = os.environ.get("ME3_PIPER_VOICE", "/home/jan/me3/models/en_US-lessac-low.onnx")
PIPER_VOICE_JSON = os.environ.get("ME3_PIPER_VOICE_JSON", PIPER_VOICE + ".json")
PIPER_PY = os.environ.get("ME3_PIPER_PYTHON", sys.executable)
_SAMPLE_RATE = 16000  # en_US-lessac-low native rate


class PiperLocalTTSService(TTSService):
    """Local TTS backed by the Piper ONNX model. No network."""

    def __init__(self, voice: str = PIPER_VOICE, **kwargs):
        kwargs.pop("settings", None)
        super().__init__(**kwargs)
        self._voice_path = voice
        self._sample_rate = _SAMPLE_RATE
        self._voice = None
        self._use_subprocess = False
        self._load()

    def _load(self):
        try:
            from piper import PiperVoice
            self._voice = PiperVoice.load(self._voice_path, config_path=PIPER_VOICE_JSON)
            logger.info(f"[piper-tts] voice loaded: {self._voice_path}")
        except Exception as e:
            logger.warning(f"[piper-tts] in-proc load failed ({e!r}); using subprocess {PIPER_PY}")
            self._use_subprocess = True
            self._voice = None

    # -- the single abstract method (pipecat 1.12) ---------------------------
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = (text or "").strip()
        if not text:
            return
        logger.info(f"[piper-tts] synthesizing: {text!r}")
        try:
            raw = await asyncio.to_thread(self._synthesize, text)
        except Exception as e:
            logger.error(f"[piper-tts] synth failed: {e!r}")
            raw = b""
        if raw:
            yield TTSAudioRawFrame(raw, self._sample_rate, 1)

    # -- synthesis ------------------------------------------------------------
    def _synthesize(self, text: str) -> bytes:
        """Return 16 kHz mono int16 PCM bytes for `text`."""
        if not self._use_subprocess and self._voice is not None:
            buf = BytesIO()
            with wave.open(buf, "wb") as wf:
                self._voice.synthesize_wav(text, wf)
            return self._extract_pcm(buf.getvalue())
        return self._synth_subprocess(text)

    @staticmethod
    def _extract_pcm(wav_bytes: bytes) -> bytes:
        import struct
        if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
            return wav_bytes
        i = 12
        while i + 8 <= len(wav_bytes):
            cid = wav_bytes[i:i+4]
            size = struct.unpack("<I", wav_bytes[i+4:i+8])[0]
            if cid == b"data":
                return wav_bytes[i+8:i+8+size]
            i += 8 + size + (size & 1)
        return b""

    def _synth_subprocess(self, text: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        try:
            code = (
                "import wave,sys;"
                f"from piper import PiperVoice;"
                f"v=PiperVoice.load({self._voice_path!r},config_path={PIPER_VOICE_JSON!r});"
                f"wf=wave.open({path!r},'wb');v.synthesize_wav(sys.argv[1],wf);wf.close()"
            )
            r = subprocess.run([PIPER_PY, "-c", code, text], capture_output=True, timeout=60)
            if r.returncode != 0:
                raise RuntimeError(f"piper subprocess failed: {r.stderr.decode()[:200]}")
            with wave.open(path, "rb") as wf:
                return wf.readframes(wf.getnframes())
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


