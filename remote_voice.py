#
# ME3 -- Remote ASR/TTS via the DGX3 HPC (SHARC UP VPN).
#
#   ASR: Qwen2-Audio-7B-Instruct    http://10.153.60.233:8000  /v1/chat/completions
#   TTS: Qwen3-TTS-1.7B-CustomVoice  http://10.153.60.233:8001  /v1/audio/speech
#
# These replace the local whisper.cpp (ASR) and Piper (TTS) so the heavy neural
# legs run on the DGX3's GPUs, easing the Pi's CPU load (and power draw). The
# VAD (Silero) stays local for low-latency turn detection.
#
import asyncio
import base64
import io
import os
import time
import wave
from collections.abc import AsyncGenerator

import numpy as np

from loguru import logger

from pipecat.frames.frames import (Frame, TranscriptionFrame, InterimTranscriptionFrame,
                                   TTSAudioRawFrame, ErrorFrame)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService

# --- Config (overridable via env) -----------------------------------------
# ASR/TTS provider: "groq" (default) or "dgx3" (legacy DGX3 Qwen2-Audio/Qwen3-TTS).
# Groq ASR uses the OpenAI-compatible /v1/audio/transcriptions (Whisper) endpoint;
# DGX3 uses /v1/chat/completions with an input_audio content part.
ASR_PROVIDER = os.environ.get("ME3_ASR_PROVIDER", "groq")
# Groq (OpenAI-compatible) credentials + endpoints.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY",
                              "***REMOVED-GROQ-KEY***")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
ASR_MODEL = os.environ.get("ME3_ASR_MODEL", "whisper-large-v3-turbo")
# DGX3 (legacy fallback for TTS / ASR)
DGX3_HOST = os.environ.get("ME3_DGX3_HOST", "10.153.60.233")
ASR_BASE_URL = os.environ.get("ME3_ASR_BASE_URL", f"http://{DGX3_HOST}:8000/v1")
TTS_BASE_URL = os.environ.get("ME3_TTS_BASE_URL", f"http://{DGX3_HOST}:8001/v1")
DGX3_API_KEY = os.environ.get("ME3_DGX3_API_KEY",
                              "***REMOVED-DGX3-KEY***")
TTS_MODEL = os.environ.get("ME3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
TTS_VOICE = os.environ.get("ME3_TTS_VOICE", "vivian")
# TTS_SAMPLE_RATE: the rate we hand to the pipeline / welcome path. The DGX3
# Qwen3-TTS emits a FIXED 24 kHz PCM stream (it rejects a sample_rate override
# with HTTP 400), so _synthesize resamples 24 kHz -> TTS_SAMPLE_RATE with soxr.
TTS_SAMPLE_RATE = int(os.environ.get("ME3_TTS_SAMPLE_RATE", "48000"))
TTS_NATIVE_SR = int(os.environ.get("ME3_TTS_NATIVE_SR", "24000"))  # DGX3 output rate
NOISE_FLOOR_RMS = float(os.environ.get("ME3_NOISE_FLOOR_RMS", "0.004"))

# --- Lead-in silence (USB first-chunk loss) --------------------------------
# The EMEET M1A USB output drops the head of every freshly-opened stream
# (~200-450 ms). We prepend this much silence to EVERY playback (welcome +
# each reply) so the first word is not cut. Same fix as ME2 (WARMUP_S) and the
# existing welcome path. Env-tunable.
LEAD_SILENCE_S = float(os.environ.get("ME3_LEAD_SILENCE_S", "0.45"))

# --- edge-tts (primary voice) ----------------------------------------------
# The chosen Boots voice: Microsoft "Roger" (en-US-RogerNeural) via the free
# edge-tts endpoint, +30 Hz pitch, +8% rate. No API key / no hard limit (it is
# the public MS speech endpoint), but it NEEDS INTERNET and can be region
# -blocked or rate-limited, so it is used as the PRIMARY with a fail-soft
# fallback to the local-over-VPN Qwen3-TTS (vivian). Env-tunable:
#   ME3_TTS_BACKEND = edge | qwen3 | auto (default auto = edge, fall back to qwen3)
EDGE_VOICE = os.environ.get("ME3_EDGE_VOICE", "en-US-RogerNeural")
EDGE_PITCH = os.environ.get("ME3_EDGE_PITCH", "+30Hz")
EDGE_RATE = os.environ.get("ME3_EDGE_RATE", "+8%")
EDGE_TIMEOUT_S = float(os.environ.get("ME3_EDGE_TIMEOUT_S", "8.0"))
TTS_BACKEND = os.environ.get("ME3_TTS_BACKEND", "auto").lower()


def _to_wav_bytes(samples_f32: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Encode float32 [-1,1] mono samples as a 16-bit PCM WAV (full container)."""
    s = np.clip(samples_f32, -1.0, 1.0)
    pcm = (s * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _clean_transcript(text: str) -> str:
    """Strip Qwen2-Audio's chatty preamble ("The transcription is: ...",
    surrounding quotes, "Transcription:" labels) so the LLM sees only the
    user's words. Best-effort; returns the text stripped if nothing matches."""
    import re
    t = (text or "").strip()
    # Drop leading preamble phrases (case-insensitive). Use \b so a partial
    # match (e.g. "transcript" inside "transcription") can't over-strip.
    for pre in (
        r"^(?:the\s+)?transcription\s+is\s*[:\-]?\s*",
        r"^(?:the\s+)?audio\s+says\s*[:\-]?\s*",
        r"^here\s+is\s+the\s+transcription\s*[:\-]?\s*",
        r"^i\s+heard\s*[:\-]?\s*",
        r"^transcript\s+is\s*[:\-]?\s*",
        r"^transcription\s*[:\-]?\s*",
        r"^transcript\s*[:\-]?\s*",
    ):
        t = re.sub(pre, "", t, count=1, flags=re.IGNORECASE)
    # Strip one layer of surrounding quotes.
    t = t.strip()
    if len(t) >= 2 and t[0] in "\"'“”‘’" and t[-1] in "\"'“”‘’":
        t = t[1:-1].strip()
    return t.strip()


# =========================================================================== #
# ASR -- Qwen2-Audio via /v1/chat/completions (audio in, text out)
# =========================================================================== #
class Qwen2AudioSTTService(SegmentedSTTService):
    """Remote ASR: SegmentedSTTService buffers VAD-delimited mic audio and
    hands the segment to run_stt as a 16 kHz WAV. We base64 that WAV and send
    it to DGX3's Qwen2-Audio model, then emit a TranscriptionFrame."""

    def __init__(self, **kwargs):
        from pipecat.services.stt_service import STTSettings
        from pipecat.transcriptions.language import Language
        settings = STTSettings(
            model=ASR_MODEL,
            language=Language.EN,
        )
        super().__init__(
            sample_rate=16000,  # EMEET M1A input rate (mic is 16 kHz)
            settings=settings, **kwargs)
        self._client = None
        import httpx
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers={"Authorization": f"Bearer {DGX3_API_KEY}"},
        )
        logger.info(f"[asr-dgx3] Qwen2-Audio ASR ready ({ASR_BASE_URL})")

    # FIX (LLM never receives transcript): pipecat 1.12's SegmentedSTTService
    # pushes the TranscriptionFrame from a background _segment_task. In this
    # deployment that downstream push lands in the next processor's queue but is
    # not reliably run, so the frame never reaches the user_aggregator and the
    # LLM errors with "No user query found" (Boots never replies). We therefore
    # deliver each TranscriptionFrame directly to the downstream
    # user_aggregator via the processor chain, bypassing the tapped
    # StatusIndicator. We do NOT also run the normal super().push_frame for the
    # transcript, to avoid delivering the user turn twice.
    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if direction == FrameDirection.DOWNSTREAM and isinstance(
                frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            try:
                # Walk the processor chain: self -> status -> user_aggregator.
                nxt = getattr(self, "_next", None)
                for _ in range(3):
                    if nxt is None:
                        break
                    if type(nxt).__name__ == "LLMUserAggregator":
                        await nxt.queue_frame(frame, FrameDirection.DOWNSTREAM)
                        return  # delivered; skip the unreliable normal push
                    nxt = getattr(nxt, "_next", None)
            except Exception as e:
                logger.debug(f"[asr-dgx3] aggregator re-deliver failed: {e!r}")
        await super().push_frame(frame, direction)

    def _transcribe_sync(self, wav_b64: str) -> str:
        """Blocking ASR call. Run via asyncio.to_thread.

        Provider "groq" (default): POST the raw WAV to the OpenAI-compatible
        Groq /v1/audio/transcriptions (Whisper) endpoint as multipart/form-data.
        Provider "dgx3" (legacy): POST a chat/completions request with an
        input_audio content part to the DGX3 Qwen2-Audio service.
        """
        import json
        import urllib.request
        if ASR_PROVIDER == "groq":
            return self._transcribe_groq(wav_b64)
        return self._transcribe_dgx3(wav_b64)

    @staticmethod
    def _transcribe_groq(wav_b64: str) -> str:
        """Groq Whisper via /v1/audio/transcriptions.

        Uses the `requests` client (openai SDK's dependency, already in the
        venv) instead of urllib: Groq's Cloudflare edge (WAF "error 1010")
        bans urllib's browser signature on the multipart ASR upload, while
        `requests` passes. Returns the transcript text (may be empty)."""
        import base64
        import requests
        wav_bytes = base64.b64decode(wav_b64)
        r = requests.post(
            GROQ_BASE_URL + "/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("segment.wav", wav_bytes, "audio/wav")},
            data={"model": ASR_MODEL},
            timeout=60,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Groq ASR HTTP {r.status_code}: {r.text[:200]}")
        return (r.json().get("text") or "").strip()

    @staticmethod
    def _transcribe_dgx3(wav_b64: str) -> str:
        """Legacy DGX3 Qwen2-Audio ASR via /v1/chat/completions."""
        import json
        import urllib.request
        body = {
            "model": "Qwen/Qwen2-Audio-7B-Instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio",
                     "input_audio": {"data": wav_b64, "format": "wav"}},
                    {"type": "text",
                     "text": "Transcribe the audio exactly. Return only the words "
                             "spoken, with no preamble, quotes, or labels."},
                ],
            }],
            "max_tokens": 120,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            ASR_BASE_URL + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {DGX3_API_KEY}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("choices", [{}])[0].get("message", {})
                .get("content") or "").strip()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """pipecat SegmentedSTTService hook. `audio` is the VAD-delimited
        segment -- a 16 kHz 16-bit mono WAV container by default
        (wants_wav_segments=True); we also tolerate raw PCM. We gate on a
        noise floor, then base64 the WAV and send it to DGX3 Qwen2-Audio."""
        if not audio:
            return
        # Normalise to a WAV container (16 kHz 16-bit mono) for the DGX3 API.
        if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
            wav_bytes = audio  # already a WAV container
        else:
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            wav_bytes = _to_wav_bytes(samples, 16000)
        # Noise-floor gate on the raw PCM payload of the WAV.
        try:
            pcm = np.frombuffer(wav_bytes[44:], dtype=np.int16)
            rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float32) / 32768.0))))
        except Exception:
            rms = 0.0
        if rms < NOISE_FLOOR_RMS:
            logger.debug(f"[asr-dgx3] turn below noise floor (rms={rms:.5f}); skip")
            return
        wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
        try:
            text = await asyncio.to_thread(self._transcribe_sync, wav_b64)
            text = _clean_transcript(text)
            if text:
                logger.info(f"[asr-dgx3] transcript: {text!r}")
                # FIX (UI "heard" bubble): pipecat 1.12's SegmentedSTTService
                # pushes the TranscriptionFrame from a background _segment_task,
                # and that downstream push is not reliably delivered to the next
                # processor (it lands in the process queue but is not run), so
                # StatusIndicator/LLM never saw the transcript. We therefore push
                # the transcript straight to the Boots UI here (where we have the
                # value in hand), guaranteeing the user bubble + heard field.
                self._push_heard_to_ui(text)
                yield TranscriptionFrame(text, "", "dgx3")
        except Exception as e:
            logger.error(f"[asr-dgx3] transcription failed: {e!r}")
            yield ErrorFrame(error=f"DGX3 ASR failure: {e!r}")

    def _push_heard_to_ui(self, text: str):
        """Push the transcript to the Boots UI (me3.heard) in a background thread.

        The pipecat TranscriptionFrame push is unreliable in 1.12 (see run_stt),
        so we update the UI directly from the value we already have. Best-effort:
        never raises into the ASR path.
        """
        try:
            import json as _j, os as _os, threading as _t, urllib.request as _u
            base = _os.environ.get("ME3_BOOTSUI", "http://127.0.0.1:8330")

            def _go():
                try:
                    body = {"heard": text[:300], "phase": "listening"}
                    req = _u.Request(base + "/me3", data=_j.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
                    with _u.urlopen(req, timeout=0.4) as r:
                        r.read()
                except Exception:
                    pass

            _t.Thread(target=_go, daemon=True).start()
        except Exception:
            pass

    async def cleanup(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# =========================================================================== #
# TTS -- Qwen3-TTS via /v1/audio/speech (OpenAI-compatible, returns PCM)
# =========================================================================== #
class Qwen3TTSService(TTSService):
    """Remote TTS: sends text to DGX3's Qwen3-TTS /v1/audio/speech and yields
    TTSAudioRawFrame chunks at TTS_SAMPLE_RATE (48 kHz to match the EMEET)."""

    def __init__(self, **kwargs):
        from pipecat.services.tts_service import TTSSettings
        from pipecat.transcriptions.language import Language
        settings = TTSSettings(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            language=Language.EN,
        )
        super().__init__(sample_rate=TTS_SAMPLE_RATE, settings=settings, **kwargs)
        import httpx
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers={"Authorization": f"Bearer {DGX3_API_KEY}"},
        )
        self._chunk_size = 4096 * 4
        logger.info(f"[tts-dgx3] Qwen3-TTS ready ({TTS_BASE_URL}, voice={TTS_VOICE})")

    def _synthesize(self, text: str, sample_rate: int = 16000) -> bytes:
        """Call DGX3 Qwen3-TTS and return 16-bit mono PCM bytes at
        `sample_rate`. The DGX3 emits a fixed 24 kHz PCM stream (it rejects a
        sample_rate override with HTTP 400), so we resample 24 kHz ->
        `sample_rate` with soxr. Used both by run_tts (48 kHz, pipeline) and
        by agent.py's welcome path (16 kHz, then soxr-resampled to 48 kHz)."""
        import json
        import urllib.request
        import soxr
        payload = {
            "model": TTS_MODEL,
            "input": text,
            "voice": TTS_VOICE,
            "response_format": "pcm",
            # no "sample_rate" key: DGX3 Qwen3-TTS rejects it (HTTP 400) and
            # always returns its native TTS_NATIVE_SR (24 kHz) PCM.
        }
        req = urllib.request.Request(
            TTS_BASE_URL + "/audio/speech",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {DGX3_API_KEY}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status != 200:
                raise RuntimeError(f"DGX3 TTS HTTP {resp.status}")
            native = resp.read()
        if sample_rate == TTS_NATIVE_SR:
            return native
        samples = np.frombuffer(native, dtype=np.int16).astype(np.float64)
        out = soxr.resample(samples, TTS_NATIVE_SR, sample_rate)
        out = np.clip(out, -32768, 32767)
        return out.astype(np.int16).tobytes()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = (text or "").strip()
        if not text:
            return
        logger.info(f"[tts-dgx3] synthesizing: {text!r}")
        try:
            raw = await asyncio.to_thread(self._synthesize, text, TTS_SAMPLE_RATE)
        except Exception as e:
            logger.error(f"[tts-dgx3] synth failed: {e!r}")
            yield ErrorFrame(error=f"DGX3 TTS failure: {e!r}")
            return
        if not raw:
            return
        # Lead-in silence: the EMEET M1A USB output drops the head of every
        # freshly-opened stream, so prepend LEAD_SILENCE_S of zeros.
        if LEAD_SILENCE_S > 0:
            raw = b"\x00\x00" * int(TTS_SAMPLE_RATE * LEAD_SILENCE_S) + raw
        # Yield in ~0.25 s chunks so the audio streams instead of buffering.
        chunk = max(1, TTS_SAMPLE_RATE * 2 // 4)
        for i in range(0, len(raw), chunk):
            yield TTSAudioRawFrame(raw[i:i + chunk], TTS_SAMPLE_RATE, 1,
                                   context_id=context_id)

    async def cleanup(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None



# =========================================================================== #
# edge-tts (primary voice: en-US-RogerNeural, +30Hz, +8%) with Qwen3 fallback
# =========================================================================== #
class _EdgeSynth:
    """text -> `out_sr` int16 mono PCM via edge-tts + miniaudio.
    Raises on any failure so the caller can fall back to Qwen3-TTS."""

    @staticmethod
    def synthesize(text: str, out_sr: int = TTS_SAMPLE_RATE) -> bytes:
        import asyncio
        import concurrent.futures
        import os as _os
        import tempfile
        import miniaudio
        import edge_tts
        import numpy as _np
        mp3 = _os.path.join(tempfile.gettempdir(), "me3_edge_%d.mp3" % _os.getpid())

        def _do():
            asyncio.run(asyncio.wait_for(
                edge_tts.Communicate(text, EDGE_VOICE,
                                     pitch=EDGE_PITCH, rate=EDGE_RATE).save(mp3),
                timeout=EDGE_TIMEOUT_S))

        try:
            # run in a fresh thread so there is no running-loop conflict
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                ex.submit(_do).result(timeout=EDGE_TIMEOUT_S + 3)
            dec = miniaudio.decode_file(
                mp3, output_format=miniaudio.SampleFormat.FLOAT32,
                sample_rate=out_sr, nchannels=1)
            arr = _np.clip(_np.asarray(dec.samples, dtype=_np.float64), -1.0, 1.0)
            return (arr * 32767.0).astype(_np.dtype("<i2")).tobytes()
        finally:
            try:
                if _os.path.exists(mp3):
                    _os.unlink(mp3)
            except Exception:
                pass


class EdgeTTSService(Qwen3TTSService):
    """TTS that uses edge-tts (Roger +30Hz +8%) as the PRIMARY voice and the
    DGX3 Qwen3-TTS (vivian) as the fail-soft fallback. Every pipeline output
    (run_tts) leads with LEAD_SILENCE_S of silence (EMEET USB first-chunk loss).
    _synthesize() is clean (no lead) so the welcome path can add its own lead."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._edge_ok = True
        self._edge_fail_ts = 0.0
        logger.info(
            f"[tts-edge] edge-tts primary ({EDGE_VOICE} {EDGE_PITCH} {EDGE_RATE}), "
            f"qwen3 fallback, backend={TTS_BACKEND}, lead={LEAD_SILENCE_S}s")

    def _edge_enabled(self) -> bool:
        if TTS_BACKEND == "qwen3":
            return False
        if not self._edge_ok and (time.time() - self._edge_fail_ts) < 60.0:
            return False  # back off 60s after a failure
        return True

    def _synthesize(self, text: str, sample_rate: int = 16000) -> bytes:
        if self._edge_enabled():
            try:
                t0 = time.time()
                raw = _EdgeSynth.synthesize(text, sample_rate)
                if raw:
                    logger.info(f"[tts-edge] edge-tts OK ({time.time()-t0:.2f}s, {len(raw)} B)")
                    return raw
                logger.warning("[tts-edge] edge-tts empty; qwen3 fallback")
            except Exception as e:
                self._edge_ok = False
                self._edge_fail_ts = time.time()
                logger.warning(f"[tts-edge] edge-tts failed ({e!r}); qwen3 fallback")
        return super()._synthesize(text, sample_rate)


class BootsTTSService(EdgeTTSService):
    """The ME3 TTS used by run_bot(): edge-tts (Roger) primary, Qwen3 (vivian)
    fallback, every reply leads with LEAD_SILENCE_S. The welcome calls
    _synthesize() on the same instance, so the welcome voice matches the
    replies exactly. Select backend via ME3_TTS_BACKEND (edge|qwen3|auto)."""
    pass
