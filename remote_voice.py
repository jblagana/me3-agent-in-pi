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
# Keys come ONLY from the environment (/etc/me3/groq.env on the Pi) -- no
# hardcoded fallbacks: this repo is public (2026-11-07 secrets cleanup).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
ASR_MODEL = os.environ.get("ME3_ASR_MODEL", "whisper-large-v3-turbo")
# DGX3 (legacy fallback for TTS / ASR)
DGX3_HOST = os.environ.get("ME3_DGX3_HOST", "10.153.60.233")
ASR_BASE_URL = os.environ.get("ME3_ASR_BASE_URL", f"http://{DGX3_HOST}:8000/v1")
TTS_BASE_URL = os.environ.get("ME3_TTS_BASE_URL", f"http://{DGX3_HOST}:8001/v1")
DGX3_API_KEY = os.environ.get("ME3_DGX3_API_KEY", "")
TTS_MODEL = os.environ.get("ME3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
TTS_VOICE = os.environ.get("ME3_TTS_VOICE", "vivian")
TTS_SPEED = float(os.environ.get("ME3_TTS_SPEED", "1.08"))   # DGX3 fallback voice: +8 %
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

# Turn-start hook: agent.py points this at StatusIndicator.on_transcript.
ON_TRANSCRIPT = None

# fd-benchmark stage mark (2026-11-07). Our transcript is re-delivered
# straight to the aggregator (push_frame below), so the benchmark's Pipecat
# observer never sees the TranscriptionFrame -> ASR final + WER would read
# "not reported". Its documented level-2 hook: mark("asr_final", text=...).
# Only importable when launch_assistant.py runs Boots; a no-op otherwise.
try:
    from fdbench_marks import mark as _fdb_mark
except Exception:
    _fdb_mark = None


import re as _re
_URL_RE = _re.compile(r"https?://[^\s<>\"'\])]+")
_URL_TOK = "\x00URL\x00"
# "<url> and <url>", "<url>, <url>, or <url>" -> one plural phrase
_URL_RUN_RE = _re.compile(_re.escape(_URL_TOK) + r"(?:\s*(?:,\s*)?(?:and|or|,)\s*" + _re.escape(_URL_TOK) + r")+")


# Spoken site names (2026-11-07): Boots never reads a domain out loud
# ("en dot wikipedia dot org"); it says the site's name instead.
_SITE_NAMES = {"wikipedia": "Wikipedia", "britannica": "Britannica", "statista": "Statista",
               "nationalgeographic": "National Geographic", "bbc": "BBC", "cnn": "CNN",
               "nasa": "NASA", "youtube": "YouTube", "github": "GitHub", "reddit": "Reddit",
               "google": "Google", "nytimes": "the New York Times", "theguardian": "the Guardian",
               "reuters": "Reuters", "apnews": "AP News", "phivolcs": "PHIVOLCS",
               "pagasa": "PAGASA", "inquirer": "the Inquirer", "rappler": "Rappler",
               "worldbank": "the World Bank", "who": "the WHO", "un": "the UN"}
_SLDS = {"com", "org", "net", "gov", "edu", "ac", "co", "mil", "int"}
_SKIP_LABELS = {"www", "m", "en", "mobile", "amp"}
_TLDS = ("com|org|net|gov|edu|io|ai|co|ph|uk|us|ca|au|de|fr|jp|info|news|"
         "dev|app|me|tv|int|mil")
_BARE_DOMAIN_RE = _re.compile(r"\b(?:[a-z0-9-]+\.)+(?:%s)\b(?:/[^\s,;!?)]*)?" % _TLDS, _re.I)


def site_name(host: str) -> str:
    """'en.wikipedia.org' -> 'Wikipedia', 'www.britannica.com' -> 'Britannica',
    'phivolcs.dost.gov.ph' -> 'PHIVOLCS', 'tanawexplorer.com' -> 'Tanawexplorer'."""
    labels = [l for l in str(host or "").lower().split("/")[0].split(":")[0].split(".") if l]
    if len(labels) >= 2:
        labels = labels[:-1]                              # TLD
        if len(labels) >= 2 and labels[-1] in _SLDS:
            labels = labels[:-1]                          # .gov.ph / .co.uk
    labels = [l for l in labels if l not in _SKIP_LABELS] or labels
    if not labels:
        return "the site"
    for l in labels:                                      # any well-known name wins
        if l in _SITE_NAMES:
            return _SITE_NAMES[l]
    return labels[-1].replace("-", " ").title()


def _speakable(text: str) -> str:
    """Text for TTS with links NOT read aloud (2026-11-07): the bubble keeps
    the clickable URL; the voice says "the link in the chat" instead of
    "h t t p colon slash slash one nine two dot…". Trailing sentence
    punctuation stays outside the replacement. Pipecat's sentence splitter
    keeps a URL inside one chunk (verified), so per-chunk replacement is safe."""
    if "http" not in text:
        # bare domains ("I'm reading en.wikipedia.org") -> the site's name
        return _BARE_DOMAIN_RE.sub(lambda m: site_name(m.group(0)), text) if "." in text else text

    def _tok(m):
        u = m.group(0)
        tail = _re.search(r"[.,;:!?]+$", u)
        return _URL_TOK + (tail.group(0) if tail else "")

    t = _URL_RE.sub(_tok, text)
    t = _URL_RUN_RE.sub("the links in the chat", t)
    t = t.replace(_URL_TOK, "the link in the chat")
    return _BARE_DOMAIN_RE.sub(lambda m: site_name(m.group(0)), t)


# --- Groq ASR transport (2026-11-07) ----------------------------------------
# ONE kept-alive HTTPS session (a fresh TLS handshake per utterance cost ~0.7 s),
# FLAC instead of WAV (half the upload), and a pre-warm the moment the user
# STARTS speaking so an idle-closed connection is open again by the end.
# ME3_ASR_LANGUAGE (e.g. "en") pins the language; unset = Whisper auto-detect
# (keeps Taglish/Tagalog working; it made no speed difference in the test).
ASR_LANGUAGE = os.environ.get("ME3_ASR_LANGUAGE", "").strip()
_GROQ = {"session": None, "warm_ts": 0.0}


def _groq_session():
    import requests
    if _GROQ["session"] is None:
        _GROQ["session"] = requests.Session()
    return _GROQ["session"]


def prewarm_asr():
    """Open (or keep open) the HTTPS connection to Groq, off the audio path.
    Called on 'user started speaking'; throttled to once per 20 s."""
    import threading
    now = time.time()
    if now - _GROQ["warm_ts"] < 20:
        return
    _GROQ["warm_ts"] = now

    def _go():
        try:
            _groq_session().get(GROQ_BASE_URL + "/models", timeout=5,
                                headers={"Authorization": f"Bearer {GROQ_API_KEY}"})
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


def _wav_to_flac(wav_bytes: bytes):
    """WAV -> FLAC bytes (lossless, ~half the size); None if unavailable."""
    try:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16")
        out = io.BytesIO()
        sf.write(out, data, sr, format="FLAC")
        return out.getvalue()
    except Exception:
        return None


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
        wav_bytes = base64.b64decode(wav_b64)
        # Faster ASR leg (2026-11-07, measured on the Pi, 4.2 s utterance):
        #   new connection per call + WAV ....... 1.54 s
        #   kept-alive Session ................... 0.82 s  (no TLS handshake)
        #   + FLAC upload (61 KB vs 132 KB) ...... 0.61 s
        name, data, mime = "segment.wav", wav_bytes, "audio/wav"
        flac = _wav_to_flac(wav_bytes)
        if flac:
            name, data, mime = "segment.flac", flac, "audio/flac"
        form = {"model": ASR_MODEL, "response_format": "json"}
        if ASR_LANGUAGE:
            form["language"] = ASR_LANGUAGE
        r = _groq_session().post(
            GROQ_BASE_URL + "/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (name, data, mime)}, data=form, timeout=60)
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
            # No words at all (".", "…" from a noise burst) -> not a turn
            # (2026-11-07: "." was sent to the LLM as a user message).
            if not _re.search(r"[A-Za-z0-9]", text or ""):
                if text:
                    logger.info(f"[asr-dgx3] dropped word-less transcript {text!r}")
                text = ""
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
                if _fdb_mark is not None:
                    _fdb_mark("asr_final", text=text)
                # Turn-start hook (agent.py sets it to StatusIndicator.
                # on_transcript): the TranscriptionFrame is re-delivered
                # straight to the aggregator (push_frame above), so the
                # status tap never sees it otherwise. Never breaks ASR.
                if ON_TRANSCRIPT is not None:
                    try:
                        ON_TRANSCRIPT(text)
                    except Exception as _e:
                        logger.warning(f"[asr-dgx3] ON_TRANSCRIPT hook failed: {_e!r}")
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
                    # new_turn: one per transcript, so the server starts a
                    # fresh turn even when the words repeat ("hey" twice)
                    # -- heard-text change alone can't see a repeat.
                    body = {"heard": text[:300], "phase": "listening",
                            "new_turn": True}
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
        # Speaking rate +8 % like the primary voice (Roger +8 %), 2026-11-07.
        # The server honours `speed` (tested: 1.5 -> 2.5 s vs 4.0 s; 0.7 ->
        # 7.0 s); retried without it below if a server version rejects it.
        if TTS_SPEED and TTS_SPEED != 1.0:
            payload["speed"] = TTS_SPEED

        def _post(body):
            req = urllib.request.Request(
                TTS_BASE_URL + "/audio/speech",
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": f"Bearer {DGX3_API_KEY}",
                         "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"DGX3 TTS HTTP {resp.status}")
                return resp.read()
        try:
            native = _post(payload)
        except urllib.error.HTTPError as e:
            if e.code == 400 and "speed" in payload:     # older server: no speed
                payload.pop("speed")
                native = _post(payload)
            else:
                raise
        if sample_rate == TTS_NATIVE_SR:
            return native
        samples = np.frombuffer(native, dtype=np.int16).astype(np.float64)
        out = soxr.resample(samples, TTS_NATIVE_SR, sample_rate)
        out = np.clip(out, -32768, 32767)
        return out.astype(np.int16).tobytes()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = _speakable((text or "").strip())   # never read URLs aloud
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


def _mp3_to_pcm24k(buf):
    """Decode the MP3 received so far -> int16 mono 24 kHz (edge-tts's native rate)."""
    import miniaudio
    try:
        d = miniaudio.decode(bytes(buf), output_format=miniaudio.SampleFormat.SIGNED16,
                             nchannels=1, sample_rate=24000)
        return np.asarray(d.samples, dtype=np.int16)
    except miniaudio.DecodeError:
        return np.zeros(0, np.int16)


# Lead silence for PIPELINE replies (2026-11-07). The 0.45 s LEAD_SILENCE_S
# covers the EMEET dropping the head of a FRESHLY OPENED stream (the welcome
# path opens its own); the pipeline's output stream stays open, so replies
# need far less. Benchmark mode (direct feed, no device) uses 0.
REPLY_LEAD_S = float(os.environ.get(
    "ME3_REPLY_LEAD_S", "0" if os.environ.get("ME3_BENCH") == "1" else "0.1"))
# (0.1 s since the speaker stream is kept running between replies -- agent.py
#  _KeepAliveMixer -- so there is no stream (re)start for padding to cover.)
_STREAM_FIRST_BYTES = 720      # ~0.12 s of 48 kbit/s MP3 -> first audio ASAP
_STREAM_NEXT_BYTES = 2400      # then decode every ~0.4 s of audio


class StreamingEdgeTTSService(EdgeTTSService):
    """edge-tts STREAMED (2026-11-07): plays the first audio while the rest of
    the sentence is still being synthesized, instead of synthesizing the whole
    sentence to an MP3 file and decoding it before the first sample.

    Measured on the Pi: first audio ~0.7-1.0 s instead of 1.0-1.9 s, same voice.
    How: re-decode the growing MP3 buffer and emit only the new samples
    (verified bit-identical to a single full decode -> no seams), then a
    streaming soxr resampler 24 kHz -> 48 kHz. Falls back to the old
    full-synthesis path (and its Qwen3 fallback) if streaming fails before any
    audio was sent."""

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = _speakable((text or "").strip())   # never read URLs aloud
        if not text:
            return
        if not self._edge_enabled():
            async for f in super().run_tts(text, context_id):
                yield f
            return
        import edge_tts, soxr
        t0 = time.time()
        rs = soxr.ResampleStream(24000, TTS_SAMPLE_RATE, 1, dtype="int16")
        buf = bytearray(); emitted = 0; last_len = 0; sent = False
        try:
            if REPLY_LEAD_S > 0:
                yield TTSAudioRawFrame(b"\x00\x00" * int(TTS_SAMPLE_RATE * REPLY_LEAD_S),
                                       TTS_SAMPLE_RATE, 1, context_id=context_id)
            comm = edge_tts.Communicate(text, EDGE_VOICE, pitch=EDGE_PITCH, rate=EDGE_RATE)
            agen = comm.stream()
            while True:
                try:
                    ch = await asyncio.wait_for(agen.__anext__(), timeout=EDGE_TIMEOUT_S)
                except StopAsyncIteration:
                    break
                if ch.get("type") != "audio":
                    continue
                buf += ch["data"]
                need = _STREAM_FIRST_BYTES if not sent else _STREAM_NEXT_BYTES
                if len(buf) - last_len < need:
                    continue
                last_len = len(buf)
                pcm = _mp3_to_pcm24k(buf)
                new = pcm[emitted:]; emitted = len(pcm)
                if len(new):
                    out = rs.resample_chunk(new)
                    if len(out):
                        if not sent:
                            logger.info(f"[tts-edge] streaming: first audio {time.time()-t0:.2f}s")
                        sent = True
                        yield TTSAudioRawFrame(out.tobytes(), TTS_SAMPLE_RATE, 1, context_id=context_id)
            pcm = _mp3_to_pcm24k(buf)
            tail = rs.resample_chunk(pcm[emitted:], last=True)
            if len(tail):
                sent = True
                yield TTSAudioRawFrame(tail.tobytes(), TTS_SAMPLE_RATE, 1, context_id=context_id)
            logger.info(f"[tts-edge] streamed {len(text)} chars in {time.time()-t0:.2f}s")
        except Exception as e:
            self._edge_ok = False
            self._edge_fail_ts = time.time()
            logger.warning(f"[tts-edge] streaming failed ({e!r})"
                           + ("; qwen3 fallback" if not sent else "; reply cut short"))
            if not sent:
                async for f in Qwen3TTSService.run_tts(self, text, context_id):
                    yield f


class BootsTTSService(StreamingEdgeTTSService):
    """The ME3 TTS used by run_bot(): edge-tts (Roger) primary, Qwen3 (vivian)
    fallback, every reply leads with LEAD_SILENCE_S. The welcome calls
    _synthesize() on the same instance, so the welcome voice matches the
    replies exactly. Select backend via ME3_TTS_BACKEND (edge|qwen3|auto)."""
    pass
