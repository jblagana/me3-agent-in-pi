#
# ME3 -- Deepgram speech services (2026-11-07).
#
#   ME3_STT=deepgram -> DeepgramBootsSTT: Nova-3 STREAMING recognition. Audio is
#       sent while the user talks; on VAD stop Pipecat asks Deepgram to
#       finalize, so the transcript is ready almost at once.
#   ME3_TTS=deepgram -> DeepgramBootsTTS: Aura-2 voice (default "aries") over
#       ONE kept-open websocket (same warm-up / barge-in handling as the edge
#       fix). Any Deepgram failure before audio -> the edge-tts Roger path.
#
# Key: DEEPGRAM_API_KEY (in /etc/me3/groq.env, loaded by agent.py).
#
import asyncio
import json
import os
import re
import time

import numpy as np
from loguru import logger

from pipecat.frames.frames import (InterimTranscriptionFrame, TranscriptionFrame,
                                   TTSAudioRawFrame, VADUserStartedSpeakingFrame,
                                   VADUserStoppedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection

import remote_voice as rv

DG_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
DG_STT_MODEL = os.environ.get("ME3_DG_STT_MODEL", "nova-3")
DG_STT_LANGUAGE = os.environ.get("ME3_DG_STT_LANGUAGE", "en")
DG_VOICE = os.environ.get("ME3_DG_VOICE", "aura-2-aries-en")
DG_SPEED = float(os.environ.get("ME3_DG_SPEED", "1.0"))
DG_IDLE_REOPEN_S = float(os.environ.get("ME3_DG_IDLE_REOPEN_S", "25"))
SR = rv.TTS_SAMPLE_RATE  # 48 kHz, the pipeline output rate


# =========================================================================== #
# Speech-to-text
# =========================================================================== #
def make_stt():
    from pipecat.services.deepgram.stt import DeepgramSTTService, DeepgramSTTSettings

    class DeepgramBootsSTT(DeepgramSTTService):
        """Deepgram streaming STT + Boots' hooks, fired ONCE per user turn
        (Deepgram may send several finals per utterance)."""

        def __init__(self):
            super().__init__(
                api_key=DG_KEY, sample_rate=16000,
                settings=DeepgramSTTSettings(model=DG_STT_MODEL, language=DG_STT_LANGUAGE,
                                             smart_format=True, punctuate=True,
                                             interim_results=True))
            self._turn_text = []
            self._stopped = False
            self._fired = False
            self._fire_task = None
            logger.info(f"[asr-deepgram] {DG_STT_MODEL} ({DG_STT_LANGUAGE}) streaming")

        async def process_frame(self, frame, direction):
            if isinstance(frame, VADUserStartedSpeakingFrame):
                self._turn_text, self._stopped, self._fired = [], False, False
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                self._stopped = True
                # if every final already arrived before the stop, fire shortly
                self._fire_task = asyncio.ensure_future(self._fire_later(0.1))
            await super().process_frame(frame, direction)

        async def _keepalive_handler(self):
            """Keep-alive + WATCHDOG (2026-11-07). Pipecat only reconnects when
            the socket closes cleanly; seen live: the socket went half-dead,
            every keep-alive failed and Boots stayed deaf for minutes. Two
            failures in a row (10 s) now force a reconnect."""
            from pipecat.services.deepgram import stt as _dgstt
            fails = 0
            while True:
                await asyncio.sleep(5)
                if not self._connection:
                    fails = 0
                    continue
                try:
                    await self._connection.send_keep_alive(_dgstt.ListenV1KeepAlive(type="KeepAlive"))
                    fails = 0
                except Exception as e:
                    fails += 1
                    logger.warning(f"[asr-deepgram] keep-alive failed ({fails}): {e!r}")
                    if fails >= 2:
                        fails = 0
                        logger.warning("[asr-deepgram] connection is dead: forcing a reconnect")
                        try:
                            await self._reconnect()
                            logger.info("[asr-deepgram] reconnected")
                        except Exception as e2:
                            logger.error(f"[asr-deepgram] reconnect failed: {e2!r}")

        async def _fire_later(self, delay):
            await asyncio.sleep(delay)
            self._fire()

        def _fire(self):
            if self._fired:
                return
            text = " ".join(t for t in self._turn_text if t).strip()
            if not text:
                return
            self._fired = True
            logger.info(f"[asr-deepgram] transcript: {text!r}")
            try:
                rv.Qwen2AudioSTTService._push_heard_to_ui(self, text)
            except Exception:
                pass
            if rv._fdb_mark is not None:
                rv._fdb_mark("asr_final", text=text)
            if rv.ON_TRANSCRIPT is not None:
                try:
                    rv.ON_TRANSCRIPT(text)
                except Exception as e:
                    logger.warning(f"[asr-deepgram] ON_TRANSCRIPT hook failed: {e!r}")

        async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
            if direction == FrameDirection.DOWNSTREAM and isinstance(
                    frame, (TranscriptionFrame, InterimTranscriptionFrame)):
                if isinstance(frame, TranscriptionFrame):
                    t = rv._clean_transcript(frame.text or "")
                    if not re.search(r"[A-Za-z0-9]", t):
                        return                          # word-less noise: not a turn
                    frame.text = t
                    self._turn_text.append(t)
                    if self._stopped:
                        frame.finalized = True   # turn over: let the LLM start now
                    if self._stopped:
                        if self._fire_task:
                            self._fire_task.cancel()
                        self._fire()
                # Same delivery as the Groq path: straight to the user aggregator
                # (a push from a background task isn't reliably processed in 1.12).
                try:
                    nxt = getattr(self, "_next", None)
                    for _ in range(3):
                        if nxt is None:
                            break
                        if type(nxt).__name__ == "LLMUserAggregator":
                            await nxt.queue_frame(frame, FrameDirection.DOWNSTREAM)
                            return
                        nxt = getattr(nxt, "_next", None)
                except Exception as e:
                    logger.debug(f"[asr-deepgram] aggregator re-deliver failed: {e!r}")
            await super().push_frame(frame, direction)

    return DeepgramBootsSTT()


# =========================================================================== #
# Text-to-speech
# =========================================================================== #
class _DGLink:
    """One kept-open Deepgram TTS websocket (Speak + Flush per sentence)."""

    def __init__(self):
        self.session = None
        self.ws = None
        self.last = 0.0
        self.busy = False
        self.opening = None

    def _stale(self):
        return (self.ws is None or self.ws.closed or self.busy
                or time.monotonic() - self.last > DG_IDLE_REOPEN_S)

    async def _close(self):
        ws, self.ws, self.busy = self.ws, None, False
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def _open(self):
        import aiohttp
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        t0 = time.monotonic()
        q = f"model={DG_VOICE}&encoding=linear16&sample_rate={SR}"
        if DG_SPEED and DG_SPEED != 1.0:
            q += f"&speed={DG_SPEED}"
        ws = await self.session.ws_connect(f"wss://api.deepgram.com/v1/speak?{q}",
                                           headers={"Authorization": f"Token {DG_KEY}"},
                                           heartbeat=None)
        old, self.ws, self.last, self.busy = self.ws, ws, time.monotonic(), False
        if old is not None and not old.closed:
            asyncio.get_running_loop().call_later(10, lambda: asyncio.ensure_future(old.close()))
        logger.info(f"[tts-deepgram] connection opened ({time.monotonic()-t0:.2f}s) {DG_VOICE}")

    def prewarm(self):
        if (self.opening and not self.opening.done()) or not self._stale():
            return
        try:
            self.opening = asyncio.get_running_loop().create_task(self._open_quiet())
        except RuntimeError:
            pass

    async def _open_quiet(self):
        try:
            await self._open()
        except Exception as e:
            logger.debug(f"[tts-deepgram] pre-warm failed: {e!r}")

    async def stream(self, text):
        """Yield raw int16 PCM chunks at SR for one sentence."""
        import aiohttp
        if self.opening and not self.opening.done():
            try:
                await asyncio.wait_for(asyncio.shield(self.opening), timeout=8)
            except Exception:
                pass
        if self.ws is None or self.ws.closed or self.busy:
            await self._open()
        msg = json.dumps({"type": "Speak", "text": text})
        try:
            await self.ws.send_str(msg)
        except Exception:
            await self._open()
            await self.ws.send_str(msg)
        await self.ws.send_str(json.dumps({"type": "Flush"}))
        ws = self.ws
        self.busy = True
        async for m in ws:
            if m.type == aiohttp.WSMsgType.BINARY:
                if m.data:
                    yield m.data
            elif m.type == aiohttp.WSMsgType.TEXT:
                try:
                    t = json.loads(m.data).get("type")
                except Exception:
                    t = None
                if t == "Flushed":
                    if self.ws is ws:
                        self.busy = False
                        self.last = time.monotonic()
                    return
                if t in ("Error", "Warning"):
                    logger.warning(f"[tts-deepgram] {m.data[:200]}")
                    if t == "Error":
                        break
            else:
                break
        if self.ws is ws:
            await self._close()
        raise RuntimeError("Deepgram TTS socket closed mid-sentence")


_DG = _DGLink()


class DeepgramBootsTTS(rv.BootsTTSService):
    """Aura-2 voice first; the inherited edge-tts Roger path (and its Qwen3
    fallback) when Deepgram fails before any audio."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info(f"[tts-deepgram] {DG_VOICE} primary (speed {DG_SPEED}), edge-tts Roger fallback")

    async def run_tts(self, text, context_id):
        spoken = rv._speakable((text or "").strip())
        if not spoken:
            return
        t0 = time.time()
        sent = False
        try:
            if rv.REPLY_LEAD_S > 0:
                yield TTSAudioRawFrame(b"\x00\x00" * int(SR * rv.REPLY_LEAD_S), SR, 1, context_id=context_id)
            pending = b""
            agen = _DG.stream(spoken)
            while True:
                try:
                    chunk = await asyncio.wait_for(agen.__anext__(), timeout=rv.EDGE_TIMEOUT_S)
                except StopAsyncIteration:
                    break
                pending += chunk
                if len(pending) < 2:
                    continue
                n = len(pending) // 2 * 2
                if not sent:
                    logger.info(f"[tts-deepgram] streaming: first audio {time.time()-t0:.2f}s")
                sent = True
                yield TTSAudioRawFrame(pending[:n], SR, 1, context_id=context_id)
                pending = pending[n:]
            logger.info(f"[tts-deepgram] streamed {len(spoken)} chars in {time.time()-t0:.2f}s")
        except Exception as e:
            logger.warning(f"[tts-deepgram] failed ({e!r})"
                           + ("; edge-tts fallback" if not sent else "; reply cut short"))
            if not sent:
                async for f in super().run_tts(text, context_id):
                    yield f

    def _synthesize(self, text, sample_rate=16000):
        """Welcome path: one-shot REST synthesis at `sample_rate`."""
        import requests
        try:
            r = requests.post(
                f"https://api.deepgram.com/v1/speak?model={DG_VOICE}&encoding=linear16"
                f"&sample_rate={sample_rate}&container=none"
                + (f"&speed={DG_SPEED}" if DG_SPEED and DG_SPEED != 1.0 else ""),
                headers={"Authorization": f"Token {DG_KEY}", "Content-Type": "application/json"},
                json={"text": text}, timeout=15)
            if r.status_code == 200 and len(r.content) > 1000:
                return r.content
            logger.warning(f"[tts-deepgram] welcome HTTP {r.status_code}; edge-tts fallback")
        except Exception as e:
            logger.warning(f"[tts-deepgram] welcome failed ({e!r}); edge-tts fallback")
        return super()._synthesize(text, sample_rate)


def prewarm_tts():
    _DG.prewarm()
