import os
import sys
import time

from loguru import logger


def _load_llm_env():
    try:
        with open("/etc/me3/llm.env") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except Exception as e:
        logger.warning(f"could not load /etc/me3/llm.env: {e!r}")

_load_llm_env()

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://202.92.159.240:8001/v1")
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "not-needed")
LLM_MODEL    = os.environ.get("LLM_MODEL", "Qwen/Qwen3.8-27B")
ASR_BASE_URL = os.environ.get("ME3_ASR_BASE_URL", "http://127.0.0.1:8080/v1")
ME3_ASR_MODEL = os.environ.get("ME3_ASR_MODEL", "whisper-1")
def _find_mic_index():
    """Auto-detect the USB mic (EMEET M1A) by name; fallback to first input device."""
    import pyaudio
    p = pyaudio.PyAudio()
    inputs = []
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d.get("maxInputChannels", 0) > 0:
            inputs.append((i, d["name"]))
            name = d["name"].lower()
            if "emeet" in name or "m1a" in name or "officecore" in name:
                return i
    return inputs[0][0] if inputs else None




MIC_DEV = _find_mic_index()
if MIC_DEV is None:
    logger.error("NO MICROPHONE FOUND (no input devices). Plug in the EMEET M1A and re-run.")
    raise SystemExit(2)
SPK_DEV = int(os.environ.get("ME3_SPK_DEV", MIC_DEV))  # EMEET is mic+speaker
if SPK_DEV == 0 and MIC_DEV != 0:
    SPK_DEV = MIC_DEV  # never fall back to the dead jack when the EMEET is present
logger.info(f"microphone device index: {MIC_DEV}, speaker: {SPK_DEV}")

logger.info("ME3 starting...")
import pyaudio

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    LLMRunFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    FunctionCallsStartedFrame,
    TTSStartedFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
    TTSAudioRawFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair, LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transports.local.audio import (
    LocalAudioInputTransport, LocalAudioOutputTransport, LocalAudioTransportParams,
)
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.processors.frame_processor import FrameProcessor

# make sibling modules importable regardless of how agent.py is launched
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import me2_ui_mcp as ui
import web_tools as web
import spotify_tools as sp
from piper_tts import PiperLocalTTSService
from asr import WhisperCPPSTTService
from remote_voice import (Qwen2AudioSTTService, Qwen3TTSService,
                        BootsTTSService, LEAD_SILENCE_S)
from prompt import SYSTEM_PROMPT

logger.info("components imported")


# --------------------------------------------------------------------------- #
# Live status indicator â€” prints a single updating line to the terminal so you
# can see what ME3 is doing (listening / transcribing / thinking / tool /
# speaking / idle). It also logs what it HEARD (transcript) and what it SAID
# (LLM reply) so you are never blind.
# --------------------------------------------------------------------------- #
_USE_ANSI = sys.stdout.isatty() and os.environ.get("ME3_NO_STATUS", "0") != "1"
_LAST_STATUS = {"phase": None}
_RECENT_TEXT = {"transcript": None, "reply": None}

# Boots UI live conversation channel. The agent POSTs its phase + heard +
# thinking + response to the Boots UI server (default 127.0.0.1:8330, env
# ME3_BOOTSUI) so the dashboard can show what Boots heard, is thinking, and
# its reply. Fire-and-forget on a daemon thread so it never blocks the pipeline
# and never crashes the agent if the server is down.
_BOOTSUI = os.environ.get("ME3_BOOTSUI", "http://127.0.0.1:8330")


def _me3_push(phase: str, label: str = "") -> None:
    """Best-effort POST of the current conversation state to the Boots UI."""
    import json as _json
    import threading as _th
    import urllib.request as _ur

    body = {
        "phase": phase,
        "heard": (_RECENT_TEXT.get("transcript") or "")[:300],
        "thinking": (label or "")[:300],
        "response": (_RECENT_TEXT.get("reply") or "")[:400],
        "tool": (label or "")[:120] if phase == "tool" else "",
        "active": phase != "idle",
    }

    def _go():
        try:
            req = _ur.Request(
                _BOOTSUI + "/me3",
                data=_json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with _ur.urlopen(req, timeout=1.0) as _r:
                _r.read()
        except Exception:
            pass  # fail-soft: dashboard just won't update; agent unaffected

    _th.Thread(target=_go, daemon=True).start()


def _me3_heartbeat():
    """Daemon thread: push a liveness heartbeat every 2s so the UI can show
    LISTENING (agent alive) vs OFF (agent down). phase mirrors the live status."""
    import json as _j, threading as _t, urllib.request as _u, time as _tm
    def _loop():
        while True:
            try:
                body = {"hb": True, "active": True,
                        "phase": _LAST_STATUS.get("phase", "listening")}
                req = _u.Request(_BOOTSUI + "/me3", data=_j.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
                with _u.urlopen(req, timeout=1.0) as r:
                    r.read()
            except Exception:
                pass
            _tm.sleep(2.0)
    _t.Thread(target=_loop, daemon=True, name="me3-heartbeat").start()


def _print_status(label: str, detail: str = ""):
    """Overwrite a single status line (ANSI tty) or log a new line (non-tty)."""
    if _USE_ANSI:
        sys.stdout.write("\r\x1b[2K")
        text = f"  \U0001F7E2 {label:<14}"
        if detail:
            text += f"  {detail}"
        sys.stdout.write(text.ljust(90))
        sys.stdout.flush()
    else:
        logger.info(f"[status] {label}" + (f"  {detail}" if detail else ""))


class StatusIndicator(FrameProcessor):
    """Pipeline tap that reports ME3's current phase + transcripts + replies,
    AND gates the mic while ME3 is speaking (no AEC on the EMEET M1A, so the
    mic would otherwise pick up the bot's own voice -> endless barge-in loop)."""

    def __init__(self, state=None, **kwargs):
        super().__init__(name="StatusIndicator", **kwargs)
        self._state = state if state is not None else {"bot_speaking": False}
        self._idle_timer = None
        self._reply_buf = []
        self._thinking_buf = []
        self._mic_active = False
        self._tool_t0 = None
        # time-based cooldown: after ME3 stops, keep the mic closed this many
        # seconds to absorb the speaker's echo tail before re-opening.
        self._rearm_at = 0.0
        self._COOLDOWN_SECS = 0.35

    def _rearm(self):
        """Re-open the mic once the cooldown has elapsed."""
        if self._state.get("bot_speaking") and time.monotonic() >= self._rearm_at:
            self._state["bot_speaking"] = False

    def _set(self, phase: str, label: str, detail: str = ""):
        _LAST_STATUS["phase"] = phase
        _print_status(label, detail)
        try:
            _me3_push(phase, label)
        except Exception:
            pass  # never let a dashboard push break the pipeline

    def _clear_idle_timer(self):
        if self._idle_timer is not None:
            try:
                self._idle_timer.cancel()
            except Exception:
                pass
            self._idle_timer = None

    def _schedule_idle(self, delay: float = 3.0):
        self._clear_idle_timer()
        try:
            import asyncio
            loop = asyncio.get_running_loop()
        except Exception:
            return

        def _go():
            self._rearm()
            if not self._state.get("bot_speaking") and _LAST_STATUS.get("phase") in (
                    "speaking", "thinking", "tool", "transcribing", "listening"):
                # FIX (issue 6): Boots is ALWAYS listening when idle — never show
                # "Idle". The default resting state is "listening".
                self._set("listening", "Listening... (ready)")
        self._idle_timer = loop.call_later(delay, _go)

    async def process_frame(self, frame, direction):
        self._rearm()
        if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            _RECENT_TEXT["reply"] = None
            self._reply_buf = []
            self._thinking_buf = []
            self._mic_active = True
            self._clear_idle_timer()
            self._set("listening", "Listening... (you're speaking)")
        elif isinstance(frame, (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            self._mic_active = False
            self._clear_idle_timer()
            self._schedule_idle(2.5)
        elif isinstance(frame, InterimTranscriptionFrame):
            text = (frame.text or "").strip()
            if text:
                _RECENT_TEXT["transcript"] = text
                self._me3_push_now()
            self._set("listening", "Heard (interim): " + text[:60])
        elif isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            _RECENT_TEXT["transcript"] = text
            _RECENT_TEXT["reply"] = None
            self._reply_buf = []
            self._thinking_buf = []
            self._me3_push_now()
            self._set("thinking", "Thinking... (LLM generating reply)")
            if text:
                logger.info(f"[me3] you said: {text}")
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._state["bot_speaking"] = True
            self._clear_idle_timer()
            self._reply_buf = []
            self._thinking_buf = []
            self._set("thinking", "Thinking... (LLM generating reply)")
        elif isinstance(frame, LLMTextFrame):
            self._reply_buf.append(frame.text or "")
            reply = "".join(self._reply_buf).strip()
            _RECENT_TEXT["reply"] = reply
            self._thinking_buf = reply
            self._me3_push_now()
        elif isinstance(frame, FunctionCallsStartedFrame):
            names = [fc.function_name for fc in (frame.function_calls or [])]
            tool_name = ", ".join(names)
            _RECENT_TEXT["tool"] = tool_name
            self._tool_t0 = time.monotonic()
            self._set("tool", "Running tool: " + tool_name)
            if names:
                logger.info(f"[me3] calling tool(s): {names}")
        elif isinstance(frame, TTSStartedFrame):
            self._state["bot_speaking"] = True
            self._clear_idle_timer()
            self._set("speaking", "Speaking...")
        elif isinstance(frame, TTSAudioRawFrame):
            if not self._state.get("bot_speaking"):
                self._state["bot_speaking"] = True
                self._set("speaking", "Speaking...")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if getattr(self, "_tool_t0", None):
                dt = time.monotonic() - self._tool_t0
                logger.info(f"[me3] tool+LLM phase took {dt:.1f}s")
                self._tool_t0 = None
            self._rearm_at = time.monotonic() + self._COOLDOWN_SECS
            self._schedule_idle(2.5)
        elif isinstance(frame, InterruptionFrame):
            self._clear_idle_timer()
            self._mic_active = True
            self._set("listening", "Barge-in: you interrupted me")
        await self.push_frame(frame)

    def _me3_push_now(self):
        """Push current heard/thinking/response/tool/mic to the Boots UI."""
        try:
            import json as _j, threading as _t, urllib.request as _u
            body = {"phase": _LAST_STATUS.get("phase", "listening"),
                    "heard": (_RECENT_TEXT.get("transcript") or "")[:300],
                    "thinking": ("".join(self._thinking_buf) or "")[:400],
                    "response": (_RECENT_TEXT.get("reply") or "")[:400],
                    "tool": (_RECENT_TEXT.get("tool") or "")[:120],
                    "mic": bool(self._mic_active), "active": True}
            def _go():
                try:
                    req = _u.Request(_BOOTSUI + "/me3", data=_j.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
                    with _u.urlopen(req, timeout=1.0) as r:
                        r.read()
                except Exception:
                    pass
            _t.Thread(target=_go, daemon=True).start()
        except Exception:
            pass


class InputGate(FrameProcessor):
    """Drops incoming audio while ME3 is speaking so the no-AEC mic cannot hear
    the bot's own voice (otherwise the VAD self-triggers into a barge-in loop).
    Shares the same state dict as StatusIndicator."""

    def __init__(self, state=None, **kwargs):
        super().__init__(name="InputGate", **kwargs)
        self._state = state if state is not None else {"bot_speaking": False}
        self._last_mic_push = 0.0

    async def process_frame(self, frame, direction):
        from pipecat.frames.frames import InputAudioRawFrame
        if isinstance(frame, InputAudioRawFrame):
            # FIX: compute a live mic level (RMS) on the RAW audio *before* the
            # gate mutes it, so the UI level meter reflects what the mic hears.
            self._update_mic_level(frame.audio)
            if self._state.get("bot_speaking"):
                # replace audio with silence so the VAD/ASR see no speech from the bot
                n = len(frame.audio)
                frame.audio = b"\x00\x00" * (n // 2)
        await self.push_frame(frame)

    def _update_mic_level(self, audio: bytes):
        """RMS of the int16 mono chunk -> 0..1, stored in shared state + throttled
        push to the UI so the mic meter moves in real time."""
        import struct
        if not audio or len(audio) < 2:
            return
        n = len(audio) // 2
        # sample down for speed (every 4th sample is plenty for a meter)
        total = 0
        cnt = 0
        for i in range(0, n, 4):
            v = struct.unpack_from("<h", audio, i * 2)[0]
            total += v * v
            cnt += 1
        if cnt == 0:
            return
        rms = (total / cnt) ** 0.5          # 0..32767
        level = max(0.0, min(1.0, rms / 8000.0))  # ~-14dBFS => 1.0
        self._state["mic_level"] = level
        import time as _t
        now = _t.monotonic()
        if now - getattr(self, "_last_mic_push", 0.0) >= 0.08:   # ~12 pushes/s
            self._last_mic_push = now
            try:
                self._push_mic(level)
            except Exception:
                pass

    def _push_mic(self, level: float):
        import json as _j, threading as _t, urllib.request as _u
        body = {"mic_level": round(level, 3)}
        def _go():
            try:
                req = _u.Request(_BOOTSUI + "/me3mic", data=_j.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
                with _u.urlopen(req, timeout=0.4) as r:
                    r.read()
            except Exception:
                pass
        _t.Thread(target=_go, daemon=True).start()


async def _spotify_handler(params):
    """Route a Spotify function call to spotify_tools.dispatch(). ASYNC (pipecat 1.12 awaits the handler)."""
    import asyncio as _aio, functools as _ft
    args = dict(params.arguments or {})
    return await _aio.to_thread(_ft.partial(sp.dispatch, params.function_name), **args)


async def _web_handler(params):
    """Route a web tool call to web_tools.dispatch(). ASYNC (pipecat 1.12 awaits
    the handler). Offloaded to a thread so network calls never block the event
    loop. Returns a string the LLM can speak back to the user."""
    import asyncio
    tool = params.get("name") if isinstance(params, dict) else None
    args = params.get("arguments", {}) if isinstance(params, dict) else {}
    if not tool:
        return "I couldn't run that web action."
    return await asyncio.to_thread(web.dispatch, tool, **args)


async def _ui_handler(params):
    """Route a Boots UI function call to ui.dispatch() -> Boots UI server :8330. ASYNC (pipecat 1.12 awaits the handler)."""
    import asyncio as _aio, functools as _ft
    args = dict(params.arguments or {})
    return await _aio.to_thread(_ft.partial(ui.dispatch, params.function_name), **args)


def _function_schemas():
    """Build pipecat FunctionSchema tools for ME2 UI + Spotify.

    Each schema carries a `handler`, so the LLM service auto-registers it â€” no
    separate register_function call is needed (pipecat 1.12).
    """
    out = []
    for (n, d, p) in ui.TOOL_SPECS:
        out.append(FunctionSchema(n, d, dict(p), list(p.keys()), handler=_ui_handler))
    for (n, d, p) in sp.TOOL_SPECS:
        out.append(FunctionSchema(n, d, dict(p), list(p.keys()), handler=_spotify_handler))
    for (n, d, p) in web.TOOL_SPECS:
        out.append(FunctionSchema(n, d, dict(p), list(p.keys()), handler=_web_handler))
    return out


async def _speak_welcome(tts: "PiperLocalTTSService", spk_dev: int, state: dict):
    """Deterministic spoken welcome on boot.

    Synthesizes the greeting with the Piper service and plays it DIRECTLY
    through a PyAudio output stream on the EMEET (spk_dev). This bypasses
    the pipeline and uses the same proven path as the aplay test. (A
    TTSSpeakFrame queued at the task root is dropped by the LLM stage
    before reaching the TTS stage in this topology.)
    """
    text = "Hi there. I am Boots."
    state["bot_speaking"] = True  # gate the mic while we speak (no self-feedback)
    logger.info(f"[me3] speaking welcome: {text!r}")
    # UNIFIED VOICE: synthesize at the SAME rate the pipeline uses (48 kHz),
    # via the same _synthesize() path as responses. This removes the old
    # double-resample (24k->16k->48k) that made the welcome sound different
    # from Boots' normal reply voice. _synthesize() handles the DGX3 24 kHz
    # native -> 48 kHz resample internally (soxr).
    out_sr = 48000
    try:
        import numpy as _np
        raw48 = await asyncio.to_thread(tts._synthesize, text, out_sr)  # int16 mono @48k
        if not raw48:
            logger.warning("[me3] welcome: empty synth; skipping playback")
            state["bot_speaking"] = False
            return
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=out_sr,
                            output=True, output_device_index=spk_dev)
            try:
                # EMEET M1A USB first-chunk loss: lead ~0.45 s of silence so
                # the head of the speech is not dropped by the fresh stream.
                stream.write(_np.zeros(int(out_sr * LEAD_SILENCE_S), dtype=_np.int16).tobytes())
                chunk = 4000
                for i in range(0, len(raw48), chunk):
                    stream.write(raw48[i:i + chunk])
            finally:
                stream.stop_stream()
                stream.close()
        finally:
            p.terminate()
        logger.info("[me3] welcome playback finished")
    except Exception as e:
        logger.error(f"[me3] welcome failed: {e!r}")
    finally:
        state["bot_speaking"] = False

async def run_bot():
    pa = pyaudio.PyAudio()
    # Voice I/O: remote (DGX3 Qwen2-Audio ASR / Qwen3-TTS) vs local
    # (whisper.cpp / Piper). ME3_VOICE=remote (default) offloads the heavy
    # neural legs to the DGX3 GPUs via the SHARC UP VPN; local is the
    # offline fallback. VAD stays local either way.
    _voice = os.environ.get("ME3_VOICE", "remote")
    if _voice == "remote":
        stt = Qwen2AudioSTTService()
    else:
        stt = WhisperCPPSTTService()
    llm = OpenAILLMService(
        api_key=LLM_API_KEY, model=LLM_MODEL, base_url=LLM_BASE_URL,
        settings=OpenAILLMService.Settings(system_instruction=SYSTEM_PROMPT, temperature=0.4, extra={'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}),
    )
    if _voice == "remote":
        tts = BootsTTSService()  # edge-tts (Roger) primary, qwen3 fallback
    else:
        tts = PiperLocalTTSService()
    logger.info("ME3 voice mode: %s (%s)"
                % (_voice, "DGX3 remote" if _voice == "remote" else "local"))
    # VAD tuned for this environment: higher confidence + volume so ambient
    # room noise / mic self-trigger don't cause endless barge-in loops.
    # NOTE: pass params via the constructor; pipecat calls set_sample_rate()
    # on the first audio frame, which re-applies these params (calling
    # set_params here directly would ZeroDivisionError: sample_rate is 0).
    from pipecat.audio.vad.vad_analyzer import VADParams
    vad = SileroVADAnalyzer(params=VADParams(
        confidence=0.90, start_secs=0.2, stop_secs=0.3, min_volume=0.85))
    gate_state = {"bot_speaking": False, "mic_level": 0.0}
    input_gate = InputGate(state=gate_state)
    status = StatusIndicator(state=gate_state)
    _me3_heartbeat()   # FIX: liveness heartbeat (UI: LISTENING vs OFF)

    context = LLMContext(tools=_function_schemas())
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=vad))

    in_params = LocalAudioTransportParams(audio_in_enabled=True, audio_in_sample_rate=16000,
                                          audio_in_channels=1, input_device_index=MIC_DEV)
    out_params = LocalAudioTransportParams(audio_out_enabled=True, audio_out_sample_rate=48000,
                                           audio_out_channels=1, output_device_index=SPK_DEV)
    in_transport = LocalAudioInputTransport(pa, in_params)
    out_transport = LocalAudioOutputTransport(pa, out_params)

    # NOTE: `status` sits after the output transport so it sees the full frame
    # stream (speech, transcripts, LLM, TTS) without touching the audio path.
    pipeline = Pipeline([
        in_transport, input_gate, stt, user_aggregator, llm, tts, out_transport,
        status, assistant_aggregator,
    ])
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True, allow_interruptions=True),
                        cancel_on_idle_timeout=False, idle_timeout_secs=3600)
    task.add_observer(MetricsLogObserver())  # per-turn TTFB/TTFAT/TTFA/TURN/reasoning -> log

    # (greeting: deterministic spoken welcome now runs just before the pipeline starts)

    if _USE_ANSI:
        status._set("listening", "Ready. Say something (or wait for the greeting).")
        sys.stdout.write("\n")
        sys.stdout.flush()
    runner = PipelineRunner(handle_sigint=True)
    # Speak the welcome BEFORE the pipeline starts so it can open the
    # EMEET output device cleanly. Once runner.run() begins, the output
    # transport holds SPK_DEV and a second PyAudio stream on it fails
    # with "Device unavailable".
    await _speak_welcome(tts, SPK_DEV, gate_state)
    await runner.run(task)
    if _USE_ANSI:
        sys.stdout.write("\r\x1b[2K")
        sys.stdout.flush()
    pa.terminate()


async def main():
    logger.info("starting local audio (mic #%s -> speaker #%s)", MIC_DEV, SPK_DEV)
    await run_bot()


if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nME3 stopped.")