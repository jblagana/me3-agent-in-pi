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


def _load_groq_env():
    # Groq provider config (ASR + LLM). Loaded AFTER llm.env and applied with a
    # direct set (NOT setdefault) so Groq WINS over the legacy llm.env vLLM
    # defaults -- Groq is the active provider. The values below are then read
    # from os.environ, which this function has populated with the Groq config.
    try:
        with open("/etc/me3/groq.env") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip()
    except Exception as e:
        logger.warning(f"could not load /etc/me3/groq.env: {e!r}")


_load_groq_env()

# LLM: default to Groq (OpenAI-compatible, Qwen3.8-27B hosted on Groq -- same
# model the local vLLM ran, so behaviour is unchanged). Override via
# LLM_BASE_URL/LLM_API_KEY/LLM_MODEL (e.g. /etc/me3/groq.env or llm.env).
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")
LLM_MODEL    = os.environ.get("LLM_MODEL", "qwen/qwen3.8-27b")
# True when the LLM is the local Qwen vLLM (needs enable_thinking off). Groq and
# other OpenAI-compatible servers do NOT accept that extra_body, so gate on it.
LLM_USE_VLLM_THINKING = os.environ.get("LLM_USE_VLLM_THINKING", "0") == "1"
ASR_BASE_URL = os.environ.get("ME3_ASR_BASE_URL", "http://127.0.0.1:8080/v1")
ME3_ASR_MODEL = os.environ.get("ME3_ASR_MODEL", "whisper-large-v3-turbo")
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
    LLMFullResponseEndFrame,
    LLMTextFrame,
    FunctionCallsStartedFrame,
    FunctionCallResultFrame,
    TTSStartedFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
    TTSAudioRawFrame,
    BotStartedSpeakingFrame,
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
import me2_tools as ui
import web_tools as web
import spotify_tools as sp
import files_tools as ft
import shell_tools as sh
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
_RECENT_TEXT = {"transcript": None, "reply": None, "tool": None, "tool_args": None,
                "thinking": None, "tool_result": None}
# Monotonic LLM round counter (2026-11-06): bumped at every
# LLMFullResponseStartFrame. Pushed to the UI as cot_round so the CoT box
# gets ONE think row per LLM round (a tool turn has 2+ rounds: plan, then
# post-tool deliberation) — the 1 Hz poll can't detect round boundaries
# itself (the agent clears 'thinking' at round end, so a sub-1s round is
# invisible to it). Global counter (never reset): the UI keys on CHANGE,
# and it resets its own marker at each turn boundary.
_LLM_ROUND = {"n": 0}
# Per-leg voice turn latencies (ms) — 2026-11-06. The UI's LATENCY column was
# dead for voice turns (only the typed /chat path ever set it). The agent
# stamps these on the Pi's own clock: asr = TranscriptionFrame -> LLM round
# start, llm = LLM round start -> first spoken text token (TTFT), tool =
# FunctionCallsStarted -> FunctionCallResult, tts = LLM end -> BotStarted
# Speaking, total = heard -> first audio out. Pushed as m3.latency on the
# BotStartedSpeaking push (the server merges it into /ui_state).
_VOICE_LAT = {"asr": None, "llm": None, "tool": None, "tts": None, "total": None}
# Turn-boundary timestamps (monotonic) shared between StatusIndicator (which
# sees the audio/ASR/TTS frames) and the reply tap (which sees the LLM
# frames). Set by _voice_turn_start() / _voice_llm_start() / _voice_tts_end().
_VOICE_T = {"heard": None, "llm_start": None, "tts_done": None,
            "first_audio": None, "first_round": None}
# Tool-call wall clock: set by StatusIndicator on FunctionCallsStartedFrame,
# read by the reply tap on FunctionCallResultFrame (they're different
# processors, so a shared global is the clean handoff).
_TOOL_T0 = {"t": None}
# Per-round flag: did the reasoning stream deliver real thinking tokens THIS
# LLM round? When yes, the LLMTextFrame handler must NOT overwrite
# _RECENT_TEXT['thinking'] with the reply text (that was the old CoT bug:
# the box mirrored the answer, not the deliberation). Reset at round start.
_REASONING_SEEN = {"round": 0}


def _voice_turn_start():
    """A new utterance was transcribed — reset the per-turn latency stamps."""
    _VOICE_T["heard"] = time.monotonic()
    # ASR leg = VAD speech-stop -> final transcript (the real ASR time).
    # speech_end is stamped by StatusIndicator (it sees the VAD frames; the
    # reply tap does not).
    if _VOICE_T.get("speech_end"):
        _VOICE_LAT["asr"] = int((_VOICE_T["heard"] - _VOICE_T["speech_end"]) * 1000)
    _VOICE_T["llm_start"] = None
    _VOICE_T["tts_done"] = None
    _VOICE_T["first_audio"] = None
    _VOICE_T["first_round"] = None
    _VOICE_LAT.update({"asr": None, "llm": None, "tool": None, "tts": None, "total": None})


def _voice_llm_start():
    """First LLM round of the turn began (LLMFullResponseStartFrame)."""
    _VOICE_T["llm_start"] = time.monotonic()
    if _VOICE_T["first_round"] is None:
        _VOICE_T["first_round"] = _VOICE_T["llm_start"]


def _voice_tts_end():
    """LLM finished its final round — start the TTS leg clock."""
    _VOICE_T["tts_done"] = time.monotonic()

# Human-readable "Boots is doing X" phrases for tool calls. Groq's Qwen often
# responds to a command with a pure function call (NO text tokens), so without
# this the UI's THINKING bubble is blank and no assistant bubble appears. We
# synthesize a short phrase from the tool the LLM chose to run.
_TOOL_PHRASE = {
    "boots_play_music": "Playing music",
    "boots_pause_music": "Pausing music",
    "boots_resume_music": "Resuming music",
    "boots_stop_music": "Stopping music",
    "boots_next_song": "Skipping to the next song",
    "boots_set_volume": "Adjusting the volume",
    "boots_lights": "Toggling the lights",
    "boots_set_bright": "Setting the brightness",
    "boots_set_color": "Setting the light color",
    "boots_set_temp": "Adjusting the temperature",
    "boots_set_timer": "Setting a timer",
    "boots_set_alarm": "Setting an alarm",
    "boots_get_time": "Checking the time",
    "boots_get_weather": "Fetching the weather",
    "boots_make_call": "Placing a call",
    "boots_send_message": "Sending a message",
    "boots_create_reminder": "Setting a reminder",
    "boots_list_reminders": "Listing reminders",
    "spotify_play_search": "Searching Spotify and playing",
    "spotify_pause": "Pausing Spotify",
    "spotify_resume": "Resuming Spotify",
    "spotify_next": "Skipping a track",
    "spotify_previous": "Going back a track",
    "spotify_now_playing": "Checking what's playing",
    "spotify_search": "Searching Spotify",
    "web_search": "Searching the web",
    "fetch_url": "Opening a page",
    "write_website": "Building a website",
}


def _tool_phrase(tool_name: str) -> str:
    """Short human phrase for a tool call (used to fill the thinking bubble)."""
    if not tool_name:
        return ""
    base = _TOOL_PHRASE.get(tool_name, tool_name)
    args = _RECENT_TEXT.get("tool_args")
    return f"{base}{(' — ' + args) if args else ''}…"


# Boots UI live conversation channel. The agent POSTs its phase + heard +
# thinking + response to the Boots UI server (default 127.0.0.1:8330, env
# ME3_BOOTSUI) so the dashboard can show what Boots heard, is thinking, and
# its reply. Fire-and-forget on a daemon thread so it never blocks the pipeline
# and never crashes the agent if the server is down.
_BOOTSUI = os.environ.get("ME3_BOOTSUI", "http://127.0.0.1:8330")


def _voice_state() -> dict:
    """The VOICE LLM identity (GroqResilientLLM's provider/model), for the UI's
    provider-aware sub-labels. The UI's chat LLM (s.llm) is a DIFFERENT engine
    (boots_ui.py's /chat) — the sub-labels must name the voice one, not the
    chat one, and not a hardcoded 'qwen3.8-27b · vllm' (2026-11-06)."""
    try:
        base = (LLM_BASE_URL or "").lower()
        prov = "groq" if "groq.com" in base else ("vllm" if "202.92" in base or "8001" in base else "llm")
        model = LLM_MODEL or "llm"
        return {"provider": prov, "model": model.split("/")[-1]}
    except Exception:
        return {"provider": "llm", "model": "llm"}


def _me3_push(phase: str, label: str = "") -> None:
    """Best-effort POST of the current conversation state to the Boots UI."""
    import json as _json
    import threading as _th
    import urllib.request as _ur

    # 'thinking' carries ONLY the real reasoning/tool phrase (_RECENT_TEXT
    # ['thinking']), set when the LLM streams text or fires a tool. We do NOT
    # fall back to the status `label` (e.g. "Listening...", "Barge-in...") --
    # those are UI status strings and would make the THINKING bubble flicker
    # with non-reasoning text. Empty here => the UI clears the bubble.
    thinking = _RECENT_TEXT.get("thinking") or ""
    body = {
        "phase": phase,
        "heard": (_RECENT_TEXT.get("transcript") or "")[:300],
        "thinking": thinking[:400],
        "response": (_RECENT_TEXT.get("reply") or "")[:400],
        "tool": (_RECENT_TEXT.get("tool") or "")[:120],
        "tool_args": (_RECENT_TEXT.get("tool_args") or "")[:400],
        "cot_round": _LLM_ROUND["n"],
        "tool_result": (_RECENT_TEXT.get("tool_result") or "")[:200],
        "voice": _voice_state(),
        "active": phase != "idle",
    }

    def _go():
        try:
            if body.get("heard") or body.get("thinking") or body.get("response"):
                logger.info(f"[me3push] {phase!r} heard={body.get('heard')[:40]!r} resp={body.get('response')[:40]!r}")
            req = _ur.Request(
                _BOOTSUI + "/me3",
                data=_json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with _ur.urlopen(req, timeout=1.0) as _r:
                _r.read()
        except Exception as _e:
            logger.debug(f"[me3push] failed: {_e}")

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
        self._tool_t0 = None  # (legacy; tool latency now uses the shared _TOOL_T0)
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
            _RECENT_TEXT["tool"] = None
            _RECENT_TEXT["tool_args"] = None
            _RECENT_TEXT["thinking"] = None
            _RECENT_TEXT["tool_result"] = None
            self._reply_buf = []
            self._thinking_buf = []
            self._mic_active = True
            self._clear_idle_timer()
            self._set("listening", "Listening... (you're speaking)")
        elif isinstance(frame, (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            self._mic_active = False
            self._clear_idle_timer()
            self._schedule_idle(2.5)
            _VOICE_T["speech_end"] = time.monotonic()  # ASR leg clock start
        elif isinstance(frame, InterimTranscriptionFrame):
            text = (frame.text or "").strip()
            if text:
                _RECENT_TEXT["transcript"] = text
                self._me3_push_now()
            self._set("listening", "Heard (interim): " + text[:60])
        elif isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            _voice_turn_start()  # new utterance: reset the per-turn latency legs
            _RECENT_TEXT["transcript"] = text
            _RECENT_TEXT["reply"] = None
            _RECENT_TEXT["tool"] = None
            _RECENT_TEXT["tool_args"] = None
            _RECENT_TEXT["thinking"] = None
            _RECENT_TEXT["tool_result"] = None
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
            # NOTE: the LLM-round counter (_LLM_ROUND) is bumped in
            # _install_reply_tap's observer (the reliable hook for every LLM
            # round in pipecat 1.12) — NOT here, or the round would count
            # double (both StatusIndicator and the tap see the frame).
            self._set("thinking", "Thinking... (LLM generating reply)")
        # NOTE: LLMTextFrame / LLMFullResponseEndFrame / FunctionCallResultFrame
        # are handled by _install_reply_tap's on_after_push_frame event handler on
        # the LLM (pipecat 1.12 does not deliver those frames up to StatusIndicator,
        # which sits before the LLM).
        elif isinstance(frame, FunctionCallsStartedFrame):
            fcs = list(frame.function_calls or [])
            names = [fc.function_name for fc in fcs]
            tool_name = ", ".join(names)
            _RECENT_TEXT["tool"] = tool_name
            # Capture the tool ARGUMENTS so the UI can show what the LLM asked the
            # tool to do (e.g. spotify_play_search(query="...")). Groq/Qwen return
            # a parsed args dict per call; fall back to the raw text if odd.
            try:
                if len(fcs) == 1:
                    _RECENT_TEXT["tool_args"] = (
                        ", ".join(f"{k}={v!r}" for k, v in (fcs[0].arguments or {}).items())
                        or None)
                else:
                    _RECENT_TEXT["tool_args"] = (
                        "; ".join(f"{fc.function_name}({fc.arguments})" for fc in fcs) or None)
            except Exception:
                _RECENT_TEXT["tool_args"] = None
            self._tool_t0 = time.monotonic()
            _TOOL_T0["t"] = self._tool_t0  # shared clock for the reply tap's tool leg
            # Fill the thinking bubble with a human phrase for the tool the LLM
            # chose to run (Groq's Qwen often emits a pure function call with NO
            # text tokens, which otherwise leaves the bubble blank). Use the first
            # tool for the phrase; if the LLM also streamed some text this turn,
            # that text already lives in _thinking_buf and takes precedence.
            primary = names[0] if names else ""
            if not (self._thinking_buf and "".join(self._thinking_buf).strip()):
                _RECENT_TEXT["thinking"] = _tool_phrase(primary)
            # Seed the reply with the tool phrase too, so the assistant bubble has
            # something to show if the LLM never streams a spoken reply for this
            # command turn (it often doesn't for pure actions).
            if not (_RECENT_TEXT.get("reply") or "").strip():
                _RECENT_TEXT["reply"] = _tool_phrase(primary)
            self._me3_push_now()   # push immediately so the tool + args hit the UI
            self._set("tool", "Running tool: " + tool_name)
            if names:
                logger.info(f"[me3] calling tool(s): {names} args={_RECENT_TEXT.get('tool_args')!r}")
        # NOTE: FunctionCallResultFrame is handled by _install_reply_tap's event
        # handler on the LLM (see above).
        elif isinstance(frame, TTSStartedFrame):
            self._state["bot_speaking"] = True
            self._clear_idle_timer()
            self._set("speaking", "Speaking...")
        elif isinstance(frame, (BotStartedSpeakingFrame, TTSAudioRawFrame)):
            if not self._state.get("bot_speaking"):
                self._state["bot_speaking"] = True
                self._set("speaking", "Speaking...")
            # First real audio out (2026-11-06): stamp the TTS leg (LLM end ->
            # first audio) + the total (heard -> first audio), then push the
            # latency dict so the UI's LATENCY column shows the voice turn's
            # real numbers (it was dead for voice turns — typed chat only).
            if self._state.get("bot_speaking") and _VOICE_T["tts_done"] \
                    and _VOICE_LAT["tts"] is None:
                _VOICE_LAT["tts"] = int((time.monotonic() - _VOICE_T["tts_done"]) * 1000)
                # total = user finished speaking (transcript) -> first audio
                # out of the speaker (the E2E number for the deck).
                if _VOICE_T["heard"]:
                    _VOICE_LAT["total"] = int((time.monotonic() - _VOICE_T["heard"]) * 1000)
                try:
                    import json as _j, threading as _t, urllib.request as _u
                    lat = {k: v for k, v in _VOICE_LAT.items() if isinstance(v, (int, float))}
                    body = {"latency": lat, "active": True,
                            "phase": _LAST_STATUS.get("phase", "speaking")}
                    def _lat_go():
                        try:
                            req = _u.Request(_BOOTSUI + "/me3", data=_j.dumps(body).encode("utf-8"),
                                             headers={"Content-Type": "application/json"}, method="POST")
                            with _u.urlopen(req, timeout=1.0) as r:
                                r.read()
                        except Exception:
                            pass
                    _t.Thread(target=_lat_go, daemon=True).start()
                except Exception:
                    pass
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
        # FIX (ASR stall root cause): forward with the ORIGINAL direction. The
        # aggregator's VAD broadcasts VADUserStoppedSpeakingFrame upstream
        # (direction=UPSTREAM) so it can reach the STT, which sits upstream of
        # the aggregator in the pipeline. The previous code called
        # push_frame(frame) with no direction -> default DOWNSTREAM, which
        # reflected those upstream VAD frames back toward the LLM and starved
        # the SegmentedSTTService of its speech-stop trigger (run_stt never
        # fired -> no transcript -> no "heard" bubble in the UI).
        await self.push_frame(frame, direction)

    def _me3_push_now(self):
        """Push current heard/thinking/response/tool/mic to the Boots UI."""
        try:
            import json as _j, threading as _t, urllib.request as _u
            # 'thinking' = streamed LLM text if any, else the tool phrase (so the
            # bubble is never blank on pure function-call turns).
            thinking = ("".join(self._thinking_buf) or _RECENT_TEXT.get("thinking") or "")
            body = {"phase": _LAST_STATUS.get("phase", "listening"),
                    "heard": (_RECENT_TEXT.get("transcript") or "")[:300],
                    "thinking": thinking[:400],
                    "response": (_RECENT_TEXT.get("reply") or "")[:400],
                    "tool": (_RECENT_TEXT.get("tool") or "")[:120],
                    "tool_args": (_RECENT_TEXT.get("tool_args") or "")[:400],
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


def _me3_push_state(phase: str, thinking: str, reply: str) -> None:
    """Push the current reply/thinking/tool to the Boots UI (fire-and-forget)."""
    try:
        import json as _j, threading as _t, urllib.request as _u
        body = {"phase": phase,
                "heard": (_RECENT_TEXT.get("transcript") or "")[:300],
                "thinking": (thinking or "")[:400],
                "response": (reply or "")[:400],
                "tool": (_RECENT_TEXT.get("tool") or "")[:120],
                "tool_args": (_RECENT_TEXT.get("tool_args") or "")[:400],
                "cot_round": _LLM_ROUND["n"],
                "tool_result": (_RECENT_TEXT.get("tool_result") or "")[:200],
                "voice": _voice_state(),
                "latency": {k: v for k, v in _VOICE_LAT.items() if isinstance(v, (int, float))} or None,
                "mic": False, "active": True}
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


def _install_reply_tap(llm):
    """Attach a pure OBSERVER on the LLM service (via pipecat's event_handler)
    to capture the frames the upstream StatusIndicator never sees in pipecat 1.12:

      * LLMTextFrame            -> the LLM's streamed spoken reply tokens
      * LLMFullResponseStart/End-> reply turn boundaries
      * FunctionCallResultFrame -> a tool finished (surface its result)

    WHY an event handler on the LLM and not a downstream FrameProcessor:
    (1) a ReplyTap FrameProcessor inserted between the LLM and TTS silently
    BROKE the LLM->TTS frame flow in pipecat 1.12 (TTS was never invoked, the
    tap saw no frames). (2) The LLM's `on_after_push_frame` event IS the
    reliable hook -- it fires for every frame the LLM pushes, including
    LLMTextFrame (verified: it streams token-by-token). It does NOT sit in the
    frame path, so TTS keeps working while we observe the reply tokens for the
    UI bubble. (A TTS-side observer was tried but TTS only receives the text as
    AggregatedTextFrame, not LLMTextFrame/TextFrame, so the LLM is the right
    place.) (D21/D22: tool phrase + streaming reply bubble.)
    """
    reply_buf = []

    @llm.event_handler("on_after_push_frame")
    async def _on_llm_push(processor, frame):
        # NOTE on phase: the dashboard's reply bubble (liveReply) only renders
        # when phase=="speaking" (me3_muji_ui_mockup.html: `if(ph==='speaking'
        # && m3.response){ liveReply(m3.response); }`). So the reply text must be
        # pushed with phase="speaking" to appear in the response bubble. The
        # streaming tokens also go to the thinking bubble (liveThinking renders
        # m3.response when phase=="thinking"). We push the reply with
        # phase="speaking" so it lands in the response bubble; StatusIndicator
        # independently drives the real speaking/listening phase transitions.
        try:
            if isinstance(frame, LLMFullResponseStartFrame):
                reply_buf.clear()
                _RECENT_TEXT["reply"] = ""
                _RECENT_TEXT["thinking"] = ""  # start a fresh reply; stop tool phrase
                # New LLM round (2026-11-06): bumped HERE (the tap sees every
                # LLM round in pipecat 1.12, unlike StatusIndicator). The UI's
                # CoT box keys its think rows on this number — one row per
                # round (plan, post-tool, ...), and a sub-1s round is still
                # counted because the number is what the poll sees, not the
                # (cleared) thinking text.
                _LLM_ROUND["n"] += 1
                _voice_llm_start()   # ASR leg = heard -> this moment (see _VOICE_LAT)
                _me3_push_state(_LAST_STATUS.get("phase", "thinking"), "", "")
            elif isinstance(frame, LLMTextFrame):
                reply_buf.append(frame.text or "")
                reply = "".join(reply_buf).strip()
                if reply:
                    # LLM leg = round start -> first spoken text token (TTFT),
                    # stamped ONCE per round (2026-11-06: the UI's LATENCY
                    # column was dead for voice turns).
                    if _VOICE_LAT["llm"] is None and _VOICE_T["first_round"]:
                        _VOICE_LAT["llm"] = int((time.monotonic() - _VOICE_T["first_round"]) * 1000)
                    _RECENT_TEXT["reply"] = reply
                    # CoT shows the REAL thinking, not the reply (2026-11-06):
                    # when the reasoning stream delivered tokens this round,
                    # _RECENT_TEXT['thinking'] already holds the model's
                    # deliberation — do NOT overwrite it with the reply text.
                    # The reply-mirror fallback stays for rounds WITHOUT
                    # reasoning (pure tool-call turns, reasoning_effort=none).
                    if _REASONING_SEEN["round"] != _LLM_ROUND["n"]:
                        _RECENT_TEXT["thinking"] = reply
                    # Push with the REAL phase (2026-11-06 dup-bubble fix):
                    # forcing "speaking" per token made the UI see
                    # thinking->speaking->thinking... per token, and the server
                    # appends a history turn on EVERY speaking->non-speaking
                    # transition -> N identical assistant bubbles. The UI
                    # renders the reply bubble whenever phase==speaking OR the
                    # turn just finished (its own liveReply gate), so the real
                    # phase is safe.
                    _me3_push_state(_LAST_STATUS.get("phase", "thinking"), reply, reply)
            elif isinstance(frame, LLMFullResponseEndFrame):
                # Finalize: keep whatever was streamed. Fall back to the seeded
                # tool phrase ONLY when the LLM produced no text (a pure
                # function-call turn) — the old `or _RECENT_TEXT["reply"]`
                # could clobber a real reply with the tool phrase if a
                # FunctionCallResultFrame raced in.
                reply = "".join(reply_buf).strip()
                if not reply:
                    reply = _RECENT_TEXT.get("reply") or ""
                _RECENT_TEXT["reply"] = reply
                _RECENT_TEXT["thinking"] = ""  # done; stop the thinking spinner text
                _voice_tts_end()  # TTS leg clock starts now (LLM finished)
                # Push phase="speaking" at LLM end (2026-11-06): the server
                # appends a history turn on the speaking->non-speaking
                # transition, and with the real-phase text pushes above the
                # phase never reached "speaking" before the turn ended
                # (TTSStartedFrame's push raced the 2.5s idle timer) -> voice
                # turns never landed in chat history. The LLM end IS the turn
                # end for UI purposes; TTSStartedFrame pushes "speaking" again
                # (no-op) and BotStoppedSpeaking's idle push performs the
                # speaking->listening transition that finalizes the append.
                _me3_push_state("speaking", "", reply)
            elif isinstance(frame, FunctionCallResultFrame):
                result = str(getattr(frame, "result", "") or "").strip()
                fname = getattr(frame, "function_name", "") or ""
                if result and result not in ("IN_PROGRESS", "OK"):
                    _RECENT_TEXT["reply"] = result
                elif fname:
                    _RECENT_TEXT["reply"] = _tool_phrase(fname)
                _RECENT_TEXT["thinking"] = ""
                # Dedicated tool_result field (2026-11-06): the UI flips the
                # voice tool chip green + settles the card on THIS. It can't
                # sniff 'response' — the final reply can be IDENTICAL to the
                # tool result ("Lights are on, boo." is both).
                if result and result not in ("IN_PROGRESS", "OK"):
                    _RECENT_TEXT["tool_result"] = result
                # Tool leg = FunctionCallsStarted -> this result (shared clock:
                # StatusIndicator sets _TOOL_T0, we read it here).
                if _TOOL_T0["t"]:
                    _VOICE_LAT["tool"] = int((time.monotonic() - _TOOL_T0["t"]) * 1000)
                    _TOOL_T0["t"] = None
                _me3_push_state(_LAST_STATUS.get("phase", "tool"), "",
                                _RECENT_TEXT.get("reply") or "")
                logger.info(f"[me3] tool result {fname!r}: {result[:80]!r}")
        except Exception as e:
            logger.warning(f"[me3] reply-tap observer error: {e!r}")


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
        # FIX (ASR stall root cause): MUST forward the frame with its ORIGINAL
        # `direction`. The upstream VAD stop-frames (VADUserStoppedSpeakingFrame)
        # that the aggregator's VAD broadcasts carry direction=UPSTREAM; pushing
        # them with the default (DOWNSTREAM) reflected them back toward the LLM
        # instead of letting them continue upstream to the STT, so the
        # SegmentedSTTService never saw a speech-stop and run_stt never fired.
        await self.push_frame(frame, direction)

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



# ---------------------------------------------------------------------------
# MCP bridge (2026-11-06): Boots runs its tools through the MCP server
# (mcp_server.py) over stdio -- the same MCP path an external client uses.
# The 5 handlers below call _MCP.call() first; if the bridge is down they
# fall back to direct dispatch() so voice never dies on an MCP hiccup.
# ---------------------------------------------------------------------------
class _MCPBridge:
    def __init__(self):
        self._session = None
        self.available = False
        self._stack = None
        self.calls_via_mcp = 0
        self.calls_fallback = 0

    async def start(self):
        try:
            from contextlib import AsyncExitStack
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            here = os.path.dirname(os.path.abspath(__file__))
            params = StdioServerParameters(
                command=os.path.join(here, ".venv", "bin", "python"),
                args=[os.path.join(here, "mcp_server.py")],
                cwd=here)
            self._stack = AsyncExitStack()
            r, w = await self._stack.enter_async_context(stdio_client(params))
            self._session = await self._stack.enter_async_context(ClientSession(r, w))
            await self._session.initialize()
            tools = await self._session.list_tools()
            self.available = True
            logger.info("[mcp] bridge up: %d tools via mcp_server.py (stdio)"
                        % len(tools.tools))
        except Exception as e:
            self.available = False
            logger.error("[mcp] bridge failed to start: %r "
                         "(handlers fall back to direct dispatch)" % (e,))

    async def call(self, tool, args):
        """Return the tool result string via MCP, or None to signal fallback."""
        if not self.available or self._session is None:
            return None
        try:
            res = await asyncio.wait_for(
                self._session.call_tool(tool, dict(args or {})), timeout=25)
            text = "".join(getattr(c, "text", "") or "" for c in res.content)
            self.calls_via_mcp += 1
            logger.info("[mcp] ran %s via bridge (%d total)" % (tool, self.calls_via_mcp))
            return text or "(tool returned nothing)"
        except Exception as e:
            logger.warning("[mcp] call %s failed: %r" % (tool, e))
            return None

    async def close(self):
        try:
            if self._stack is not None:
                await self._stack.aclose()
        except Exception:
            pass

_MCP = _MCPBridge()


async def _spotify_handler(params):
    """Route a Spotify function call to spotify_tools.dispatch().

    2026-11-06 toolfix: pipecat 1.12 handlers must report via
    params.result_callback(result) -- returning a value is ignored, so the
    FunctionCallResultFrame never fired, the LLM re-prompt never ran, and
    voice tool turns hung on the in-flight spinner. Callback + return kept
    for older pipecat compatibility."""
    import asyncio as _aio, functools as _ft
    args = dict(params.arguments or {})
    result = await _MCP.call(params.function_name, args)
    if result is None:
        # bridge down -- direct dispatch fallback
        result = await _aio.to_thread(_ft.partial(sp.dispatch, params.function_name), **args)
        _MCP.calls_fallback += 1
    try:
        if getattr(params, "result_callback", None) is not None:
            await params.result_callback(result)
    except Exception:
        pass
    return result


async def _web_handler(params):
    """Route a web tool call to web_tools.dispatch(). ASYNC (pipecat 1.12 awaits
    the handler). Offloaded to a thread so network calls never block the event
    loop. Returns a string the LLM can speak back to the user.

    `params` is pipecat 1.12's FunctionCallParams dataclass -- it exposes
    .function_name and .arguments (NOT a dict). The old dict-style
    params.get("name") raised AttributeError, so check_weather (and every web
    tool) crashed before running and the LLM got no result to speak."""
    import asyncio
    try:
        tool = getattr(params, "function_name", None) or (params.get("name") if isinstance(params, dict) else None)
        args = getattr(params, "arguments", None)
        if args is None and isinstance(params, dict):
            args = params.get("arguments", {})
        args = dict(args or {})
    except Exception:
        args = {}
        tool = None
    if not tool:
        result = "I couldn't run that web action."
    else:
        result = await _MCP.call(tool, args)
        if result is None:
            # bridge down -- direct dispatch fallback
            result = await asyncio.to_thread(web.dispatch, tool, **args)
            _MCP.calls_fallback += 1
    # 2026-11-06 toolfix: report via result_callback (see _spotify_handler).
    try:
        if getattr(params, "result_callback", None) is not None:
            await params.result_callback(result)
    except Exception:
        pass
    return result


async def _ui_handler(params):
    """Route a Boots UI function call to ui.dispatch() -> Boots UI server :8330.

    2026-11-06 toolfix: report via params.result_callback(result) (see
    _spotify_handler) so the tool result reaches the LLM and the UI."""
    import asyncio as _aio, functools as _ft
    args = dict(params.arguments or {})
    result = await _MCP.call(params.function_name, args)
    if result is None:
        # bridge down -- direct dispatch fallback
        result = await _aio.to_thread(_ft.partial(ui.dispatch, params.function_name), **args)
        _MCP.calls_fallback += 1
    try:
        if getattr(params, "result_callback", None) is not None:
            await params.result_callback(result)
    except Exception:
        pass
    return result


async def _files_handler(params):
    """Route a file tool call to files_tools.dispatch().

    Same result_callback contract as the other handlers (2026-11-06 toolfix)
    so the tool result reaches the LLM and the UI."""
    import asyncio as _aio, functools as _ft
    args = dict(params.arguments or {})
    result = await _MCP.call(params.function_name, args)
    if result is None:
        # bridge down -- direct dispatch fallback
        result = await _aio.to_thread(_ft.partial(ft.dispatch, params.function_name), **args)
        _MCP.calls_fallback += 1
    try:
        if getattr(params, "result_callback", None) is not None:
            await params.result_callback(result)
    except Exception:
        pass
    return result


async def _shell_handler(params):
    """Route a shell tool call to shell_tools.dispatch().

    Same result_callback contract as the other handlers (2026-11-06 toolfix)
    so the tool result reaches the LLM and the UI."""
    import asyncio as _aio, functools as _ft
    args = dict(params.arguments or {})
    result = await _MCP.call(params.function_name, args)
    if result is None:
        # bridge down -- direct dispatch fallback
        result = await _aio.to_thread(_ft.partial(sh.dispatch, params.function_name), **args)
        _MCP.calls_fallback += 1
    try:
        if getattr(params, "result_callback", None) is not None:
            await params.result_callback(result)
    except Exception:
        pass
    return result


def _normalize_props(p):
    """Return a valid JSON-schema `properties` dict for a tool's param dict.

    TOOL_SPECS use a shorthand ("query": "string") that is NOT a valid JSON
    schema (a property must be an object like {"type": "string"} or a boolean).
    The local Qwen vLLM tolerated the shorthand, but Groq validates the tool
    schema strictly and rejects it with HTTP 400 ("got string, want boolean or
    object"). Normalise every property to {"type": <t>} so Groq accepts it.
    Properties that are already dicts / booleans pass through unchanged.
    """
    norm = {}
    for k, v in (p or {}).items():
        if isinstance(v, str):            # shorthand "string" -> {"type": "string"}
            norm[k] = {"type": v}
        elif isinstance(v, (dict, bool)):  # already a valid schema fragment
            norm[k] = v
        else:                             # defensive fallback
            norm[k] = {"type": "string"}
    return norm


def _function_schemas():
    """Build pipecat FunctionSchema tools for ME2 UI + Spotify.

    Each schema carries a `handler`, so the LLM service auto-registers it â€” no
    separate register_function call is needed (pipecat 1.12).
    """
    out = []
    for (n, d, p) in ui.TOOL_SPECS:
        out.append(FunctionSchema(n, d, _normalize_props(p), list(p.keys()), handler=_ui_handler))
    for (n, d, p) in sp.TOOL_SPECS:
        out.append(FunctionSchema(n, d, _normalize_props(p), list(p.keys()), handler=_spotify_handler))
    for (n, d, p) in web.TOOL_SPECS:
        out.append(FunctionSchema(n, d, _normalize_props(p), list(p.keys()), handler=_web_handler))
    for (n, d, p) in ft.TOOL_SPECS:
        out.append(FunctionSchema(n, d, _normalize_props(p), list(p.keys()), handler=_files_handler))
    for (n, d, p) in sh.TOOL_SPECS:
        out.append(FunctionSchema(n, d, _normalize_props(p), list(p.keys()), handler=_shell_handler))
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
    # GREETING-UI: surface the spoken greeting as an assistant bubble in the
    # Boots UI (voice turns share the UI chat thread). Set the reply text so
    # _me3_push carries it, then push phase=speaking so the response bubble
    # renders; the finally-block idle push finalizes the turn into history.
    _RECENT_TEXT["reply"] = text
    _me3_push("speaking")
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
        # GREETING-UI: phase leaves speaking with the reply present -> the
        # dashboard appends the greeting to chat history as a real turn.
        try:
            _me3_push("idle")
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Groq-resilient LLM (D23): 429 retry + context trimming.
#
# Groq's free tier rate-limits (429 Too Many Requests) under barge-in-heavy
# turns, and pipecat's OpenAILLMService.get_chat_completions() only retries
# timeouts -- a 429 just raises and the turn gets NO reply. We wrap the LLM
# request in a bounded retry with exponential backoff on 429/rate-limit, and
# trim the conversation context to the most recent N turns so each request
# stays light (less per-request token cost = fewer 429s).
# --------------------------------------------------------------------------- #
def _inject_current_time(context) -> None:
    """Stamp the current date/time into the LLM context's system message on
    EVERY LLM call, so the model knows "now". The system instruction is set
    once at boot (SYSTEM_PROMPT); we keep the original text in
    _BASE_SYSTEM_TEXT and re-append a fresh "Current date/time: ..." line each
    call. This runs before _trim_llm_context so the (updated) system message
    is always the one that reaches the model.

    Why here (not in SYSTEM_PROMPT): a static prompt bakes in the boot-time
    clock and goes stale; re-stamping per call keeps it current for free.
    """
    global _BASE_SYSTEM_TEXT
    try:
        import time as _t
        now = _t.localtime()
        stamp = _t.strftime("%A, %B %d, %Y %H:%M %Z", now)
        line = f"\n\nCurrent date and time: {stamp} (the user's local time is Asia/Manila, UTC+8)."
        if _BASE_SYSTEM_TEXT is None:
            _BASE_SYSTEM_TEXT = SYSTEM_PROMPT  # capture the boot-time base once
        base = _BASE_SYSTEM_TEXT
        target = base + line
        # Replace the existing system/developer message content in place.
        for i, m in enumerate(context.messages):
            role = (m.get("role") if isinstance(m, dict)
                    else getattr(m, "role", "")) or ""
            if role in ("system", "developer"):
                content = (m.get("content") if isinstance(m, dict)
                           else getattr(m, "content", "")) or ""
                # Strip any previously-appended stamp so we don't stack them.
                if base in content:
                    content = content[:content.index(base)]
                new_content = content.rstrip() + line
                if isinstance(m, dict):
                    m["content"] = new_content
                else:
                    try:
                        m.content = new_content
                    except Exception:
                        pass
                break
        else:
            # No system message present (shouldn't happen) -- append one.
            context.messages.insert(0, {"role": "system", "content": target})
    except Exception as e:
        logger.debug(f"[me3] time stamp skipped: {e!r}")


_BASE_SYSTEM_TEXT: str | None = None  # captured on first call (see above)


def _trim_llm_context(context, max_messages: int) -> None:
    """Drop the oldest conversation messages, keeping the system prompt + the
    most recent turns. Mutates `context.messages` in place. The system message
    (role='system'/'developer') is always kept; everything else is capped to
    the last `max_messages`. Keeps user/assistant pairing intact where
    possible by trimming to an even boundary."""
    try:
        msgs = context.messages
        if not msgs:
            return
        # Identify the non-system slice.
        head = []
        i = 0
        while i < len(msgs):
            role = (msgs[i].get("role") if isinstance(msgs[i], dict)
                    else getattr(msgs[i], "role", "")) or ""
            if role in ("system", "developer"):
                head.append(msgs[i]); i += 1
            else:
                break
        tail = msgs[i:]
        if len(tail) > max_messages:
            # Drop from the oldest non-system message, keeping the newest.
            drop = len(tail) - max_messages
            # Prefer to start the kept window on a user message so we don't
            # begin with a dangling assistant/tool message.
            while drop < len(tail) and (tail[drop].get("role") if isinstance(tail[drop], dict)
                                        else getattr(tail[drop], "role", "")) in ("assistant", "tool"):
                drop += 1
            tail = tail[drop:]
        # LLMContext.messages is a read-only property -- mutate the list IN PLACE
        # (clear + extend) rather than reassigning, which raises AttributeError.
        new_msgs = head + tail
        try:
            msgs[:] = new_msgs
        except Exception:
            context.messages = new_msgs  # fallback if a setter exists
    except Exception as e:
        logger.debug(f"[me3] context trim skipped: {e!r}")


async def _tap_reasoning_stream(stream):
    """Async-generator wrapper over the LLM's HTTP chunk stream: forwards
    every chunk UNCHANGED, and tees `delta.reasoning` (Groq Qwen3.8's real
    thinking tokens, present when reasoning_effort is set) into
    _RECENT_TEXT['thinking'] + a throttled UI push. pipecat's base LLM only
    pushes LLMTextFrame for delta.content — the reasoning field was being
    dropped, so the voice CoT box could only mirror the reply text.

    This is a STREAM wrapper, not a FrameProcessor: it never sits in the
    pipecat frame path (a FrameProcessor between LLM and TTS broke TTS in
    1.12 — D21), so the LLM->TTS flow is byte-identical. The generator must
    be consumed exactly once (pipecat's _closing() handles aclose)."""
    buf = []
    last_push = 0.0
    try:
        async for chunk in stream:
            try:
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    if delta is not None:
                        # Groq puts reasoning on delta.reasoning (verified
                        # live 2026-11-06); keep the openai convention too.
                        r = getattr(delta, "reasoning", None) or \
                            getattr(delta, "reasoning_content", None)
                        if r:
                            buf.append(r)
                            _RECENT_TEXT["thinking"] = "".join(buf)[:400]
                            _REASONING_SEEN["round"] = _LLM_ROUND["n"]
                            now = time.monotonic()
                            if now - last_push > 0.35:   # ~3 Hz push cap
                                last_push = now
                                _me3_push_state(_LAST_STATUS.get("phase", "thinking"),
                                                _RECENT_TEXT["thinking"],
                                                _RECENT_TEXT.get("reply") or "")
            except Exception as e:
                logger.debug(f"[me3] reasoning-tap chunk error: {e!r}")
            yield chunk
    finally:
        # Round end: keep the accumulated reasoning in _RECENT_TEXT so the
        # next LLMFullResponseStart's clear-push is the ONLY thing that wipes
        # it (the UI's m3.cot server-side keeps the row visible regardless).
        pass


class GroqResilientLLM(OpenAILLMService):
    """OpenAILLMService with Groq 429 retry + context trimming (D23).

    Overrides `get_chat_completions` (the single request point pipecat 1.12
    uses for the LLM turn) to: (1) trim the context to the last N messages,
    and (2) retry on rate-limit/429 with exponential backoff (bounded). All
    other behaviour (streaming, tool calls, adapters) is inherited."""

    # Tunables (env-overridable).
    MAX_CONTEXT_MESSAGES = int(os.environ.get("ME3_LLM_MAX_MSGS", "24"))
    MAX_RETRIES = int(os.environ.get("ME3_LLM_MAX_RETRIES", "3"))
    RETRY_BASE_DELAY = float(os.environ.get("ME3_LLM_RETRY_DELAY", "1.0"))

    async def get_chat_completions(self, context):
        # 0) Stamp the CURRENT date/time into the system prompt on EVERY call,
        #    so the model always knows "now" (it otherwise has no clock and
        #    can't answer "what time is it?" or reason about relative times).
        _inject_current_time(context)
        # 0b) REAL CoT (2026-11-06): ask Groq for reasoning tokens. Without
        #     this the Qwen3.8 stream carries NO reasoning field (verified
        #     live: reasoning=None), so the UI's CoT box could only mirror
        #     the reply text. 'medium' balances token burn against the 1000-
        #     OTPM tier; override via ME3_REASONING_EFFORT ('none' = off).
        #     vLLM (local Qwen) uses enable_thinking instead — never send
        #     reasoning_effort there.
        if not LLM_USE_VLLM_THINKING:
            _eff = os.environ.get("ME3_REASONING_EFFORT", "medium").strip().lower()
            if _eff in ("minimal", "low", "medium", "high"):
                try:
                    self._settings.extra["reasoning_effort"] = _eff
                except Exception:
                    pass
        # 1) Keep the request light: trim to the most recent N messages.
        if self.MAX_CONTEXT_MESSAGES > 0:
            _trim_llm_context(context, self.MAX_CONTEXT_MESSAGES)
        # 2) Retry on 429 / rate-limit with exponential backoff.
        import asyncio as _aio
        from openai import APIStatusError, RateLimitError, APIConnectionError, APITimeoutError
        attempt = 0
        while True:
            attempt += 1
            try:
                stream = await super().get_chat_completions(context)
                # 3) REAL CoT (2026-11-06): tee the stream's reasoning deltas
                #    into the UI. pipecat only pushes LLMTextFrame for
                #    delta.content — delta.reasoning (Groq's Qwen3.8 carries
                #    it when reasoning_effort is set) is dropped on the floor,
                #    which is why the voice CoT box could only mirror the
                #    reply text. This wrapper sits on the HTTP chunk stream
                #    (NOT the frame path — a FrameProcessor in the LLM->TTS
                #    path broke TTS, D21), so TTS keeps working while the
                #    reasoning tokens stream to the CoT box live.
                return _tap_reasoning_stream(stream)
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                # 429 (RateLimitError) / transient network / timeout.
                retryable = (
                    isinstance(e, (RateLimitError, APIConnectionError, APITimeoutError))
                    or (isinstance(e, APIStatusError) and e.status_code in (408, 429, 500, 502, 503, 504))
                )
                if not retryable or attempt > self.MAX_RETRIES:
                    raise
                delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))  # 1s, 2s, 4s
                logger.warning(f"[me3] LLM {type(e).__name__} (attempt {attempt}/{self.MAX_RETRIES}); "
                               f"retrying in {delay:.1f}s")
                await _aio.sleep(delay)
            except APIStatusError as e:
                # Other HTTP errors (4xx except 429, 5xx handled above).
                if e.status_code in (500, 502, 503, 504) and attempt <= self.MAX_RETRIES:
                    delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(f"[me3] LLM HTTP {e.status_code} (attempt {attempt}/{self.MAX_RETRIES}); "
                                   f"retrying in {delay:.1f}s")
                    await _aio.sleep(delay)
                else:
                    raise

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
    # Build the LLM. The vLLM-specific enable_thinking extra_body is ONLY sent
    # for the local Qwen vLLM (Groq / other OpenAI-compatible servers reject it).
    llm_settings = dict(system_instruction=SYSTEM_PROMPT, temperature=0.4)
    if LLM_USE_VLLM_THINKING:
        llm_settings["extra"] = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    # GroqResilientLLM (D23): 429 retry + context trimming on top of the stock
    # OpenAILLMService. (Confirmed: the missing-TTS regression was the ReplyTap
    # FrameProcessor in the LLM->TTS path, NOT this subclass -- TTS works with
    # either LLM as long as ReplyTap is gone.)
    llm = GroqResilientLLM(
        api_key=LLM_API_KEY, model=LLM_MODEL, base_url=LLM_BASE_URL,
        settings=OpenAILLMService.Settings(**llm_settings),
    )
    logger.info(f"LLM: {type(llm).__name__} model={LLM_MODEL} "
                f"(ctx cap {llm.MAX_CONTEXT_MESSAGES} msgs, "
                f"429-retry x{llm.MAX_RETRIES})")
    # Attach the non-invasive reply-token observer on the LLM (event handler,
    # NOT a FrameProcessor in the LLM->TTS path, which broke TTS). The LLM's
    # on_after_push_frame fires for every LLMTextFrame it streams, so the UI
    # reply bubble can stream live tokens without disturbing the TTS flow.
    _install_reply_tap(llm)
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

    await _MCP.start()
    context = LLMContext(tools=_function_schemas())
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=vad))

    in_params = LocalAudioTransportParams(audio_in_enabled=True, audio_in_sample_rate=16000,
                                          audio_in_channels=1, input_device_index=MIC_DEV)
    out_params = LocalAudioTransportParams(audio_out_enabled=True, audio_out_sample_rate=48000,
                                           audio_out_channels=1, output_device_index=SPK_DEV)
    in_transport = LocalAudioInputTransport(pa, in_params)
    out_transport = LocalAudioOutputTransport(pa, out_params)

    # `status` (StatusIndicator) MUST sit immediately AFTER `stt` and BEFORE
    # `user_aggregator`. In pipecat 1.12 the LLMUserAggregator CONSUMES
    # TranscriptionFrame / InterimTranscriptionFrame (it feeds them to the LLM
    # and does NOT push them downstream -- see llm_response_universal.process_
    # frame). So if `status` is placed downstream of `user_aggregator` (as it
    # was), it NEVER sees the user's transcript -> me3.heard stays empty -> no
    # user bubble in the UI. Putting it right after `stt` lets it observe the
    # transcription frames first. StatusIndicator passes every frame through
    # (its process_frame ends with push_frame), so it does not break the LLM.
    # `reply_tap` sits IMMEDIATELY AFTER the LLM so it reliably receives the
    # LLMTextFrame / LLMFullResponseEndFrame / FunctionCallResultFrame that the
    # upstream StatusIndicator never sees in pipecat 1.12. It captures the
    # assistant's spoken reply + tool results into _RECENT_TEXT and pushes them
    # to the UI (fixes: blank assistant bubble on text turns, stale reply).
    pipeline = Pipeline([
        in_transport, input_gate, stt, status, user_aggregator, llm,
        tts, out_transport, assistant_aggregator,
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