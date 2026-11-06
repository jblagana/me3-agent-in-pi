import asyncio   # module-level: agent.py is also IMPORTED (fd-benchmark adapter)
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
# HPC vLLM FALLBACK (2026-11-07): Groq is primary; when a Groq request fails
# (429 / 5xx / 401 / timeout / network) the SAME request is re-sent to the
# HPC vLLM, and voice stays on HPC for ME3_LLM_FALLBACK_STICKY seconds so
# every turn doesn't pay the failed-Groq round trip first. No backoff wait
# on the primary when a fallback exists -- for voice, switching beats waiting.
# NOTE: HPC Qwen3.8 reasons regardless of enable_thinking (D18), so a
# fallback turn is slower -- but it answers instead of going silent.
HPC_LLM_BASE = os.environ.get("ME3_HPC_LLM_BASE", "").rstrip("/")
HPC_LLM_KEY = os.environ.get("ME3_HPC_API_KEY", "")
HPC_LLM_MODEL = os.environ.get("ME3_HPC_LLM_MODEL", "Qwen/Qwen3.8-27B")
LLM_PRIMARY_TIMEOUT_S = float(os.environ.get("ME3_LLM_PRIMARY_TIMEOUT", "6"))
LLM_FALLBACK_STICKY_S = float(os.environ.get("ME3_LLM_FALLBACK_STICKY", "60"))
_FB = {"until": 0.0, "client": None}
# What the voice LLM is ACTUALLY using right now (boot config, a UI hot-swap,
# or the HPC fallback) -- the UI's provider sub-label reads this.
_ACTIVE_LLM = {"base": LLM_BASE_URL.rstrip("/"), "model": LLM_MODEL, "fallback": False}


def _llm_extra(base_url, model, thinking):
    """Provider-aware request extras (the `thinking` switch, done right).

    Groq: reasoning_effort. gpt-oss accepts low|medium|high only ("none" ->
    HTTP 400); Groq Qwen accepts "none" (off) and reasons by default when the
    param is omitted. ME3_REASONING_EFFORT overrides for Groq.
    vLLM (HPC): chat_template_kwargs.enable_thinking (Groq rejects this key
    with HTTP 400 -- never send it there)."""
    base = (base_url or "").lower()
    model = (model or "").lower()
    if "groq.com" in base:
        override = os.environ.get("ME3_REASONING_EFFORT", "").strip().lower()
        if "gpt-oss" in model:
            eff = override if override in ("low", "medium", "high") else ("medium" if thinking else "low")
            return {"reasoning_effort": eff}
        if override in ("none", "low", "medium", "high"):
            return {"reasoning_effort": override}
        return {} if thinking else {"reasoning_effort": "none"}
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}}


ASR_BASE_URL = os.environ.get("ME3_ASR_BASE_URL", "http://127.0.0.1:8080/v1")
# ---------------------------------------------------------------------------
# Voice LLM hot-swap (2026-11-06): the UI dropdown (boots_ui.py POST /llm)
# writes bootsui/llm.json; this poller watches it and re-points the LIVE
# voice pipeline at the new provider mid-run -- no restart, no SSH.
# The AsyncOpenAI client is recreated on change (it bakes base_url+key),
# and the model rides in settings.model (read per request). The swap defers
# while a turn is in flight (bot_speaking or an LLM request open) so a
# mid-response swap can't orphan a completion.
# ---------------------------------------------------------------------------
LLM_JSON_PATH = os.path.expanduser("~/me3/bootsui/llm.json")
_llm_swap = {"mtime": 0.0, "sig": None, "inflight": 0}


def _llm_sig():
    try:
        import json as _json
        with open(LLM_JSON_PATH) as f:
            d = _json.load(f)
        base = d.get("base_url", "").rstrip("/")
        key = d.get("api_key", "")
        # Empty key (the UI no longer persists preset keys into llm.json):
        # use the env key for that provider instead of failing the swap with
        # "Missing credentials" (seen 2026-11-06 16:54).
        if not key:
            if "groq.com" in base:
                key = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY", "")
            elif HPC_LLM_BASE and base == HPC_LLM_BASE:
                key = HPC_LLM_KEY
        return (base, key, d.get("model", ""),
                str(d.get("thinking", "off")).lower() == "on")
    except Exception:
        return None


def _apply_llm_swap(llm, sig):
    # Re-point the live LLM service at (base_url, key, model, thinking).
    from openai import AsyncOpenAI, DefaultAsyncHttpxClient
    old = (getattr(llm, "_settings", None) and llm._settings.model,
           getattr(getattr(llm, "_client", None), "base_url", None))
    llm._client = AsyncOpenAI(api_key=sig[1], base_url=sig[0],
                              http_client=DefaultAsyncHttpxClient())
    llm._settings.model = sig[2]
    # Provider-aware thinking switch (was inverted: "on" sent
    # enable_thinking=False, and Groq got the vLLM-only key -> HTTP 400).
    llm._settings.extra = _llm_extra(sig[0], sig[2], sig[3])
    _ACTIVE_LLM.update(base=sig[0], model=sig[2], fallback=False)
    _FB["until"] = 0.0  # a new primary gets a fresh chance
    th = "on" if sig[3] else "off"
    logger.info(f"[me3] voice LLM swapped: {old[1]} {old[0]} -> {sig[0]} {sig[2]} (thinking {th})")


async def _llm_swap_poller(llm, gate_state):
    import asyncio as _aio
    while True:
        await _aio.sleep(2.0)
        try:
            st = os.stat(LLM_JSON_PATH)
            if st.st_mtime == _llm_swap["mtime"]:
                continue
            _llm_swap["mtime"] = st.st_mtime
            sig = _llm_sig()
            if not sig or sig == _llm_swap["sig"]:
                continue
            # Defer until no turn is in flight (speaking or LLM request open).
            for _ in range(50):
                if not gate_state.get("bot_speaking") and _llm_swap["inflight"] == 0:
                    break
                await _aio.sleep(0.3)
            _apply_llm_swap(llm, sig)
            _llm_swap["sig"] = sig
        except Exception as e:
            logger.warning(f"[me3] llm swap poller: {e!r}")



ME3_ASR_MODEL = os.environ.get("ME3_ASR_MODEL", "whisper-large-v3-turbo")
# Mic gate while Boots speaks (InputGate). 1 = mute the mic during TTS (the
# original no-AEC assumption); 0 = keep it open for BARGE-IN. The EMEET M1A
# was measured (2026-11-07) to do full-duplex echo cancellation on board:
# its own playback reaches the mic as digital silence (-99 dBFS), while the
# user's voice passes at full level (-25 dBFS) during playback.
MIC_GATE = os.environ.get("ME3_GATE_MIC", "1").strip() != "0"
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




# Benchmark mode (fd-benchmark direct feed): no audio device is used, so the
# EMEET check is skipped (the live agent must not hold it anyway).
BENCH_MODE = os.environ.get("ME3_BENCH", "0") == "1"
MIC_DEV = None if BENCH_MODE else _find_mic_index()
if MIC_DEV is None and not BENCH_MODE:
    logger.error("NO MICROPHONE FOUND (no input devices). Plug in the EMEET M1A and re-run.")
    raise SystemExit(2)
SPK_DEV = None if BENCH_MODE else int(os.environ.get("ME3_SPK_DEV", MIC_DEV))  # EMEET is mic+speaker
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
    ErrorFrame,
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
# Per-LLM-round flag: did this round end in a tool call? (set by the reply
# tap on FunctionCallsStartedFrame, reset at LLMFullResponseStartFrame)
_ROUND = {"tool": False}
# Per-round flag: did the reasoning stream deliver real thinking tokens THIS
# LLM round? When yes, the LLMTextFrame handler must NOT overwrite
# _RECENT_TEXT['thinking'] with the reply text (that was the old CoT bug:
# the box mirrored the answer, not the deliberation). Reset at round start.
_REASONING_SEEN = {"round": 0}


def _voice_turn_start():
    """A new utterance was transcribed — reset the per-turn latency stamps."""
    _VOICE_T["heard"] = time.monotonic()
    _VOICE_T["llm_start"] = None
    _VOICE_T["tts_done"] = None
    _VOICE_T["first_audio"] = None
    _VOICE_T["first_token"] = None
    _VOICE_T["first_round"] = None
    _VOICE_LAT.update({"asr": None, "llm": None, "tool": None, "tts": None, "total": None})
    # ASR leg = VAD speech-stop -> final transcript (the real ASR time).
    # speech_end is stamped by StatusIndicator (it sees the VAD frames; the
    # reply tap does not). Computed AFTER the reset (it used to be computed
    # first and then wiped by the update above -> the ASR row was always '–').
    if _VOICE_T.get("speech_end"):
        _VOICE_LAT["asr"] = int((_VOICE_T["heard"] - _VOICE_T["speech_end"]) * 1000)
    # Snapshot THIS turn's end-of-speech: a later VAD stop (noise, a pause)
    # overwrites speech_end, and the total used to be measured from that
    # newer stamp -> 7.35 s of stages vs a 4.08 s "total" (2026-11-07).
    _VOICE_T["turn_speech_end"] = _VOICE_T.get("speech_end")
    # A turn is now in flight: the idle timer must not flip the UI to
    # 'listening' while the LLM thinks / TTS synthesizes (cleared on
    # BotStoppedSpeaking or a text-less round end; 45 s safety cap).
    _TURN_INFLIGHT["until"] = time.monotonic() + 45.0
    _VOICE_T["final_seen"] = False    # set when the FINAL reply round ends
    _PROGRESS["count"] = 0            # progress updates spoken this turn
    _PROGRESS["turn"] = _VOICE_T["heard"]


# Turn-in-flight guard for StatusIndicator's idle timer (2026-11-07).
_TURN_INFLIGHT = {"until": 0.0}

# ---------------------------------------------------------------------------
# PROGRESS UPDATES (2026-11-07): during a long tool chain (research, website
# building) Boots SAYS short updates — "Searching the web for coffee
# statistics.", "Reading britannica.com." — and each renders as a small
# intermediate bubble in the thread, until the final reply. Spoken through
# the pipeline TTS (TTSSpeakFrame, not added to the LLM context), throttled
# (>= PROGRESS_MIN_GAP_S apart, <= PROGRESS_MAX per turn), plus a "Still
# working on it." if nothing was said for PROGRESS_IDLE_S. ME3_PROGRESS=0 off.
# ---------------------------------------------------------------------------
PROGRESS_ENABLED = os.environ.get("ME3_PROGRESS", "1").strip() != "0"
# Each spoken update DELAYS the turn by its own length: pipecat defers the
# tool result to the LLM until the bot stops speaking ("pushing deferred
# context frame") -- 4 updates turned a ~20 s website build into ~60 s
# (measured 2026-11-07). Defaults: ONE update per turn + a 15 s idle check.
PROGRESS_MIN_GAP_S = float(os.environ.get("ME3_PROGRESS_GAP_S", "15"))
PROGRESS_IDLE_S = float(os.environ.get("ME3_PROGRESS_IDLE_S", "15"))
PROGRESS_MAX = int(os.environ.get("ME3_PROGRESS_MAX", "1"))
_SLOW_TOOLS = {"web_search", "fetch_url", "write_website", "spotify_play_search",
               "search_files", "run_command"}
_PROGRESS = {"tts": None, "last": 0.0, "count": 0, "turn": None}


def _progress_text(name, args):
    """Short spoken line for a slow tool call (None = say nothing)."""
    a = args if isinstance(args, dict) else {}
    if name == "web_search":
        q = " ".join(str(a.get("query") or "").replace("'", "").replace('"', "").split()[:8])
        return f"Searching the web for {q}." if q else "Searching the web."
    if name == "fetch_url":
        # the site's NAME, never the domain ("Reading Wikipedia.", not
        # "Reading en dot wikipedia dot org.") — 2026-11-07
        try:
            import urllib.parse as _up
            from remote_voice import site_name
            host = _up.urlparse(str(a.get("url") or "")).netloc
            return f"Reading {site_name(host)}." if host else "Reading the page."
        except Exception:
            return "Reading the page."
    if name == "write_website":
        return "Writing the website now, this takes a few seconds."
    if name == "spotify_play_search":
        return "Looking that up on Spotify."
    if name == "search_files":
        return "Searching your files."
    if name == "run_command":
        return "Running that command."
    return None


def _speak_progress(text, force=False):
    """Speak + render one progress update (throttled). Runs on the loop."""
    if not (PROGRESS_ENABLED and text and _PROGRESS["tts"] is not None):
        return
    now = time.monotonic()
    if not force and (now - _PROGRESS["last"] < PROGRESS_MIN_GAP_S
                      or _PROGRESS["count"] >= PROGRESS_MAX):
        return
    _PROGRESS["last"] = now
    _PROGRESS["count"] += 1
    logger.info(f"[me3] progress: {text}")
    # UI first (the bubble appears as the voice starts), then the voice.
    _me3_push_state(_LAST_STATUS.get("phase") or "tool", _RECENT_TEXT.get("thinking") or "",
                    _RECENT_TEXT.get("reply") or "", extra={"progress": text})
    try:
        import asyncio as _aio
        _aio.get_running_loop().create_task(
            _PROGRESS["tts"].queue_frame(TTSSpeakFrame(text, append_to_context=False)))
    except Exception as e:
        logger.warning(f"[me3] progress TTS failed: {e!r}")


def _schedule_still_working(loop):
    """'Still working on it.' if a turn runs PROGRESS_IDLE_S with nothing said."""
    turn = _PROGRESS["turn"]

    def _check():
        if _PROGRESS["turn"] is not turn or _VOICE_T.get("final_seen") \
                or time.monotonic() >= _TURN_INFLIGHT["until"]:
            return                                   # turn over / superseded
        if time.monotonic() - _PROGRESS["last"] >= PROGRESS_IDLE_S:
            _speak_progress("Still working on it.", force=True)
        loop.call_later(PROGRESS_IDLE_S, _check)
    loop.call_later(PROGRESS_IDLE_S, _check)


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
    # Names MUST match the real TOOL_SPECS (2026-11-07: most keys here were
    # stale -- boots_pause_music, boots_set_temp, ... -- so the UI showed the
    # raw tool name instead of a phrase).
    "boots_play_music": "Playing music",
    "boots_pause": "Pausing music",
    "boots_resume": "Resuming music",
    "boots_stop": "Stopping music",
    "boots_next": "Skipping to the next song",
    "boots_volume_up": "Turning the volume up",
    "boots_volume_down": "Turning the volume down",
    "boots_set_volume": "Adjusting the volume",
    "boots_lights": "Toggling the lights",
    "boots_set_bright": "Setting the brightness",
    "boots_set_color": "Setting the light color",
    "boots_set_temperature": "Adjusting the temperature",
    "boots_set_timer": "Setting a timer",
    "boots_set_alarm": "Setting an alarm",
    "boots_set_reminder": "Setting a reminder",
    "boots_make_call": "Placing a call",
    "check_weather": "Checking the weather",
    "list_dir": "Listing a folder",
    "read_file": "Reading a file",
    "write_file": "Writing a file",
    "edit_file": "Editing a file",
    "search_files": "Searching files",
    "run_command": "Running a command",
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
        # LIVE identity (hot-swap / fallback aware), not the boot constants.
        fb = bool(_ACTIVE_LLM.get("fallback"))
        base = ((HPC_LLM_BASE if fb else _ACTIVE_LLM.get("base")) or "").lower()
        prov = "groq" if "groq.com" in base else ("vllm" if "202.92" in base or "8001" in base else "llm")
        if fb:
            prov += " (fallback)"
        model = (HPC_LLM_MODEL if fb else _ACTIVE_LLM.get("model")) or "llm"
        # gate: lets the UI say "barge-in armed" vs "mic muted" truthfully
        return {"provider": prov, "model": model.split("/")[-1], "gate": MIC_GATE}
    except Exception:
        return {"provider": "llm", "model": "llm"}


def _me3_push(phase: str, label: str = "", extra=None) -> None:
    """Best-effort POST of the current conversation state to the Boots UI.
    extra: flags merged into the body (e.g. {"greeting": True})."""
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
        "thinking": thinking[:1500],  # full reasoning (was [:400] -> CoT froze mid-thought)
        "response": (_RECENT_TEXT.get("reply") or "")[:4000],
        "tool": (_RECENT_TEXT.get("tool") or "")[:120],
        "tool_args": (_RECENT_TEXT.get("tool_args") or "")[:400],
        "cot_round": _LLM_ROUND["n"],
        "tool_result": (_RECENT_TEXT.get("tool_result") or "")[:200],
        "voice": _voice_state(),
        "active": phase != "idle",
    }
    if extra:
        body.update(extra)

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
                # voice: so the UI's model label follows a dropdown switch /
                # fallback within 2 s, not only on the next spoken turn.
                body = {"hb": True, "active": True,
                        "phase": _LAST_STATUS.get("phase") or "listening",
                        "voice": _voice_state()}
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
        # True between BotStartedSpeaking and BotStoppedSpeaking (pipecat's
        # own audio-out events) — the reliable "Boots is talking right now".
        self._bot_talking = False
        self._COOLDOWN_SECS = 0.35

    def _rearm(self):
        """Re-open the mic once the cooldown has elapsed.

        2026-11-07 fix: only AFTER Boots actually stopped (_bot_talking False).
        _rearm_at starts at 0.0 and _rearm runs on every frame, so the old
        check cleared bot_speaking on the very next frame after it was set —
        the mic gate (ME3_GATE_MIC=1) was effectively never applied."""
        if self._state.get("bot_speaking") and not self._bot_talking \
                and time.monotonic() >= self._rearm_at:
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
            # Turn still in flight (LLM thinking / tool / TTS synth): the old
            # 2.5 s timer flipped the UI to 'listening' mid-turn. Re-check in 1 s.
            if time.monotonic() < _TURN_INFLIGHT["until"]:
                self._idle_timer = loop.call_later(1.0, _go)
                return
            if not self._state.get("bot_speaking") and _LAST_STATUS.get("phase") in (
                    "speaking", "thinking", "tool", "transcribing", "listening"):
                # FIX (issue 6): Boots is ALWAYS listening when idle — never show
                # "Idle". The default resting state is "listening".
                self._set("listening", "Listening... (ready)")
        self._idle_timer = loop.call_later(delay, _go)

    async def process_frame(self, frame, direction):
        self._rearm()
        if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            # Pre-warm the Groq ASR connection while the user is still talking
            # (an idle-closed TLS session cost ~0.7 s on the next transcript).
            try:
                import remote_voice as _rvw
                _rvw.prewarm_asr()
            except Exception:
                pass
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
            self.on_transcript(frame.text)
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
            # (2026-11-07) thinking is NOT overwritten with the tool phrase any
            # more: it erased the round's real reasoning from the CoT think
            # row. The CoT's TOOL row (from tool/tool_args) shows the call —
            # same as the typed path.
            # (2026-11-07) The reply is NO LONGER seeded with the tool phrase
            # ("Building a website…"): the dashboard drew it as a Boots bubble
            # that later "mutated" into the real answer. The thread's tool line
            # shows the call; the bubble carries only what Boots actually says.
            self._me3_push_now()   # push immediately so the tool + args hit the UI
            self._set("tool", "Running tool: " + tool_name)
            if names:
                logger.info(f"[me3] calling tool(s): {names} args={_RECENT_TEXT.get('tool_args')!r}")
            # Progress update for a slow tool (spoken + intermediate bubble),
            # and arm the "still working" check for this turn (once).
            if PROGRESS_ENABLED and fcs:
                try:
                    if names[0] in _SLOW_TOOLS:
                        _speak_progress(_progress_text(names[0], fcs[0].arguments or {}))
                    if not getattr(self, "_still_armed_for", None) is _PROGRESS["turn"]:
                        self._still_armed_for = _PROGRESS["turn"]
                        import asyncio as _aio
                        _schedule_still_working(_aio.get_running_loop())
                except Exception as e:
                    logger.debug(f"[me3] progress hook skipped: {e!r}")
        # NOTE: FunctionCallResultFrame is handled by _install_reply_tap's event
        # handler on the LLM (see above).
        elif isinstance(frame, TTSStartedFrame):
            self._state["bot_speaking"] = True
            self._clear_idle_timer()
            self._set("speaking", "Speaking...")
        elif isinstance(frame, (BotStartedSpeakingFrame, TTSAudioRawFrame)):
            if isinstance(frame, BotStartedSpeakingFrame):
                self._bot_talking = True
            if not self._state.get("bot_speaking"):
                self._state["bot_speaking"] = True
                self._set("speaking", "Speaking...")
            # First real audio out (2026-11-06): stamp the TTS leg (LLM end ->
            # first audio) + the total (heard -> first audio), then push the
            # latency dict so the UI's LATENCY column shows the voice turn's
            # real numbers (it was dead for voice turns — typed chat only).
            # Contiguous legs (2026-11-07) so asr+llm+tool+tts == total:
            # tts = first SPOKEN token -> first audio; total = END OF SPEECH ->
            # first audio (the E2E number for the deck). tts used to start at
            # LLM end and total at the transcript, so the legs overlapped
            # (5.8 s of legs vs a 4.0 s "total").
            if _VOICE_T.get("first_token") and _VOICE_LAT["tts"] is None:
                _VOICE_LAT["tts"] = int((time.monotonic() - _VOICE_T["first_token"]) * 1000)
                _t0 = _VOICE_T.get("turn_speech_end") or _VOICE_T["heard"]
                if _t0:
                    _VOICE_LAT["total"] = int((time.monotonic() - _t0) * 1000)
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
            self._bot_talking = False
            # Reply audio done -> idle timer may rest. NOT after a progress
            # update mid-turn (the final reply hasn't come yet), or the UI
            # would flip to 'listening' between updates.
            if _VOICE_T.get("final_seen"):
                _TURN_INFLIGHT["until"] = 0.0
            elif time.monotonic() < _TURN_INFLIGHT["until"]:
                # a progress update just finished; the turn continues
                self._set("tool" if _RECENT_TEXT.get("tool") else "thinking",
                          "Working... (after progress update)")
            if getattr(self, "_tool_t0", None):
                dt = time.monotonic() - self._tool_t0
                logger.info(f"[me3] tool+LLM phase took {dt:.1f}s")
                self._tool_t0 = None
            self._rearm_at = time.monotonic() + self._COOLDOWN_SECS
            self._schedule_idle(2.5)
        elif isinstance(frame, InterruptionFrame):
            self._clear_idle_timer()
            self._mic_active = True
            # The interrupted reply is over: release the turn guard + the
            # speaking flag so the new utterance starts a clean turn.
            # pipecat emits an InterruptionFrame at EVERY user-turn start, so
            # it's only a barge-in when Boots was actually talking.
            was_speaking = self._bot_talking
            _TURN_INFLIGHT["until"] = 0.0
            self._state["bot_speaking"] = False
            if was_speaking:
                logger.info("[me3] BARGE-IN: user interrupted Boots mid-reply")
                self._set("listening", "Barge-in: you interrupted me")
            else:
                self._set("listening", "Listening... (you're speaking)")
        # FIX (ASR stall root cause): forward with the ORIGINAL direction. The
        # aggregator's VAD broadcasts VADUserStoppedSpeakingFrame upstream
        # (direction=UPSTREAM) so it can reach the STT, which sits upstream of
        # the aggregator in the pipeline. The previous code called
        # push_frame(frame) with no direction -> default DOWNSTREAM, which
        # reflected those upstream VAD frames back toward the LLM and starved
        # the SegmentedSTTService of its speech-stop trigger (run_stt never
        # fired -> no transcript -> no "heard" bubble in the UI).
        await self.push_frame(frame, direction)

    def on_transcript(self, text):
        """A new utterance was transcribed: start the turn (latency clocks,
        turn-in-flight guard, UI reset, phase -> thinking).

        2026-11-07: the remote ASR delivers TranscriptionFrames straight to
        the aggregator (remote_voice.Qwen2AudioSTTService.push_frame), so
        this processor NEVER saw them -- _voice_turn_start() never ran (no
        ASR/total latency, so the voice CoT never settled), the idle guard
        never armed, and "[me3] you said" was never logged. remote_voice now
        calls this via its ON_TRANSCRIPT hook; the frame branch stays for the
        local whisper path. Same text within 2 s = the other route -> skip."""
        text = (text or "").strip()
        now = time.monotonic()
        if text and text == getattr(self, "_last_tx", None) \
                and now - getattr(self, "_last_tx_t", 0.0) < 2.0:
            return
        self._last_tx, self._last_tx_t = text, now
        _voice_turn_start()  # new utterance: reset the per-turn latency legs
        _RECENT_TEXT["transcript"] = text
        _RECENT_TEXT["reply"] = None
        _RECENT_TEXT["tool"] = None
        _RECENT_TEXT["tool_args"] = None
        _RECENT_TEXT["thinking"] = None
        _RECENT_TEXT["tool_result"] = None
        self._reply_buf = []
        self._thinking_buf = []
        self._clear_idle_timer()
        self._me3_push_now()
        self._set("thinking", "Thinking... (LLM generating reply)")
        if text:
            logger.info(f"[me3] you said: {text}")

    def _me3_push_now(self):
        """Push current heard/thinking/response/tool/mic to the Boots UI."""
        try:
            import json as _j, threading as _t, urllib.request as _u
            # 'thinking' = streamed LLM text if any, else the tool phrase (so the
            # bubble is never blank on pure function-call turns).
            thinking = ("".join(self._thinking_buf) or _RECENT_TEXT.get("thinking") or "")
            body = {"phase": _LAST_STATUS.get("phase", "listening"),
                    "heard": (_RECENT_TEXT.get("transcript") or "")[:300],
                    "thinking": thinking[:1500],  # full reasoning (was [:400] -> CoT froze mid-thought)
                    "response": (_RECENT_TEXT.get("reply") or "")[:4000],
                    "tool": (_RECENT_TEXT.get("tool") or "")[:120],
                    "tool_args": (_RECENT_TEXT.get("tool_args") or "")[:400],
                    # round number too (2026-11-07): without it the server fell
                    # back to guessing a round from "thinking reappeared" and
                    # minted a 2nd think row ("Need" + "Need to set alarm.")
                    "cot_round": _LLM_ROUND["n"],
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


def _me3_push_state(phase: str, thinking: str, reply: str, extra=None) -> None:
    """Push the current reply/thinking/tool to the Boots UI (fire-and-forget).
    extra: flags merged into the body (e.g. {"final": True})."""
    try:
        import json as _j, threading as _t, urllib.request as _u
        body = {"phase": phase,
                "heard": (_RECENT_TEXT.get("transcript") or "")[:300],
                "thinking": (thinking or "")[:1500],  # full reasoning, like the typed path
                "response": (reply or "")[:4000],
                "tool": (_RECENT_TEXT.get("tool") or "")[:120],
                "tool_args": (_RECENT_TEXT.get("tool_args") or "")[:400],
                "cot_round": _LLM_ROUND["n"],
                "tool_result": (_RECENT_TEXT.get("tool_result") or "")[:200],
                "voice": _voice_state(),
                "latency": {k: v for k, v in _VOICE_LAT.items() if isinstance(v, (int, float))} or None,
                "mic": False, "active": True}
        if extra:
            body.update(extra)
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
            if isinstance(frame, FunctionCallsStartedFrame):
                _ROUND["tool"] = True        # this round ends in a tool call
                _pre = "".join(reply_buf).strip()
                if _pre:   # text said before the tool call -> history step
                    _me3_push_state(_LAST_STATUS.get("phase") or "tool", "", _pre,
                                    extra={"preamble": _pre})
            if isinstance(frame, LLMFullResponseStartFrame):
                _ROUND["tool"] = False
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
                    # llm = transcript -> first SPOKEN token, minus tool time
                    # (contiguous with asr before it and tts after it).
                    if _VOICE_LAT["llm"] is None and _VOICE_T["heard"]:
                        _VOICE_T["first_token"] = time.monotonic()
                        _VOICE_LAT["llm"] = max(0, int((_VOICE_T["first_token"] - _VOICE_T["heard"]) * 1000)
                                                - (_VOICE_LAT["tool"] or 0))
                    _RECENT_TEXT["reply"] = reply
                    # CoT shows the REAL thinking, not the reply (2026-11-06):
                    # when the reasoning stream delivered tokens this round,
                    # _RECENT_TEXT['thinking'] already holds the model's
                    # deliberation — do NOT overwrite it with the reply text.
                    # The reply-mirror fallback stays for rounds WITHOUT
                    # (2026-11-07) The reply-mirror fallback is GONE: the
                    # think row carries ONLY real reasoning, exactly like the
                    # typed /chat_stream path. A round with no reasoning gets
                    # no think row (cotDone drops the empty one) — the reply
                    # lives in the chat bubble, not the CoT.
                    # Push with the REAL phase (2026-11-06 dup-bubble fix):
                    # forcing "speaking" per token made the UI see
                    # thinking->speaking->thinking... per token, and the server
                    # appends a history turn on EVERY speaking->non-speaking
                    # transition -> N identical assistant bubbles. The UI
                    # renders the reply bubble whenever phase==speaking OR the
                    # turn just finished (its own liveReply gate), so the real
                    # phase is safe.
                    _me3_push_state(_LAST_STATUS.get("phase", "thinking"),
                                    _RECENT_TEXT.get("thinking") or "", reply)
            elif isinstance(frame, LLMFullResponseEndFrame):
                # Finalize: keep whatever was streamed. Fall back to the seeded
                # tool phrase ONLY when the LLM produced no text (a pure
                # function-call turn) — the old `or _RECENT_TEXT["reply"]`
                # could clobber a real reply with the tool phrase if a
                # FunctionCallResultFrame raced in.
                streamed = "".join(reply_buf).strip()
                reply = streamed or _RECENT_TEXT.get("reply") or ""
                _RECENT_TEXT["reply"] = reply
                _RECENT_TEXT["thinking"] = ""  # done; stop the thinking spinner text
                _voice_tts_end()  # TTS leg clock starts now (LLM finished)
                # 2026-11-07 (dup-bubble + phase-sync fix): push the REAL phase
                # (no more forced "speaking" -- that lit SPEAKING before any
                # audio, and on a tool turn's FIRST round it stored the seeded
                # tool phrase "Playing music…" as a history turn, then the real
                # reply as a second one = duplicate assistant bubbles). 'final'
                # marks a round that actually produced spoken text: the server
                # appends THAT to history. SPEAKING now comes only from real
                # audio (BotStartedSpeaking).
                # FINAL = this round produced spoken text AND did not end in a
                # tool call (2026-11-07). Pre-tool text ("Let me look that
                # up.") is NOT the turn's reply — flagging it final stored it
                # as a separate history turn (duplicate HEARD on refresh).
                is_final = bool(streamed) and not _ROUND["tool"]
                if is_final:
                    _VOICE_T["final_seen"] = True
                if not streamed and not _ROUND["tool"]:
                    _TURN_INFLIGHT["until"] = 0.0  # nothing to say -> let idle rest
                _me3_push_state(_LAST_STATUS.get("phase") or "thinking", "", reply,
                                extra={"final": True} if is_final else None)
            elif isinstance(frame, FunctionCallResultFrame):
                result = str(getattr(frame, "result", "") or "").strip()
                fname = getattr(frame, "function_name", "") or ""
                # (2026-11-07) The tool RESULT is NOT copied into the reply any
                # more: the dashboard rendered it as a Boots bubble (a raw
                # fetch_url page dump), which the final answer then overwrote
                # in place. It travels in the dedicated tool_result field below.
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
                    _VOICE_LAT["tool"] = (_VOICE_LAT["tool"] or 0) + int((time.monotonic() - _TOOL_T0["t"]) * 1000)
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
            if MIC_GATE and self._state.get("bot_speaking"):
                # replace audio with silence so the VAD/ASR see no speech from the bot.
                # (ME3_GATE_MIC=0 skips this: the EMEET M1A has full-duplex
                # hardware AEC -- measured 2026-11-07 -- so the mic stays open
                # while Boots speaks and the user can barge in.)
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
            # env=os.environ (2026-11-07): the MCP stdio client otherwise
            # passes ONLY HOME/LOGNAME/PATH/SHELL/USER, so TAVILY_API_KEY,
            # WEBSITE_HOST, ME3_BOOTSUI, ... never reached the tools.
            params = StdioServerParameters(
                command=os.path.join(here, ".venv", "bin", "python"),
                args=[os.path.join(here, "mcp_server.py")],
                cwd=here, env=dict(os.environ))
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
        import asyncio as _aio
        try:
            res = await _aio.wait_for(
                self._session.call_tool(tool, dict(args or {})), timeout=60)
            text = "".join(getattr(c, "text", "") or "" for c in res.content)
            self.calls_via_mcp += 1
            logger.info("[mcp] ran %s via bridge (%d total)" % (tool, self.calls_via_mcp))
            return text or "(tool returned nothing)"
        except _aio.TimeoutError:
            # NOT a fallback trigger (2026-11-07): the tool may still be
            # running in the MCP server, and re-running it directly would
            # double its side effects (two websites, two reminders...).
            logger.warning("[mcp] call %s timed out after 60s (no re-run)" % tool)
            return ("That's taking longer than expected, so I stopped waiting. "
                    "It may still finish in the background.")
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


def _required(mod, name, props):
    """Schema `required` = all params except the module's OPTIONAL_PARAMS."""
    opt = getattr(mod, "OPTIONAL_PARAMS", {}).get(name, ())
    return [k for k in props if k not in opt]


def _function_schemas():
    """Build pipecat FunctionSchema tools for ME2 UI + Spotify.

    Each schema carries a `handler`, so the LLM service auto-registers it â€” no
    separate register_function call is needed (pipecat 1.12).
    """
    out = []
    for (n, d, p) in ui.TOOL_SPECS:
        out.append(FunctionSchema(n, d, _normalize_props(p), _required(ui, n, p), handler=_ui_handler))
    for (n, d, p) in sp.TOOL_SPECS:
        out.append(FunctionSchema(n, d, _normalize_props(p), _required(sp, n, p), handler=_spotify_handler))
    for (n, d, p) in web.TOOL_SPECS:
        out.append(FunctionSchema(n, d, _normalize_props(p), _required(web, n, p), handler=_web_handler))
    for (n, d, p) in ft.TOOL_SPECS:
        out.append(FunctionSchema(n, d, _normalize_props(p), _required(ft, n, p), handler=_files_handler))
    for (n, d, p) in sh.TOOL_SPECS:
        out.append(FunctionSchema(n, d, _normalize_props(p), _required(sh, n, p), handler=_shell_handler))
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
    # greeting flag: the server clears the stale last-heard so the greeting
    # is stored as an assistant-only turn (it used to be paired with the
    # previous session's last utterance -> a duplicated HEARD bubble).
    _me3_push("speaking", extra={"greeting": True})
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
            _me3_push("idle", extra={"greeting": True})
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
_BASE_SYSTEM_TEXT = None  # was never defined -> NameError swallowed -> no date stamp (2026-11-07 fix)


def _stamp_system_and_strip(context, llm) -> None:
    """Guarantee the request carries EXACTLY ONE system message, at
    position 0, with a fresh date stamp (2026-11-06b, re-merged 20:20+).

    pipecat's OpenAI adapter (open_ai_adapter.py) ALWAYS prepends
    settings.system_instruction when it is set, and keeps any context
    system message too ("Keeping both") -- so a context system message
    + the settings one = two system messages = Qwen3.8 template 400
    'System message must be at the beginning' -> LLM marked unusable.
    The logged context looked clean because the debug dump runs BEFORE
    the adapter. Fix: fold the date stamp into settings.system_instruction
    (the single message the adapter prepends) and strip every
    system/developer message from the context.
    """
    global _BASE_SYSTEM_TEXT
    try:
        import time as _t
        now = _t.localtime()
        stamp = _t.strftime("%A, %B %d, %Y %H:%M %Z", now)
        line = f"\n\nCurrent date and time: {stamp} (the user's local time is Asia/Manila, UTC+8)."
        if _BASE_SYSTEM_TEXT is None:
            _BASE_SYSTEM_TEXT = SYSTEM_PROMPT
        try:
            llm._settings.system_instruction = _BASE_SYSTEM_TEXT + line
        except Exception:
            pass
        for _m in list(context.messages):
            _r = (_m.get("role") if isinstance(_m, dict)
                  else getattr(_m, "role", "")) or ""
            if _r in ("system", "developer"):
                context.messages.remove(_m)
    except Exception as e:
        logger.debug(f"[me3] system stamp/strip skipped: {e!r}")


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
            # NEVER trim away the CURRENT turn's request (2026-11-07): a long
            # tool chain (website research: 2 messages per step) used to push
            # the user's own message out of the window mid-build, so the model
            # lost what it was building and the website guard lost the intent.
            roles = [(m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) for m in tail]
            last_user = max((j for j, r in enumerate(roles) if r == "user"), default=None)
            if last_user is not None and drop > last_user:
                drop = last_user
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


import re as _re
_SITE_INTENT = _re.compile(
    # "website/web page" anywhere, or a bare "site" ONLY right after a build
    # verb ("build a site") -- "visit the site office" must not match.
    r"\b(web ?site|web ?page|landing page)\b"
    r"|\b(make|build|create|write|generate|design)\b\s+(me\s+|us\s+)?(a|an|my|the|our)?\s*(new\s+|simple\s+|small\s+)?site\b",
    _re.I)
_SITE_RESEARCH_MIN = 4   # prompt: web_search 2-3x + fetch_url 2+ before writing


def _site_guard_choice(context):
    """tool_choice forcing write_website, or None. Fires when the latest
    user turn asks for a website, >= _SITE_RESEARCH_MIN research tool
    results have come back since it, and write_website hasn't been called."""
    try:
        msgs = list(context.messages)
        role = lambda m: (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) or ""
        last_user = max((i for i, m in enumerate(msgs) if role(m) == "user"), default=None)
        if last_user is None:
            return None
        um = msgs[last_user]
        utext = um.get("content") if isinstance(um, dict) else getattr(um, "content", "")
        if not isinstance(utext, str) or not _SITE_INTENT.search(utext):
            return None
        research = 0
        for m in msgs[last_user + 1:]:
            if role(m) == "tool":
                research += 1
            for tc in ((m.get("tool_calls") if isinstance(m, dict) else None) or []):
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                if fn.get("name") == "write_website":
                    return None          # already built (or building)
        if research >= _SITE_RESEARCH_MIN:
            return {"type": "function", "function": {"name": "write_website"}}
    except Exception as e:
        logger.debug(f"[me3] website guard skipped: {e!r}")
    return None


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
                            _RECENT_TEXT["thinking"] = "".join(buf)[:1500]  # typed path keeps 1500 too
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
        # Round end: FLUSH the full reasoning (2026-11-07). The 0.35 s push
        # throttle sent only the FIRST chunk of a short burst (gpt-oss often
        # reasons in < 0.35 s) and the round-end clear came before any
        # further push -> the CoT think row was stuck on "Need". Accumulated
        # reasoning also stays in _RECENT_TEXT; the next round start clears it.
        if buf:
            try:
                _me3_push_state(_LAST_STATUS.get("phase") or "thinking",
                                "".join(buf)[:1500], _RECENT_TEXT.get("reply") or "")
            except Exception:
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
        _stamp_system_and_strip(context, self)
        # (2026-11-07) reasoning_effort is no longer forced to 'medium' on
        # every request: _llm_extra() sets the provider-correct value at boot
        # and on every hot-swap (gpt-oss rejects 'none'; vLLM rejects
        # reasoning_effort-style keys it doesn't know).
        # 1) Keep the request light: trim to the most recent N messages.
        if self.MAX_CONTEXT_MESSAGES > 0:
            _trim_llm_context(context, self.MAX_CONTEXT_MESSAGES)
        # 1b) Website guard (2026-11-07): force write_website once research is
        #     done, so the model can't SPEAK a fake "it's built" + made-up URL
        #     (gpt-oss did exactly that on the typed path). Restored right
        #     after the request opens -- it only affects this one round.
        _forced = _site_guard_choice(context)
        _old_tc = None
        _nudge = None
        if _forced:
            _old_tc = context.tool_choice
            context.set_tool_choice(_forced)
            # tool_choice alone isn't enough: Groq enforces it server-side and
            # gpt-oss still answered in text -> "Tool choice is required, but
            # model did not call a tool" (tested 2026-11-07). An explicit
            # instruction (same as the typed path's) makes it comply. It is
            # TEMPORARY: removed right after the request is built, so it never
            # enters the conversation history.
            _nudge = {"role": "user", "content": (
                "Research is done. Call write_website now — write the sections, "
                "stats, and sources from what your research found.")}
            context.messages.append(_nudge)
            logger.info("[me3] website guard: research done, forcing write_website")
        _llm_swap["inflight"] += 1
        try:
            if not self._has_fallback():
                return await self._complete_with_retry(context)
            if time.monotonic() < _FB["until"]:
                return await self._complete_via_fallback(context)   # sticky
            try:
                stream = await self._complete_once(context, timeout=LLM_PRIMARY_TIMEOUT_S)
                _ACTIVE_LLM["fallback"] = False
                return stream
            except Exception as e:
                logger.warning(f"[me3] primary LLM failed ({type(e).__name__}: {str(e)[:120]}); "
                               f"falling back to HPC for {LLM_FALLBACK_STICKY_S:.0f}s")
                _FB["until"] = time.monotonic() + LLM_FALLBACK_STICKY_S
                return await self._complete_via_fallback(context)
        finally:
            _llm_swap["inflight"] -= 1
            if _forced:
                context.set_tool_choice(_old_tc)
                try:
                    context.messages.remove(_nudge)
                except ValueError:
                    pass

    # ---- fallback helpers (2026-11-07) ----------------------------------
    def _primary_base(self) -> str:
        return str(getattr(self._client, "base_url", "") or "").rstrip("/")

    def _has_fallback(self) -> bool:
        """HPC fallback configured AND the primary isn't already the HPC."""
        return bool(HPC_LLM_BASE and HPC_LLM_KEY) and self._primary_base() != HPC_LLM_BASE

    async def _complete_once(self, context, timeout=None):
        """One request; returns the reasoning-tapped stream. The stream is
        bound to the client that opened it, so the caller may restore
        self._client right after this returns."""
        import asyncio as _aio
        coro = super().get_chat_completions(context)
        stream = await (_aio.wait_for(coro, timeout) if timeout else coro)
        # REAL CoT: tee delta.reasoning into the UI (pipecat drops it). A
        # stream wrapper, NOT a FrameProcessor in the LLM->TTS path (D21).
        return _tap_reasoning_stream(stream)

    async def _complete_via_fallback(self, context):
        """Send this request to the HPC vLLM, then restore the primary."""
        from openai import AsyncOpenAI, DefaultAsyncHttpxClient
        if _FB["client"] is None:
            _FB["client"] = AsyncOpenAI(api_key=HPC_LLM_KEY, base_url=HPC_LLM_BASE,
                                        http_client=DefaultAsyncHttpxClient())
        saved = (self._client, self._settings.model, self._settings.extra)
        self._client = _FB["client"]
        self._settings.model = HPC_LLM_MODEL
        self._settings.extra = _llm_extra(HPC_LLM_BASE, HPC_LLM_MODEL, False)
        _ACTIVE_LLM.update(fallback=True)
        try:
            return await self._complete_with_retry(context)
        finally:
            self._client, self._settings.model, self._settings.extra = saved

    async def _complete_with_retry(self, context):
        """Bounded retry with exponential backoff (D23) -- used when there is
        no further fallback (HPC itself, or no HPC configured)."""
        import asyncio as _aio
        from openai import APIStatusError, RateLimitError, APIConnectionError, APITimeoutError
        if True:
            attempt = 0
            while True:
                attempt += 1
                try:
                    return await self._complete_once(context)
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


# ---------------------------------------------------------------------------
# Fallback TTS on pipeline errors (2026-11-06, re-merged 20:20+): pipecat
# turns tool-validation failures and LLM/STT errors into ErrorFrames; with no
# handler the pipeline just goes quiet and the user gets silence. These
# processors intercept ErrorFrames and speak a short apology THROUGH the
# pipeline TTS service (a second PyAudio stream on the held speaker fails
# with OSError -9985). Cooldown dedupes bursts (a 429 fans out into frames).
# ---------------------------------------------------------------------------
_ERROR_APOLOGY_COOLDOWN_S = 20.0
_last_error_apology = 0.0

async def _speak_error_apology(tts, spk_dev: int, state: dict, text: str):
    """Speak the apology THROUGH the pipeline TTS service."""
    try:
        import inspect as _insp
        _kw = {"append_to_context": False}
        if "allow_interruptions" in _insp.signature(TTSSpeakFrame.__init__).parameters:
            _kw["allow_interruptions"] = False
        await tts.queue_frame(TTSSpeakFrame(text, **_kw))
        logger.info("[me3] error apology queued through pipeline TTS")
    except Exception as e:
        logger.error(f"[me3] error apology queue failed: {e!r}")

class ErrorApologyTTS(FrameProcessor):
    """Catch ErrorFrames and speak a short apology THROUGH the pipeline TTS
    service so the user never gets silence. Cooldown-deduped."""

    def __init__(self, tts, spk_dev: int, state: dict,
                 text: str = "Sorry, that didn't go through."):
        super().__init__()
        self._tts = tts
        self._spk_dev = spk_dev
        self._state = state
        self._text = text

    async def process_frame(self, frame, direction):
        await self.push_frame(frame, direction)
        if isinstance(frame, ErrorFrame):
            global _last_error_apology
            now = time.monotonic()
            if now - _last_error_apology > _ERROR_APOLOGY_COOLDOWN_S:
                _last_error_apology = now
                logger.warning(f"[me3] ErrorFrame caught -> apology: {frame.error!r}")
                await _speak_error_apology(self._tts, self._spk_dev, self._state, self._text)

class _KeepAliveMixer:
    """Pass-through output 'mixer' that keeps the EMEET speaker stream FED.

    2026-11-07: replies started soft and faded up. Cause (A/B-tested by ear):
    the M1A ramps its output up whenever its USB stream (re)starts. Without a
    mixer, pipecat's output loop writes NOTHING between replies -> the stream
    runs dry -> every reply restarts it -> fade-in (padding length didn't
    matter). With a mixer set, pipecat writes silence continuously while idle
    (silence never counts as bot speech), so the stream never stops. This
    mixer changes no audio: mix() returns its input unchanged."""

    def __new__(cls):
        from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer

        class _M(BaseAudioMixer):
            async def start(self, sample_rate: int):
                pass

            async def stop(self):
                pass

            async def process_frame(self, frame):
                pass

            async def mix(self, audio: bytes) -> bytes:
                return audio
        return _M()


class _SpotifyPipe:
    """Reads librespot's raw PCM (S16LE, 44.1 kHz, stereo) from a FIFO and
    keeps ~0.3 s of it as 48 kHz mono, read at REAL-TIME pace: librespot's pipe
    backend writes as fast as the reader takes, so reading only when the buffer
    runs low is what paces the music (and never skips audio). 2026-11-07."""

    def __init__(self, path):
        import threading
        self.path = path
        self.buf = bytearray()
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True, name="spotify-pipe").start()

    def _run(self):
        import numpy as _np, soxr
        keep = int(48000 * 2 * 0.3)            # target backlog, bytes at 48 kHz mono s16
        while True:
            try:
                if not os.path.exists(self.path):
                    time.sleep(2)
                    continue
                rs = soxr.ResampleStream(44100, 48000, 1, dtype="int16")
                with open(self.path, "rb", buffering=0) as f:   # blocks until librespot opens it
                    while True:
                        with self.lock:
                            full = len(self.buf) >= keep
                        if full:
                            time.sleep(0.01)
                            continue
                        chunk = f.read(4 * 882)                 # ~20 ms of stereo s16
                        if not chunk:
                            break                               # writer closed: reopen
                        st = _np.frombuffer(chunk[:len(chunk) // 4 * 4], dtype=_np.int16).reshape(-1, 2)
                        mono = st.astype(_np.int32).mean(axis=1).astype(_np.int16)
                        out = rs.resample_chunk(mono)
                        with self.lock:
                            self.buf += out.tobytes()
            except Exception as e:
                logger.debug(f"[spotify-pipe] {e!r}")
                time.sleep(1)

    def take(self, nbytes):
        with self.lock:
            if not self.buf:
                return None
            out = bytes(self.buf[:nbytes])
            del self.buf[:nbytes]
        return out


class _MusicOutputTransport(LocalAudioOutputTransport):
    """EMEET output that also plays Spotify (librespot) through the SAME stream
    the agent owns (the M1A can't be opened twice, and there is no PipeWire).
    Music is mixed at the very last step (write_audio_frame), AFTER pipecat's
    is-the-bot-speaking check, so music never counts as Boots talking. Ducked
    to ME3_MUSIC_DUCK while Boots speaks. The M1A's echo cancelling removes
    the music from the mic too, so barge-in keeps working. 2026-11-07."""

    MUSIC_GAIN = float(os.environ.get("ME3_MUSIC_GAIN", "0.8"))
    DUCK = float(os.environ.get("ME3_MUSIC_DUCK", "0.2"))

    def __init__(self, pa, params, pipe):
        super().__init__(pa, params)
        self._pipe = pipe
        self._g = 1.0

    async def write_audio_frame(self, frame):
        try:
            music = self._pipe.take(len(frame.audio)) if self._pipe else None
            if music:
                import numpy as _np
                a = _np.frombuffer(frame.audio, dtype=_np.int16)
                m = _np.frombuffer(music, dtype=_np.int16)
                if len(m) < len(a):
                    m = _np.concatenate([m, _np.zeros(len(a) - len(m), _np.int16)])
                talking = bool(len(a)) and int(_np.max(_np.abs(a))) > 64
                self._g += ((self.DUCK if talking else 1.0) - self._g) * 0.35
                mix = a.astype(_np.int32) + (m[:len(a)].astype(_np.float32) * self.MUSIC_GAIN * self._g).astype(_np.int32)
                frame.audio = _np.clip(mix, -32768, 32767).astype(_np.int16).tobytes()
        except Exception as e:
            logger.debug(f"[music-mix] {e!r}")
        return await super().write_audio_frame(frame)


SPOTIFY_FIFO = os.environ.get("ME3_SPOTIFY_FIFO", "/run/me3-spotify/spotify.pcm")


async def build_bot(transport=None):
    """Build Boots' pipeline. Returns (task, tts, gate_state, pa).

    transport=None  -> the normal service: EMEET mic/speaker via PyAudio.
    transport=<obj> -> benchmark mode (fd-benchmark direct feed, 2026-11-07):
                       the harness's Pipecat transport (.input() / .output())
                       replaces the local audio; no greeting, no UI heartbeat,
                       no audio device is opened. Everything else (VAD, ASR,
                       LLM + tools over MCP, TTS, barge-in) is identical.
    """
    bench = transport is not None
    pa = None if bench else pyaudio.PyAudio()
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
    # Provider-aware extras (thinking OFF for voice latency); the llm.json
    # poller re-derives these on every hot-swap via the same _llm_extra().
    llm_settings["extra"] = _llm_extra(LLM_BASE_URL, LLM_MODEL, False)
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
                f"429-retry x{llm.MAX_RETRIES}; HPC fallback "
                f"{'ON -> ' + HPC_LLM_MODEL if (HPC_LLM_BASE and HPC_LLM_KEY) else 'OFF (ME3_HPC_* unset)'})")
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
    # 2026-11-07 retune (barge-in + "Boots doesn't hear me"): 0.90/0.85 were
    # set when the mic had to reject Boots' own echo. The M1A's hardware AEC
    # now does that, and the strict thresholds missed the user entirely while
    # Boots spoke (0 VAD events across a 60 s reply the user talked over).
    # confidence 0.75 (pipecat default 0.7; a notch stricter for room noise,
    # which would now also interrupt Boots), min_volume 0.6 (= default),
    # stop_secs 0.6 (was 0.3: "Tell me the history of" got cut mid-sentence).
    vad = SileroVADAnalyzer(params=VADParams(
        confidence=float(os.environ.get("ME3_VAD_CONFIDENCE", "0.75")),
        start_secs=float(os.environ.get("ME3_VAD_START_SECS", "0.2")),
        stop_secs=float(os.environ.get("ME3_VAD_STOP_SECS", "0.6")),
        min_volume=float(os.environ.get("ME3_VAD_MIN_VOLUME", "0.6"))))
    logger.info("[me3] VAD: confidence=%s start=%ss stop=%ss min_volume=%s | mic gate %s"
                % (vad.params.confidence, vad.params.start_secs, vad.params.stop_secs,
                   vad.params.min_volume, "ON (muted while speaking)" if MIC_GATE else "OFF (barge-in)"))
    gate_state = {"bot_speaking": False, "mic_level": 0.0}
    import asyncio as _aio_mod
    # mtime stays 0.0: the first poll (~2s) reads llm.json and compares
    # its signature to the boot config, so a pre-existing UI choice is
    # applied at startup too (not only on the next dropdown change).
    _aio_mod.create_task(_llm_swap_poller(llm, gate_state))
    input_gate = InputGate(state=gate_state)
    status = StatusIndicator(state=gate_state)
    _PROGRESS["tts"] = tts   # progress updates speak through the pipeline TTS
    # Transcripts bypass StatusIndicator (remote_voice re-delivers them to the
    # aggregator directly) -> hand them over through the ASR's hook instead.
    import remote_voice as _rv
    _rv.ON_TRANSCRIPT = status.on_transcript
    if not bench:
        _me3_heartbeat()   # FIX: liveness heartbeat (UI: LISTENING vs OFF)

    await _MCP.start()
    context = LLMContext(tools=_function_schemas())
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=vad,
                                                     empty_user_turn=None))

    if bench:
        in_transport, out_transport = transport.input(), transport.output()
    else:
        in_params = LocalAudioTransportParams(audio_in_enabled=True, audio_in_sample_rate=16000,
                                              audio_in_channels=1, input_device_index=MIC_DEV)
        out_params = LocalAudioTransportParams(audio_out_enabled=True, audio_out_sample_rate=48000,
                                               audio_out_channels=1, output_device_index=SPK_DEV,
                                               audio_out_mixer=(_KeepAliveMixer()  # no fade-in
                                                                if os.environ.get("ME3_KEEPALIVE", "1") != "0" else None))
        in_transport = LocalAudioInputTransport(pa, in_params)
        # Spotify (librespot "Boots") mixed into the EMEET stream (2026-11-07);
        # ME3_SPOTIFY=0 turns the mixing off.
        if os.environ.get("ME3_SPOTIFY", "1") != "0":
            out_transport = _MusicOutputTransport(pa, out_params, _SpotifyPipe(SPOTIFY_FIFO))
        else:
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
    # ONE apology processor, UPSTREAM of the LLM (2026-11-07 fix). The 22:40
    # merge re-inserted two ErrorApologyTTS processors BETWEEN the LLM and the
    # TTS -- the exact D21 failure: a custom FrameProcessor in the LLM->TTS
    # path silently stops the reply text reaching the TTS (the LLM answered,
    # the UI showed it, Boots said nothing). Pipecat pushes LLM/TTS errors
    # UPSTREAM, so this one position still sees them; it speaks via
    # tts.queue_frame(), so it never needs to sit next to the TTS.
    # RULE: never put a FrameProcessor between `llm` and `tts`.
    err_apology = ErrorApologyTTS(tts, SPK_DEV, gate_state,
                                  text="Sorry, that didn't go through.")
    pipeline = Pipeline([
        in_transport, input_gate, stt, status, err_apology, user_aggregator, llm,
        tts, out_transport, assistant_aggregator,
    ])
    if bench:
        # explicit rates for the socket transport (16 kHz in, our TTS's 48 kHz
        # out) + token usage so fdbench can report context size per LLM call
        params = PipelineParams(enable_metrics=True, enable_usage_metrics=True,
                                allow_interruptions=True,
                                audio_in_sample_rate=16000, audio_out_sample_rate=48000)
    else:
        params = PipelineParams(enable_metrics=True, allow_interruptions=True)
    task = PipelineTask(pipeline, params=params,
                        cancel_on_idle_timeout=False, idle_timeout_secs=3600)
    task.add_observer(MetricsLogObserver())  # per-turn TTFB/TTFAT/TTFA/TURN/reasoning -> log
    return task, tts, gate_state, pa, status


async def run_bot():
    """The normal service: EMEET audio + spoken greeting."""
    task, tts, gate_state, pa, status = await build_bot()

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