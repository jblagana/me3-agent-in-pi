"""ME2 status page — the command UI for the Pi. Stdlib only (http.server).

Serves the UI + the endpoints pi_demo.py POSTs to:
  GET  /          the handover UI (boots_ui.html next to this file when
                  present, else the built-in dashboard below)
  GET  /debug     the built-in dashboard (history table)
  GET  /state     full JSON (for the dashboard + debugging)
  GET  /ui_state  the handover §7 data model (1 Hz poll for boots_ui.html)
  POST /fire      one fired command (JSON from pi_demo)
  POST /state     {"mode": "idle"|"awake"} heartbeat from pi_demo

Usage (on the Pi, from the package dir):
  python3 boots_ui.py --dir ~/me2 --port 8330
  # laptop browser: http://me2pi.local:8330  (or http://192.168.137.9:8330)
"""
import argparse
import json
import os
import queue
import random
import re
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess

LABELS = {
    # V3 phrasebook (20 classes, src/commands.py) — single source of truth
    "PLAY_MUSIC": "Play music", "VOLUME_UP": "Volume up",
    "VOLUME_DOWN": "Volume down", "NEXT": "Next", "PAUSE": "Pause",
    "STOP": "Stop",
    "LIGHT_ON": "Lights on", "LIGHT_OFF": "Lights off",
    "BRIGHTNESS": "Brightness", "COLOR": "Color",
    "TEMPERATURE": "Temperature",
    "WEATHER": "Weather", "TIME": "Time",
    "TIMER": "Timer", "ALARM": "Alarm",
    "CALL": "Call", "MESSAGE": "Message",
    "CREATE_REMINDER": "Reminder", "LIST_REMINDERS": "Reminders",
    "OUT_OF_SCOPE": "Out of scope",
}

STATE = {
    "lock": threading.Lock(),
    "mode": "idle",
    "mode_ts": time.time(),
    "last": None,
    "history": deque(maxlen=30),
    # Conversation memory: FULL thread of (user, assistant) pairs, SHARED
    # by typed chat AND voice turns (the UI chat is a voice proxy — one
    # pipeline, one context). The UI rebuilds the whole thread from this on
    # refresh (chat persists). The LLM only sees the LAST 5 messages
    # (context window, sliced in _chat_run) — the rest is UI memory only.
    # Cap 200 keeps the list bounded after a long session.
    "chat_history": deque(maxlen=200),
    # Command log (newest last; the UI renders newest-first and rebuilds
    # from this on refresh).
    "cmdlog": deque(maxlen=50),
    "total": 0,
    "started": time.time(),
    "vcm": None,
    "wake": None,
    "last_good": None,  # most recent non-gated fire (drives the done card)
    "last_dim": None,   # {"old","new"} light level of the last dim_lights
    "weather": None,    # {"icon","temp","condition","ts"} open-meteo cache
    "weather_online": False,
    # Live conversation state pushed by the ME3 voice agent (POST /me3).
    # phase: idle|listening|transcribing|thinking|tool|speaking
    "me3": {"phase": "listening", "heard": "", "thinking": "", "response": "",
            "tool": "", "tool_args": "", "tool_result": "", "tool_ms": 0,
            "tool_seq": 0, "turn": 0,
            "cot": "", "cot_round": 0,   # last non-empty thinking per round
            "voice": {"provider": "llm", "model": "llm"},  # voice LLM identity
            "latency": None,             # per-leg voice turn ms {asr,llm,tool,tts,total}
            "active": False, "mic": False,
            "mic_level": 0.0, "last_hb": 0.0, "ts": 0.0},
}

_BOOT_ID = "%d-%d" % (os.getpid(), int(time.time() * 1000))  # per server start
CALL_RING_S = float(os.environ.get("ME2_CALL_RING_S", "8"))  # CALL card ring window
# Lowest volume a command may set: the music volume IS the EMEET's ALSA PCM
# volume, which Boots also speaks through (0 = Boots goes silent; the stress
# test did exactly that, 2026-11-07).
VOL_MIN = int(os.environ.get("ME2_VOL_MIN", "10"))
_PAGE_HTML = None  # boots_ui.html bytes (the ME3 observability dashboard) or None

# ---------------------------------------------------------------- typed chat
# POST /chat — typed conversation (no STT/TTS): the page composer POSTs text,
# we run the real ME3 LLM + tool layer (Groq qwen3.8-27b, same prompt.py and
# tool modules as agent.py) and return the reply. ADDITIVE ONLY: the voice
# agent's /me3, /me3mic, /fire, /ui_state endpoints are untouched, so the Pi
# agent keeps working against this exact file. requests is optional — if it's
# missing the endpoint 503s and everything else in this server is unaffected.
try:
    import requests as _requests
    _CHAT_TOOLS = None
    _CHAT_PROMPT = None
    _CHAT_PROMPT_MOD = None   # prompt module (system_prompt() = base + clock)
    _CHAT_LLM = None
except ImportError:
    _requests = None

# ---------------------------------------------------------------- devices
# The UI process owns the device state (ACTUATION.md: one process, zero
# cloud). Every /fire is dispatched through the table below; the page
# re-renders every second and CSS animates the deltas.

SONGS = ["jetlag", "yellow", "finesse", "multo", "baby"]
# V3 slot taxonomy (src/slots.py): 6 parametric intents x 3 locked values.
V3_TIMER_SEC = {"10 seconds": 10, "30 seconds": 30, "1 minute": 60}
V3_ALARM_MARK = {"6 AM": "06:00", "8 AM": "08:00", "9 PM": "21:00"}
V3_DEG = {"18 degrees": 18, "22 degrees": 22, "26 degrees": 26}
V3_PCT = {"20 percent": 20, "60 percent": 60, "100 percent": 100}
V3_COLOR = {"red": "red", "blue": "blue", "green": "green"}


# --- Free-form slot parsing (2026-11-07 stress-test fixes) -------------------
# The LLM passes the user's own words ("two minutes", "noon", "sixteen",
# "chartreuse"). The old parsers only read digits + seconds/minutes and three
# colours, so they silently did nothing while the tool confirmed success.
# Every parser now returns None/0 when it truly can't read the value, and
# apply_fire reports "invalid" so the tool asks instead of claiming success.
_NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
              "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
              "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
              "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
              "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
              "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def _words_to_digits(text) -> str:
    """'twenty five minutes' -> '25 minutes', 'an hour' -> '1 hour',
    'seven thirty' -> '7 30'. Leaves everything else as-is."""
    t = str(text or "").lower()
    t = re.sub(r"\bhalf\s+an?\s+hour\b", "30 minutes", t)
    t = re.sub(r"\ban?\s+(hour|minute|second)", r"1 \1", t)
    out, acc, prev_tens = [], None, False
    for tok in re.findall(r"[a-z]+|\d+(?:\.\d+)?|[^\sa-z\d]", t):
        if tok in _NUM_WORDS:
            v = _NUM_WORDS[tok]
            if acc is not None and prev_tens and v < 10:
                acc += v
                prev_tens = False
            else:
                if acc is not None:
                    out.append(str(acc))
                acc, prev_tens = v, (v >= 20 and v % 10 == 0)
            continue
        if tok == "hundred" and acc is not None:
            acc *= 100
            prev_tens = False
            continue
        if acc is not None:
            out.append(str(acc))
            acc = None
            prev_tens = False
        out.append(tok)
    if acc is not None:
        out.append(str(acc))
    return " ".join(out)


def _free_secs(slot):
    """'90 seconds' / '1 hour' / 'two minutes' / '1 hour 30 minutes' /
    'half an hour' -> seconds (0 = unreadable). Bare number = seconds.
    Capped at 24 h. (V3 keys still work.)"""
    if slot in V3_TIMER_SEC:
        return V3_TIMER_SEC[slot]
    t = _words_to_digits(slot)
    total, found = 0.0, False
    for n, unit in re.findall(
            r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h\b|minutes?|mins?|m\b|seconds?|secs?|s\b)?", t):
        n = float(n)
        u = unit or ""
        total += n * 3600 if u.startswith("h") else n * 60 if u.startswith("m") else n
        found = True
    return int(min(total, 24 * 3600)) if found else 0


def _timer_secs(slot: str) -> int:
    """TIMER slot -> seconds (any duration; V3 keys fast-path)."""
    return _free_secs(slot)


def _free_mark(slot):
    """'7:30 AM' / '6 pm' / '18:00' / 'noon' / 'midnight' / 'seven thirty am'
    -> 'HH:MM' (None = unreadable). (V3 keys still work.)"""
    if slot in V3_ALARM_MARK:
        return V3_ALARM_MARK[slot]
    t = re.sub(r"(\d)\s*\.\s*(\d)", r"\1:\2", _words_to_digits(slot))   # 7.30 -> 7:30
    t = t.upper().replace(".", "")                                        # p.m. -> PM
    t = re.sub(r"\b([AP])\s+M\b", r"\1M", t)                            # "P M" -> PM
    if "NOON" in t:
        return "12:00"
    if "MIDNIGHT" in t:
        return "00:00"
    m = re.search(r"\b(\d{1,2})(?:\s*[:\s]\s*(\d{2}))?\s*(AM|PM)?\b", t)
    if not m:
        return None
    h = int(m.group(1))
    mm = int(m.group(2) or 0)
    ap = m.group(3)
    if ap == "PM" and h < 12:
        h += 12
    if ap == "AM" and h == 12:
        h = 0
    if h > 23 or mm > 59:
        return None
    return f"{h:02d}:{mm:02d}"


def _alarm_mark(slot: str):
    """ALARM slot -> 'HH:MM' (any time; V3 keys fast-path)."""
    return _free_mark(slot)


TEMP_MIN, TEMP_MAX = 16, 30   # a real thermostat's range (100 was accepted)


def _free_deg(slot):
    """'24 degrees' / 'sixteen' -> int degrees, clamped to TEMP_MIN..TEMP_MAX
    (None = unreadable). (V3 keys still work.)"""
    if slot in V3_DEG:
        return V3_DEG[slot]
    m = re.search(r"(\d+)", _words_to_digits(slot))
    return max(TEMP_MIN, min(TEMP_MAX, int(m.group(1)))) if m else None


def _temp_deg(slot: str):
    """TEMPERATURE slot -> degrees (any value; V3 keys fast-path)."""
    return _free_deg(slot)


_PCT_WORDS = {"off": 0, "dim": 20, "dimmer": 20, "low": 20, "half": 50,
              "medium": 50, "bright": 100, "brighter": 100, "full": 100,
              "max": 100, "maximum": 100}


def _free_pct(slot):
    """'40 percent' / '40%' / 'forty' / 'dim' / 'half' / 'full' -> 0..100
    (None = unreadable). (V3 keys still work.)"""
    if slot in V3_PCT:
        return V3_PCT[slot]
    t = _words_to_digits(slot)
    m = re.search(r"(\d+)", t)
    if m:
        return max(0, min(100, int(m.group(1))))
    for w in re.findall(r"[a-z]+", t):
        if w in _PCT_WORDS:
            return _PCT_WORDS[w]
    return None


def _bright_pct(slot: str):
    """BRIGHTNESS slot -> percent (any value; V3 keys fast-path)."""
    return _free_pct(slot)


# Any CSS named colour (the tool promises "any color name"; it used to accept
# only red/blue/green) + two lamp whites the UI already styles.
_CSS_COLORS = set("""aliceblue antiquewhite aqua aquamarine azure beige bisque black
blanchedalmond blue blueviolet brown burlywood cadetblue chartreuse chocolate coral
cornflowerblue cornsilk crimson cyan darkblue darkcyan darkgoldenrod darkgray darkgreen
darkgrey darkkhaki darkmagenta darkolivegreen darkorange darkorchid darkred darksalmon
darkseagreen darkslateblue darkslategray darkslategrey darkturquoise darkviolet deeppink
deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite forestgreen fuchsia gainsboro
ghostwhite gold goldenrod gray green greenyellow grey honeydew hotpink indianred indigo ivory
khaki lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan
lightgoldenrodyellow lightgray lightgreen lightgrey lightpink lightsalmon lightseagreen
lightskyblue lightslategray lightslategrey lightsteelblue lightyellow lime limegreen linen
magenta maroon mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen
mediumslateblue mediumspringgreen mediumturquoise mediumvioletred midnightblue mintcream
mistyrose moccasin navajowhite navy oldlace olive olivedrab orange orangered orchid
palegoldenrod palegreen paleturquoise palevioletred papayawhip peachpuff peru pink plum
powderblue purple rebeccapurple red rosybrown royalblue saddlebrown salmon sandybrown
seagreen seashell sienna silver skyblue slateblue slategray slategrey snow springgreen
steelblue tan teal thistle tomato turquoise violet wheat white whitesmoke yellow
yellowgreen""".split()) | {"warm", "amber"}


def _lamp_color(slot: str):
    """Colour name -> lamp colour (None = unknown). 'Light Blue' -> 'lightblue',
    'warm white' -> 'warm'."""
    c = re.sub(r"[^a-z]", "", str(slot or "").lower())
    if c in ("warmwhite", "softwhite"):
        c = "warm"
    return c if c in _CSS_COLORS else None


def _timer_word(slot):
    """V3 TIMER slot -> done-card display ('0:10' / '0:30' / '1 min')."""
    s = V3_TIMER_SEC.get(slot, 0)
    if not s:
        return None
    return f"0:{s:02d}" if s < 60 else f"{s // 60} min"


def _alarm_word(slot):
    """V3 ALARM slot -> done-card display ('6:00 AM' / '8:00 AM' / '9:00 PM')."""
    m = V3_ALARM_MARK.get(slot)
    if not m:
        return None
    h, mm = (int(x) for x in m.split(":"))
    return f"{h % 12 or 12}:{mm} {'AM' if h < 12 else 'PM'}"


def _temp_word(slot):
    """V3 TEMPERATURE slot -> '22°' (done-card display)."""
    d = V3_DEG.get(slot)
    return f"{d}\u00b0" if d is not None else None


def _bright_word(slot):
    """V3 BRIGHTNESS slot -> '60%' (done-card display)."""
    p = V3_PCT.get(slot)
    return f"{p}%" if p is not None else None


DEV = {
    "lights": {"on": False, "level": 70, "color": "white"},
    "timer": None,            # {"until": ts, "secs": n, "mins": n/60, "done": bool}
    "alarms": [],
    "temp": {"target": None, "current": 21.5},
    "music": {"playing": False, "paused": False, "track": None, "vol": 60,
              "user_paused": False},
    "reminders": [],
    "call": None,       # {"number", "ts"} of the last CALL (CALL card)
    "chips": deque(maxlen=6),  # (text, ts)
    "speaking_until": 0.0,     # TTS self-trigger window (pi_demo guard)
}
_LAST_FIRE = {}  # (cmd, slot) -> ts, 0.5 s debounce (was 2 s; speaker has
# auto echo cancellation — the scaling self-trigger guard is the real
# double-fire protection, so a fast re-fire of the same command is safe)


def add_chip(text: str):
    DEV["chips"].appendleft((text, time.time()))


# ---------------------------------------------------------------- music
# Real playback (instruction 15, 2026-10-02): the 5 locked songs live in
# <pkg>/music/<title>.wav (48 kHz mono s16 — the EMEET M1A only accepts
# its native 48k output) and are streamed by a CHILD PROCESS,
# boots_music_worker.py (one short-lived process per play segment).
# Why a child: re-opening ALSA output streams while pi_demo.py holds the
# M1A mic input intermittently ABORTS PortAudio at C level (assert in
# PaAlsaStream_Initialize — uncatchable from Python); an in-process
# player took the whole UI down with it (live-verified 2026-10-02).
# The worker is expendable: on pause/switch it is SIGTERMed, and a
# monitor thread maps its exit code to the tile state (0 = natural end,
# anything else = fail-soft reset). No file / no worker deps / dead
# device never break the page.
# NOTE: apply_fire runs under STATE["lock"]; the critical sections here
# stay short (spawn + wav-header read) and never touch STATE["lock"].

MUSIC_DIR = None  # Path, set in main() from --dir
import os as _os
SITES_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "sites")
_os.makedirs(SITES_DIR, exist_ok=True)
_M = {"lock": threading.Lock(), "proc": None, "track": None, "sr": 48000,
      "pos": 0, "t0": 0.0, "dur": 0}
_DUCK = {"awake": False, "defer": False}
# awake: True while the wake-duck holds music paused (ack + command window)
# defer: True when idle arrived but TTS is still playing; the TTS monitor
#        resumes the music once the reply finishes.


def _music_init():
    """Boot check: report the track count + aplay availability. No audio
    is touched until the first play (the worker is spawned lazily)."""
    n = sum(1 for s in SONGS
            if MUSIC_DIR is not None and (MUSIC_DIR / f"{s}.wav").exists())
    import shutil
    aplay_ok = shutil.which("aplay") is not None
    print(f"music: {n}/{len(SONGS)} tracks in {MUSIC_DIR}"
          + ("" if aplay_ok else "  (aplay missing — playback off)"),
          flush=True)
    espeak = _ESPEAK[0] if isinstance(_ESPEAK, (list, tuple)) else _ESPEAK
    print(f"speak: espeak-ng "
          + ("ready" if shutil.which(espeak)
             else "MISSING — apt install espeak-ng (chip-only answers)"),
          flush=True)
    _music_sync_vol_from_hw()
    # Live sync: the M1A speaker has a physical volume knob — turning it
    # moves the HW PCM level without touching the tile. A 1 Hz daemon
    # resyncs the tile when the HW and tile disagree (covers both a
    # manual knob turn AND a VOLUME fire that pushed the HW). The
    # render loop writes rng.value from the tile every poll, so the
    # slider follows automatically.
    threading.Thread(target=_music_vol_sync_loop, daemon=True).start()


def _music_vol_sync_loop() -> None:
    """1 Hz daemon: resync the tile volume from the M1A HW when they
    disagree. Fail-soft: on read error the tile keeps its value."""
    while True:
        time.sleep(1.0)
        try:
            if _music_vol_changed():
                hw = _music_read_hw_vol()
                if hw is not None:
                    DEV["music"]["vol"] = hw
        except Exception:
            pass


def _music_worker():
    return Path(__file__).resolve().parent / "boots_music_worker.py"


def _music_dev():
    """Kept for interface compat — the ALSA worker ignores the device
    index (it hardcodes plughw:CARD=M1A,DEV=0). Always returns None."""
    return None


def _music_meta(path: Path):
    """(sample_rate, duration_seconds) from the wav header."""
    import wave
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        dur = wf.getnframes() / sr if sr else 0.0
    return sr, dur


def _music_kill():
    """SIGTERM the running worker (caller holds _M["lock"])."""
    p, _M["proc"] = _M["proc"], None
    if p is not None:
        try:
            p.terminate()
        except Exception:
            pass


def _music_spawn(track: str, pos: int):
    """Spawn the worker for track from sample pos. Caller holds
    _M["lock"]. Returns True if the process launched."""
    import subprocess
    path = MUSIC_DIR / f"{track}.wav"
    dev = _music_dev()
    argv = [sys.executable, str(_music_worker()), str(path), str(pos)]
    if dev is not None:
        argv.append(str(dev))
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"music: spawn failed ({track}): {e}", file=sys.stderr)
        return False
    _M["proc"] = proc
    try:
        _M["sr"], _M["dur"] = _music_meta(path)
    except Exception:
        _M["sr"], _M["dur"] = 48000, 0.0
    _M["t0"] = time.monotonic()
    return True


def _music_monitor(track: str):
    """Daemon thread: wait for a worker to exit and map it to tile
    state. exit 0 = natural end; other = the worker died (fail-soft).
    Never raises."""
    p = _M["proc"]
    if p is None:
        return
    p.wait()
    rc = p.returncode
    with _M["lock"]:
        stale = (_M["proc"] is not p or _M["track"] != track
                 or DEV["music"].get("paused"))
        if not stale:
            DEV["music"]["playing"] = False
            DEV["music"]["paused"] = False
    if stale:
        return  # paused / replaced — state is owned by that command
    if rc == 0:
        add_chip(f"“{track}” finished")
    else:
        print(f"music: worker died (rc {rc}) — tile reset", file=sys.stderr)
        add_chip(f"music: “{track}” stopped (player error)")


def _music_play(track: str) -> bool:
    """Start a track from the top (replacing anything playing). True if
    the worker launched; state must not claim playing until it does."""
    with _M["lock"]:
        if MUSIC_DIR is None or not (MUSIC_DIR / f"{track}.wav").exists():
            return False
        _music_kill()
        _M["track"] = track
        _M["pos"] = 0
        if not _music_spawn(track, 0):
            _M["track"] = None
            return False
    threading.Thread(target=_music_monitor, args=(track,),
                     daemon=True).start()
    return True


def _music_pause() -> bool:
    """Kill the worker at the current wall-clock position; resume
    relaunches it from that sample offset."""
    with _M["lock"]:
        if _M["proc"] is None or _M["track"] is None:
            return False
        _M["pos"] = _M["pos"] + int((time.monotonic() - _M["t0"])
                                    * _M["sr"])
        _music_kill()
    DEV["music"]["paused"] = True
    return True


def _music_resume() -> bool:
    """Relaunch the worker from the paused sample position."""
    with _M["lock"]:
        if _M["track"] is None or MUSIC_DIR is None:
            return False
        track = _M["track"]
        pos = _M["pos"]
        if not _music_spawn(track, pos):
            DEV["music"]["paused"] = False
            return False
        DEV["music"]["paused"] = False
    threading.Thread(target=_music_monitor, args=(track,),
                     daemon=True).start()
    return True



def _music_sync_vol_from_hw() -> None:
    """Read the M1A PCM hardware volume and set the tile to match.
    Called at boot so the first play comes out at the speaker's actual
    level, not a stale default. Fail-soft: on any error the tile keeps
    its default (60)."""
    try:
        import re as _re
        r = subprocess.run(
            ["amixer", "-c", "M1A", "sget", "PCM"],
            capture_output=True, text=True, timeout=3,
        )
        m = _re.search(r"Playback\s+(\d+)\s+\[(\d+)%\]", r.stdout)
        if m:
            hw_vol = int(m.group(2))
            DEV["music"]["vol"] = hw_vol
            print(f"music: tile vol synced to hw ({hw_vol}%)", flush=True)
    except Exception:
        pass  # amixer missing / M1A gone — keep default


def _apply_alsa_volume(vol: int) -> None:
    """Push the music volume to the M1A ALSA PCM control (0-100 %).
    Called on every VOLUME_UP / VOLUME_DOWN / SET_VOLUME so the speaker
    actually changes level, not just the tile readout. Fail-soft: if
    amixer is missing or the M1A is wedged, the tile state still
    updates."""
    try:
        subprocess.run(
            ["amixer", "-c", "M1A", "set", "PCM", f"{vol}%"],
            capture_output=True, timeout=3,
        )
    except Exception:
        pass  # amixer missing / M1A gone — tile still reflects the value


def _music_read_hw_vol() -> int | None:
    """Read the M1A PCM hardware volume (0-100). None on any error
    (amixer missing / M1A wedged)."""
    try:
        r = subprocess.run(
            ["amixer", "-c", "M1A", "sget", "PCM"],
            capture_output=True, text=True, timeout=3,
        )
        m = re.search(r"Playback\s+(\d+)\s+\[(\d+)%\]", r.stdout)
        return int(m.group(2)) if m else None
    except Exception:
        return None


def _music_vol_changed() -> bool:
    """True if the tile volume and the M1A hardware volume disagree.
    A manual speaker-knob turn (or a VOLUME fire) moves the HW; the tile
    only follows if we resync. Fail-soft: on read error, no change."""
    hw = _music_read_hw_vol()
    return hw is not None and hw != DEV["music"]["vol"]

# ---------------------------------------------------------------- speak
# Spoken replies (instruction 16, 2026-10-02): the live v2e model merged
# ask_time + ask_weather into ONE slot-less `ask_question` class, so
# Boots can't tell which was asked — it answers BOTH (time + weather).
# Synthesis runs espeak-ng (apt, offline, robotic but clear) as a CHILD
# that writes a temp WAV; espeak's 22.05 kHz output is resampled to
# 48 kHz (the M1A only accepts its native 48k — instruction 15) with a
# local numpy import (guarded; the venv has it — the worker needs it);
# playback reuses the crash-isolated boots_music_worker.py child (same
# rationale as music: the ALSA/PortAudio abort can kill the child,
# never the UI). Music ducks: pause -> speak -> resume. espeak-ng
# missing = the chip still answers, voice skipped. piper (natural voice)
# is the post-defense upgrade: it means an onnxruntime version risk in
# the shared venv — not on defense eve.
_ESPEAK = "espeak-ng"  # argv[0]; tests may set [sys.executable, stub.py]
# TTS self-trigger window (from TTS start): the last wake window that can
# contain the reply is ~reply (<=4.2 s) + 1 s (wake stride); a real wake
# just before the reply captures <=3.5 s into it and fires ~3.6 s in —
# 6.5 s covers both (live-verified 10-02: an 8 s window expired MID-CHAIN
# and a fake media_control conf 0.995 got through).
_SPEAK_GUARD_S = 6.5
_TTS = {"lock": threading.Lock(), "proc": None}


def _time_words(t) -> str:
    """localtime -> 'seven forty two p m' (espeak text; no digits)."""
    h = t.tm_hour % 12 or 12
    hw = ["one", "two", "three", "four", "five", "six", "seven",
          "eight", "nine", "ten", "eleven", "twelve"][h - 1]
    ap = "a m" if t.tm_hour < 12 else "p m"
    m = t.tm_min
    if m == 0:
        return f"{hw} o'clock {ap}"
    _ones = ["", "one", "two", "three", "four", "five", "six",
             "seven", "eight", "nine"]
    _teens = ["ten", "eleven", "twelve", "thirteen", "fourteen",
              "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    _tens = ["", "", "twenty", "thirty", "forty", "fifty"]
    if m < 10:
        mw = _ones[m]
    elif m < 20:
        mw = _teens[m - 10]
    else:
        mw = _tens[m // 10] + (" " + _ones[m % 10] if m % 10 else "")
    return f"{hw} {mw} {ap}"


def _int_words(n) -> str:
    """0-99 int -> words (weather temp speak text; no digits)."""
    n = int(round(n))
    _ones = ["", "one", "two", "three", "four", "five", "six",
             "seven", "eight", "nine", "ten", "eleven", "twelve",
             "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
             "eighteen", "nineteen"]
    _tens = ["", "ten", "twenty", "thirty", "forty", "fifty",
             "sixty", "seventy", "eighty", "ninety"]
    if n < 20:
        return _ones[n]
    return _tens[n // 10] + (" " + _ones[n % 10] if n % 10 else "")


def _time_reply():
    """(chip text, speak text) for ask_time — time only, live local
    (Pi TZ = Asia/Manila, NTP-synced)."""
    t = time.localtime()
    h = t.tm_hour % 12 or 12
    ap = "AM" if t.tm_hour < 12 else "PM"
    return (f"{h}:{t.tm_min:02d} {ap}", _time_words(t))


def _weather_reply():
    """(chip text, speak text) for ask_weather — weather only, live
    open-meteo, spoken with the location name (ME2_LOCATION) so the
    answer says WHERE the number is from."""
    w = STATE.get("weather")
    if w and w.get("temp") is not None and STATE.get("weather_online"):
        cond = str(w.get("condition") or "all good").strip().capitalize()
        return (f"{ME2_LOCATION}: {cond}, {w['temp']:.0f}°",
                f"In {ME2_LOCATION}, {cond}, "
                f"{_int_words(w['temp'])} degrees")
    if w and w.get("temp") is not None:
        # Offline but the last fetch is <=15 min old: the caller speaks
        # apology + stale-lead clips, then this cached data as the tail.
        cond = str(w.get("condition") or "all good").strip().capitalize()
        return (f"{ME2_LOCATION}: {cond}, {w['temp']:.0f}° (15 min old)",
                f"In {ME2_LOCATION}, {cond}, "
                f"{_int_words(w['temp'])} degrees")
    return ("Sorry, no internet connection.",
            "Sorry, no internet connection.")


def _ask_reply():
    """(chip text, speak text) for the merged ask_question class (current
    live model): answer BOTH time + weather. The new model ships separate
    ask_time / ask_weather classes -> handled separately in apply_fire."""
    chip_t, speak_t = _time_reply()
    chip_w, speak_w = _weather_reply()
    return (f"{chip_t}. {chip_w}.", f"{speak_t}. {speak_w}.")


def _speak_resample48(path: str) -> str:
    """espeak-ng outputs 22.05 kHz; the M1A only accepts 48 kHz.
    Upsample with linear interp (upsample-only -> no aliasing) into a
    sibling temp wav. Raises on any problem (caller fails soft)."""
    import wave
    import numpy as np
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        arr = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if sr == 48000 or len(arr) == 0:
        return path
    out_path = path + ".48k"
    n = int(len(arr) * 48000.0 / sr)
    x = np.linspace(0.0, len(arr) - 1, n)
    out = np.interp(x, np.arange(len(arr)), arr).astype(np.int16)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(out.tobytes())
    return out_path


EDGE_VOICE = "en-US-RogerNeural"  # cloud TTS (edge-tts, 2026-10-03)
EDGE_TIMEOUT_S = 12


def _speak_synth_edge(text: str):
    """edge-tts (cloud) -> 48k temp wav path (or None). Never raises.
    Primary voice: en-US-RogerNeural. Needs internet; on any failure the
    caller falls back to espeak-ng (offline). MP3 -> 48k mono WAV via
    miniaudio (in the venv) ??? the M1A worker only plays WAV."""
    import asyncio
    import os
    import wave
    try:
        import edge_tts
        import numpy as np
    except Exception as e:
        print(f"speak: edge-tts unavailable: {e}", file=sys.stderr)
        return None
    mp3 = str(Path(tempfile.gettempdir()) / "me2_say.mp3")
    wav = str(Path(tempfile.gettempdir()) / "me2_say_edge.wav")
    try:
        async def _run():
            await asyncio.wait_for(
                edge_tts.Communicate(text, EDGE_VOICE).save(mp3),
                timeout=EDGE_TIMEOUT_S)
        asyncio.run(_run())
        if not (Path(mp3).exists() and Path(mp3).stat().st_size > 0):
            return None
        import miniaudio
        dec = miniaudio.decode_file(mp3)
        sr = int(dec.sample_rate)
        nch = int(getattr(dec, "nchannels", 1))
        frames = np.frombuffer(dec.samples, dtype=np.int16)
        if nch > 1 and len(frames):
            frames = frames.reshape(-1, nch).mean(axis=1)  # stereo -> mono
        if sr != 48000 and len(frames):
            n = int(len(frames) * 48000.0 / sr)
            x = np.linspace(0.0, len(frames) - 1, n)
            frames = np.interp(x, np.arange(len(frames)), frames)
        i16 = frames.astype(np.int16)
        with wave.open(wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(i16.tobytes())
        return wav
    except Exception as e:
        print(f"speak: edge-tts failed ({e}) ??? falling back to espeak-ng",
              file=sys.stderr)
        return None


def _speak_synth(text: str):
    """48k temp wav path (or None). Never raises. Primary: edge-tts
    (en-US-RogerNeural, cloud); fallback: espeak-ng (offline, robotic)."""
    p = _speak_synth_edge(text)
    if p is not None:
        return p
    import subprocess
    wav = str(Path(tempfile.gettempdir()) / "me2_say.wav")
    prefix = _ESPEAK if isinstance(_ESPEAK, (list, tuple)) else [_ESPEAK]
    try:
        r = subprocess.run(
            [*prefix, "-v", "en-us", "-s", "175", "-w", wav, text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10)
        if r.returncode == 0 and Path(wav).exists():
            return _speak_resample48(wav)
    except Exception as e:
        print(f"speak: synth failed: {e}", file=sys.stderr)
    return None


# ---------------------------------------------------------------- clips
# Pre-recorded reply clips (en-US-RogerNeural, 48k mono s16 WAVs in
# ./clips/, generated 2026-10-05): instant, offline, zero synth latency.
# Every command fires its clip first; a missing clip file degrades to the
# plain-TTS fallback, never to silence.
CLIP_DIR = Path(__file__).resolve().parent / "clips"
CLIP = {
    "PLAY_MUSIC": "play_music", "PAUSE": "pause", "RESUME": "resume",
    "NEXT": "next", "STOP": "stop", "VOLUME_UP": "volume_up",
    "VOLUME_DOWN": "volume_down", "LIGHT_ON": "light_on",
    "LIGHT_OFF": "light_off", "CALL": "call", "MESSAGE": "message",
    "OUT_OF_SCOPE": "oos", "TIME": "time_lead", "WEATHER": "weather_lead",
    "BRIGHTNESS": {"20 percent": "brightness_20",
                   "60 percent": "brightness_60",
                   "100 percent": "brightness_100"},
    "COLOR": {"red": "color_red", "blue": "color_blue",
              "green": "color_green"},
    "TEMPERATURE": {"18 degrees": "temp_18", "22 degrees": "temp_22",
                    "26 degrees": "temp_26"},
    "TIMER": {"10 seconds": "timer_10", "30 seconds": "timer_30",
              "1 minute": "timer_60"},
    "ALARM": {"6 AM": "alarm_6am", "8 AM": "alarm_8am",
              "9 PM": "alarm_9pm"},
    "CREATE_REMINDER": {"drink water": "reminder_drink",
                        "study": "reminder_study",
                        "exercise": "reminder_exercise"},
}


def _clip_name(cmd: str, slot: str) -> str | None:
    """(cmd, slot) -> main clip base name (no .wav), or None => plain TTS."""
    m = CLIP.get(cmd)
    if m is None:
        return None
    if isinstance(m, dict):
        return m.get(slot)
    return m


def _clip_path(name: str) -> str | None:
    p = CLIP_DIR / f"{name}.wav"
    return str(p) if p.exists() else None


def _speak_spawn(segs: list, est_s: float) -> bool:
    """Play segs = [(wav_path | None, text | None), ...] in order through
    the M1A worker (None wav = synthesize text via edge-tts/espeak at
    spawn time). Music ducks; the self-trigger guard scales with the
    estimated total reply length (clips are 1.5-3.5 s each; a 2-clip
    reply + TTS tail runs past the old fixed 6.5 s window)."""
    import subprocess
    wavs = []
    for wav, text in segs:
        if wav is None:
            wav = _speak_synth(text)
            if wav is None:
                return False
        wavs.append(wav)
    with _TTS["lock"]:
        p = _TTS["proc"]
        if p is not None and p.poll() is None:
            return False  # already talking — don't stack replies
        ducked = False
        if DEV["music"]["playing"] and not DEV["music"]["paused"]:
            ducked = _music_pause()
        dev = _music_dev()
        argv = [sys.executable, str(_music_worker()), *wavs, "0"]
        if dev is not None:
            argv.append(str(dev))
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"speak: spawn failed: {e}", file=sys.stderr)
            if ducked:
                _music_resume()
            return False
        _TTS["proc"] = proc
        STATE["speaking_until"] = time.time() + max(
            _SPEAK_GUARD_S, est_s + 2.5)
    threading.Thread(target=_speak_monitor, args=(ducked,),
                     daemon=True).start()
    return True


def _clip_dur_s(path: str) -> float:
    try:
        import wave
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 2.0


def speak_clips(names: list, tail: str | None = None) -> bool:
    """Pre-recorded reply: play the named clips (base names, no .wav) in
    order, then an optional TTS tail (live data: time / weather). Every
    named clip missing from disk => plain TTS of the tail (old behavior);
    nothing to say => no voice (chip only). Called from apply_fire
    (STATE["lock"] held). Never raises."""
    try:
        paths = [_clip_path(n) for n in names if n]
        if names and not all(paths):
            paths = []  # partial clip set => full TTS fallback
        segs = [(p, None) for p in paths]
        est = sum(_clip_dur_s(p) for p in paths)
        if tail:
            segs.append((None, tail))
            est += 4.5
        if not segs:
            return False
        return _speak_spawn(segs, est)
    except Exception as e:
        print(f"speak_clips: {e}", file=sys.stderr)
        return False


def speak(text: str) -> bool:
    """Plain TTS reply out loud on the M1A (kept for future free-text
    answers). Same duck/guard/monitor machinery as speak_clips()."""
    return _speak_spawn([(None, text)], 4.5)


def _speak_monitor(ducked: bool):
    """Daemon thread: wait for the TTS worker, clear the slot, resume the
    ducked music. Two paths: (a) speak() ducked the music itself (boss
    said a command while music was playing, no wake-duck), (b) the
    wake-duck paused the music and the /state idle handler deferred the
    resume to this monitor (TTS was still playing when idle arrived).
    Never raises."""
    p = _TTS["proc"]
    if p is None:
        return
    p.wait()
    with _TTS["lock"]:
        if _TTS["proc"] is p:
            _TTS["proc"] = None
    should_resume = ducked or _DUCK.get("defer")
    if (should_resume and DEV["music"].get("paused")
            and not DEV["music"].get("user_paused") and _M["proc"] is None):
        if not _music_resume():
            DEV["music"]["paused"] = False
        else:
            _DUCK["awake"] = False
            _DUCK["defer"] = False
    if p.returncode != 0:
        print(f"speak: worker died (rc {p.returncode}) — reply not spoken",
              file=sys.stderr)


def apply_fire(f: dict) -> str:
    """Dispatch one fire (V3, 20 classes) to device state. Returns outcome.

    V3 phrasebook (src/commands.py): 19 intents + OUT_OF_SCOPE rejection.
    Slot-bearing (PARAMETRIC, src/slots.py): TIMER / ALARM / TEMPERATURE /
    BRIGHTNESS / COLOR / CREATE_REMINDER — each with 3 locked values.
    """
    now = time.time()
    cmd = f.get("cmd")
    slot = f.get("slot")
    conf = f.get("conf", 0.0)

    if cmd == "OUT_OF_SCOPE":
        # Learned rejection (20th class) — the model's no-action path.
        # Distinct from the confidence gate: the MODEL said "not a command".
        add_chip(f"out of scope ({conf:.2f})")
        speak_clips([_clip_name(cmd, slot)])
        return "oos"
    if conf < 0.50:  # confidence gate (ACTUATION)
        add_chip(f"heard something ({conf:.2f}) — below gate")
        return "gated"
    if cmd != "SET_VOLUME" and now - _LAST_FIRE.get((cmd, slot), 0.0) < 0.5:
        return "debounced"
    _LAST_FIRE[(cmd, slot)] = now

    # --- music control ---
    if cmd == "PLAY_MUSIC":
        m = DEV["music"]
        track = random.choice(SONGS)
        if m["playing"] and not m["paused"] and track == m["track"]:
            return "noop"  # same-song repeat (ACTUATION)
        m.update(playing=True, paused=False, track=track, user_paused=False)
        _DUCK.update(awake=False, defer=False)  # new track, fresh state
        if not _music_play(track):  # real audio (instruction 15)
            m["playing"] = False  # fail-soft: don't claim it's playing
            m["track"] = "nothing playing"
            add_chip(f"music: no audio for “{track}”")
            return "nomedia"      # the tool says so instead of "done" (2026-11-07)
        speak_clips([_clip_name(cmd, slot)])
        return f"fired:{track}"
    elif cmd == "VOLUME_UP":
        DEV["music"]["vol"] = min(100, DEV["music"]["vol"] + 20)
        _apply_alsa_volume(DEV["music"]["vol"])
        add_chip(f"volume {DEV['music']['vol']}")
        speak_clips([_clip_name(cmd, slot)])
        return f"fired:{DEV['music']['vol']}"
    elif cmd == "VOLUME_DOWN":
        # floor = VOL_MIN: this is the SAME ALSA volume Boots speaks with, so
        # 0 would leave the user unable to hear Boots at all (2026-11-07).
        DEV["music"]["vol"] = max(VOL_MIN, DEV["music"]["vol"] - 20)
        _apply_alsa_volume(DEV["music"]["vol"])
        add_chip(f"volume {DEV['music']['vol']}")
        speak_clips([_clip_name(cmd, slot)])
        return f"fired:{DEV['music']['vol']}"
    elif cmd == "SET_VOLUME":
        # UI slider: set the M1A HW volume to the exact value. The
        # 1 Hz sync loop keeps the tile in lockstep with the HW.
        # (2026-11-07) number words ("fifty") parsed; unreadable -> "invalid"
        # (ask again). It used to fall back silently and "fifty" set 10 %.
        try:
            v = int(slot)
        except (TypeError, ValueError):
            m_ = re.search(r"-?\d+", _words_to_digits(slot))
            if not m_:
                return "invalid"
            v = int(m_.group(0))
        v = max(VOL_MIN, min(100, v))   # never mute Boots' own voice (see VOL_MIN)
        DEV["music"]["vol"] = v
        _apply_alsa_volume(v)
        add_chip(f"volume {v}")
        return f"fired:{v}"
    elif cmd == "NEXT":
        m = DEV["music"]
        if not m["track"]:
            add_chip("nothing playing yet")
            return "nothing"
        else:
            nxt = (SONGS[(SONGS.index(m["track"]) + 1) % len(SONGS)]
                   if m["track"] in SONGS else random.choice(SONGS))
            if not _music_play(nxt):          # no audio file: don't fake it
                m.update(playing=False, track="nothing playing")
                return "nomedia"
            m.update(playing=True, paused=False, track=nxt)
            speak_clips([_clip_name(cmd, slot)])
            return f"fired:{nxt}"
    elif cmd == "PAUSE":
        # ONE-WAY (2026-10-03): "pause music" always stops, never resumes.
        # It was a toggle before — the wake-duck pauses the music on "hey
        # boots", so a spoken "pause" then hit the resume branch and PLAYED
        # the music (reproduced live: play -> awake/duck -> pause => playing).
        # To bring it back, say "resume music" (RESUME below).
        # user_paused stops the idle-handler's auto-resume (which is meant
        # only for the wake-duck, not a deliberate user pause).
        m = DEV["music"]
        if m["playing"] and not m["paused"]:
            _music_pause()
            m.update(playing=False, paused=True, user_paused=True)
            speak_clips([_clip_name(cmd, slot)])
        elif m["paused"]:
            m.update(playing=False, user_paused=True)
            speak_clips([_clip_name(cmd, slot)])
        else:
            add_chip("nothing to pause")
            return "nothing"     # (2026-11-07) was spoken + confirmed as "Music paused."
    elif cmd == "RESUME":
        # "resume music" — relaunch from the paused sample offset.
        m = DEV["music"]
        if m["track"] and not (m["playing"] and not m["paused"]):
            if _music_resume():
                m.update(playing=True, paused=False, user_paused=False)
                _DUCK.update(awake=False, defer=False)
                speak_clips([_clip_name(cmd, slot)])
            else:
                m.update(paused=False, user_paused=False)
                add_chip("nothing to resume")
                return "nothing"
        else:
            add_chip("nothing to resume")
            return "nothing"
    elif cmd == "STOP":
        m = DEV["music"]
        # (2026-11-07) "nothing playing" is the placeholder a failed play/next
        # leaves behind, not a song: stopping it was confirmed as "Music stopped."
        if m["track"] and m["track"] != "nothing playing":
            with _M["lock"]:
                _music_kill()
                _M["track"] = None
                _M["pos"] = 0
            m.update(playing=False, paused=False, track=None, user_paused=False)
            speak_clips([_clip_name(cmd, slot)])
        else:
            add_chip("nothing playing")
            return "nothing"
    # --- lighting ---
    elif cmd == "LIGHT_ON":
        L = DEV["lights"]; L["on"] = True
        if not L["level"]:
            L["level"] = 70
        speak_clips([_clip_name(cmd, slot)])
    elif cmd == "LIGHT_OFF":
        DEV["lights"]["on"] = False
        speak_clips([_clip_name(cmd, slot)])
    elif cmd == "BRIGHTNESS":
        # (2026-11-07) "invalid" / "fired:<applied>" outcomes: the tool's
        # spoken confirmation now reports what was ACTUALLY applied, and an
        # unreadable value is asked about instead of confirmed.
        p = _bright_pct(slot)
        if p is None:
            return "invalid"
        L = DEV["lights"]; L["on"] = True; L["level"] = p
        speak_clips([_clip_name(cmd, slot)])
        return f"fired:{p}"
    elif cmd == "COLOR":
        c = _lamp_color(slot)
        if not c:
            return "invalid"
        L = DEV["lights"]; L["color"] = c; L["on"] = True
        speak_clips([_clip_name(cmd, slot)])
        return f"fired:{c}"
    # --- temperature ---
    elif cmd == "TEMPERATURE":
        d = _temp_deg(slot)
        if d is None:
            return "invalid"
        DEV["temp"]["target"] = d
        speak_clips([_clip_name(cmd, slot)])
        return f"fired:{d}"
    # --- information ---
    elif cmd == "TIME":
        # Pre-record lead-in (Roger) + TTS tail (live clock).
        chip, speak_text = _time_reply()
        add_chip(chip)
        speak_clips([_clip_name(cmd, slot)], tail=speak_text)
    elif cmd == "WEATHER":
        chip, speak_text = _weather_reply()
        add_chip(chip)
        if STATE.get("weather_online"):
            # All-TTS reply (2026-10-06): no lead clip, no glue — weather
            # is internet-dependent anyway, so the TTS round-trip is
            # already in the path. speak_text carries the location itself
            # ("In U.P. Diliman, sunny, 29 degrees") — no prefix here.
            speak_clips([], tail=speak_text)
        else:
            # Offline: default clip only — no glue, no stale tail.
            speak_clips(["no_internet"])
    # --- timers & alarms ---
    elif cmd == "TIMER":
        secs = _timer_secs(slot)
        if not secs:
            return "invalid"
        DEV["timer"] = {"until": now + secs, "secs": secs,
                        "mins": secs / 60.0, "done": False}
        speak_clips([_clip_name(cmd, slot)])
        return f"fired:{secs}"
    elif cmd == "ALARM":
        mark = _alarm_mark(slot)
        if not mark:
            return "invalid"
        if mark not in DEV["alarms"]:
            DEV["alarms"].append(mark)
            DEV["alarms"].sort()
            DEV["alarms"] = DEV["alarms"][:20]
            speak_clips([_clip_name(cmd, slot)])
        return f"fired:{mark}"
    # --- communication ---
    elif cmd == "CALL":
        if not str(slot or "").strip():
            return "invalid"          # no number: ask, don't "ring nobody"
        add_chip("dialing…")
        # Call state for the CALL card (2026-11-07): the card's ring/shake
        # animation keyed on the disabled purple tool pulse and the server
        # kept no number/time, so it never animated and its status stayed "–".
        DEV["call"] = {"number": str(slot or "").strip()[:40], "ts": now}
        speak_clips([_clip_name(cmd, slot)])
    elif cmd == "MESSAGE":
        add_chip("opening messages…")
        speak_clips([_clip_name(cmd, slot)])
    # --- reminders (structured: {text, done, ts}) ---
    elif cmd == "CREATE_REMINDER":
        if not str(slot or "").strip():
            return "invalid"          # empty reminder: ask what it should say
        if slot:
            if not any(r["text"] == slot for r in DEV["reminders"]):
                DEV["reminders"].insert(0,
                    {"text": slot, "done": False, "ts": time.time()})
                DEV["reminders"] = DEV["reminders"][:20]
                speak_clips([_clip_name(cmd, slot)])
    elif cmd == "LIST_REMINDERS":
        open_r = [r["text"] for r in DEV["reminders"] if not r["done"]]
        add_chip("reminders: " + (", ".join(open_r) if open_r else "none") + "…")
    else:
        add_chip(f"unhandled command: {cmd}")
    return "fired"


def dev_view(now: float) -> dict:
    """Server-side view of device state for /state (with expiries)."""
    t = DEV["timer"]
    if t and not t["done"] and now >= t["until"]:
        t["done"] = True
        add_chip(f"timer's up ({t['secs']} s)")
    remaining = None
    frac = None
    if t:
        left = max(0.0, t["until"] - now)
        remaining = int(left)
        frac = 1.0 - (left / (t["mins"] * 60.0)) if t["mins"] else 1.0
    cur = DEV["temp"]["current"]
    tgt = DEV["temp"]["target"]
    # Simulated HVAC: drift the room toward the target so the mode actually
    # reaches 'holding' — without this the 'cooling/heating' (and the card's
    # breathing glow) never ends, because current never moves.
    if tgt is not None and abs(tgt - cur) > 0.05:
        step = 0.15 * (1.0 if tgt > cur else -1.0)
        cur = tgt if abs(tgt - cur) <= step else round(cur + step, 2)
        DEV["temp"]["current"] = cur
    mode = "idle"
    if tgt is not None:
        mode = "heating" if tgt > cur + 0.25 else \
            ("cooling" if tgt < cur - 0.25 else "holding")
    m = DEV["music"]
    return {
        "lights": dict(DEV["lights"]),
        "timer": ({"remaining": remaining, "frac": round(frac, 4),
                   "done": t["done"], "mins": t["mins"],
                   "secs": t.get("secs", int(t["mins"] * 60))}
                  if t else None),
        "alarms": list(DEV["alarms"]),
        "temp": {"current": round(cur, 1), "target": tgt, "mode": mode},
        "music": {**m, "eq": m["playing"] and not m["paused"]},
        "reminders": [dict(r) for r in DEV["reminders"]],
        "chips": [(c, ts) for c, ts in DEV["chips"] if now - ts < 60],
    }


def load_meta(pkg_dir: Path):
    try:
        m = json.loads((pkg_dir / "meta.json").read_text())
        STATE["vcm"] = {"model": m.get("model"),
                        "classes": len(m.get("classes", []))}
    except Exception:
        pass
    try:
        w = json.loads((pkg_dir / "wake_meta.json").read_text())
        STATE["wake"] = {"model": w.get("model"),
                         "threshold": w.get("threshold")}
    except Exception:
        pass


def state_json() -> dict:
    with STATE["lock"]:
        return {
            "mode": STATE["mode"], "mode_ts": STATE["mode_ts"],
            "last": STATE["last"],
            "history": list(STATE["history"]),
            "total": STATE["total"],
            "started": STATE["started"],
            "vcm": STATE["vcm"], "wake": STATE["wake"],
            "dev": dev_view(time.time()),
            "now": time.time(),
        }


# ---------------------------------------------------------------- weather
# open-meteo, stdlib only, 15-min cache. Fail-soft: no internet (Pi on a
# phone hotspot) -> last good value, or None until the first success.
# Location is BOSS-SET, not geocoded (Opt 1, 2026-10-06): the exact point
# is hardcoded here (U.P. Diliman campus) and overridable via env vars
# ME2_LOCATION / ME2_LAT / ME2_LON. The name is spoken back in the reply
# so the answer is specific about WHERE the number is from.
ME2_LOCATION = os.environ.get("ME2_LOCATION", "U.P. Diliman")
WMO = {
    0: (chr(0x2600)+chr(0xFE0F), "clear"),
    1: (chr(0x2600)+chr(0xFE0F), "mainly clear"),
    2: (chr(0x26C5), "partly cloudy"),
    3: (chr(0x2601)+chr(0xFE0F), "overcast"),
    45: (chr(0x1F32B)+chr(0xFE0F), "fog"),
    48: (chr(0x1F32B)+chr(0xFE0F), "rime fog"),
    51: (chr(0x1F327)+chr(0xFE0F), "light drizzle"),
    53: (chr(0x1F327)+chr(0xFE0F), "drizzle"),
    55: (chr(0x1F327)+chr(0xFE0F), "heavy drizzle"),
    56: (chr(0x1F328)+chr(0xFE0F), "freezing drizzle"),
    57: (chr(0x1F328)+chr(0xFE0F), "freezing drizzle"),
    61: (chr(0x1F327)+chr(0xFE0F), "light rain"),
    63: (chr(0x1F327)+chr(0xFE0F), "rain"),
    65: (chr(0x1F328)+chr(0xFE0F), "heavy rain"),
    66: (chr(0x1F328)+chr(0xFE0F), "freezing rain"),
    67: (chr(0x1F328)+chr(0xFE0F), "freezing rain"),
    71: (chr(0x2744)+chr(0xFE0F), "light snow"),
    73: (chr(0x2744)+chr(0xFE0F), "snow"),
    75: (chr(0x2744)+chr(0xFE0F), "heavy snow"),
    77: (chr(0x2744)+chr(0xFE0F), "snow grains"),
    80: (chr(0x1F326)+chr(0xFE0F), "showers"),
    81: (chr(0x1F326)+chr(0xFE0F), "showers"),
    82: (chr(0x1F326)+chr(0xFE0F), "heavy showers"),
    85: (chr(0x1F328)+chr(0xFE0F), "snow showers"),
    86: (chr(0x1F328)+chr(0xFE0F), "snow showers"),
    95: (chr(0x26C8), "thunderstorm"),
    96: (chr(0x26C8), "thunderstorm"),
    99: (chr(0x26C8), "thunderstorm + hail"),
}


def _weather_fetch():
    import urllib.request
    # Exact campus point (14.6532, 121.0806) — the geocoder would return
    # Quezon City's centroid, not the campus; boss-set, env-overridable.
    lat = os.environ.get("ME2_LAT", "14.6532")   # U.P. Diliman, Quezon City
    lon = os.environ.get("ME2_LON", "121.0806")
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           "&current=temperature_2m,weather_code")
    with urllib.request.urlopen(url, timeout=6) as r:
        d = json.loads(r.read().decode())
    cur = d["current"]
    icon, cond = WMO.get(int(cur["weather_code"]), ("\u2753", "unknown"))
    STATE["weather"] = {"icon": icon,
                        "temp": round(float(cur["temperature_2m"])),
                        "condition": cond, "ts": time.time(),
                        "place": os.environ.get("ME2_PLACE", "U.P. Diliman")}
    STATE["weather_online"] = True


def _weather_loop():
    while True:
        try:
            _weather_fetch()
        except Exception:
            STATE["weather_online"] = False  # no internet / rate limit
        time.sleep(900)


def _ui_music(dev: dict) -> dict:
    """Music tile payload for /ui_state: track, playing, plus the
    progress-bar data (elapsed seconds, duration, fraction 0..1).
    The UI derives its equalizer animation from `playing` and the bar
    width from `fraction` — both are server-authoritative so the tile
    can't drift from the worker. While paused the bar HOLDS its position
    (the frozen sample offset), it does not snap to zero."""
    m = dev["music"]
    playing = bool(m["eq"])
    with _M["lock"]:
        track_loaded = _M["track"] is not None
        dur = _M["dur"] if track_loaded else 0.0
        sr = _M["sr"]
        if playing and _M["proc"] is not None and _M["proc"].poll() is None:
            # live — advance from the segment start (mirrors _music_progress)
            elapsed = _M["pos"] / sr + (time.monotonic() - _M["t0"])
        elif track_loaded:  # paused — freeze at the stored sample offset
            elapsed = _M["pos"] / sr if sr else 0.0
        else:
            elapsed = 0.0
        if dur <= 0:
            frac = 0.0
        else:
            frac = max(0.0, min(1.0, elapsed / dur))
            elapsed = min(elapsed, dur)
    # (2026-11-07) Spotify on the card: with no local song loaded, show what
    # Spotify is doing (Boots plays music there now) instead of "nothing playing".
    sp = _SPOT.get("state")
    if not track_loaded and not playing and sp and time.time() - _SPOT.get("ts", 0) < 15:
        sp_el = sp["progress"] + ((time.time() - _SPOT["ts"]) if sp["playing"] else 0.0)
        sp_dur = sp["duration"] or 0.0
        sp_el = min(sp_el, sp_dur) if sp_dur else sp_el
        return {"track": sp["name"] + (f" — {sp['artist']}" if sp["artist"] else ""),
                "playing": sp["playing"], "paused": not sp["playing"],
                "elapsed": round(sp_el, 1), "duration": round(sp_dur, 1),
                "fraction": round(sp_el / sp_dur, 4) if sp_dur else 0.0,
                "vol": int(m.get("vol", 60)),
                "source": "Spotify" + (f" · {sp['device']}" if sp["device"] else "")}
    return {"track": m["track"] or "nothing playing",
            "playing": playing,
            "paused": bool(m.get("paused")) and not playing,
            "elapsed": round(elapsed, 1),
            "duration": round(dur, 1),
            "fraction": round(frac, 4),
            "vol": int(m.get("vol", 60))}


# Spotify player state for the music card (2026-11-07): polled every 4 s by a
# background thread through the same spotify_tools the MCP tools use.
_SPOT = {"state": None, "ts": 0.0, "kick": threading.Event()}


def _spotify_poll_loop():
    import importlib
    here = Path(__file__).resolve().parent
    for cand in (here.parent, here):
        if (cand / "spotify_tools.py").exists():
            sys.path.insert(0, str(cand))
            break
    try:
        sp_mod = importlib.import_module("spotify_tools")
    except Exception as e:
        print(f"spotify card: disabled ({e!r})", file=sys.stderr)
        return
    while True:
        try:
            r = sp_mod._api("GET", "/me/player")
            it = (r or {}).get("item") if isinstance(r, dict) and "error" not in r else None
            if it:
                _SPOT["state"] = {
                    "name": it.get("name") or "", "playing": bool(r.get("is_playing")),
                    "artist": (it.get("artists") or [{}])[0].get("name", ""),
                    "progress": (r.get("progress_ms") or 0) / 1000.0,
                    "duration": (it.get("duration_ms") or 0) / 1000.0,
                    "device": (r.get("device") or {}).get("name")}
            else:
                _SPOT["state"] = None
            _SPOT["ts"] = time.time()
        except Exception:
            pass
        # every 4 s, or at once when a Spotify tool just acted (/spotify_kick)
        _SPOT["kick"].wait(4)
        if _SPOT["kick"].is_set():
            time.sleep(0.6)            # let Spotify's player state settle
            _SPOT["kick"].clear()


def ui_state_json() -> dict:
    """The handover §7 data model for pi/boots_ui.html (1 Hz poll)."""
    now = time.time()
    with STATE["lock"]:
        mode = STATE["mode"]
        mode_ts = STATE["mode_ts"]
        last_good = STATE["last_good"]
        last_dim = STATE.get("last_dim")
        weather = STATE.get("weather")
        # The place the user ASKED about (check_weather -> POST /weather)
        # wins for 30 min, then the card falls back to the home reading.
        _asked = STATE.get("weather_asked")
        if _asked and now - _asked.get("ts", 0) < 1800:
            weather = _asked
        dev = dev_view(now)
        _c = DEV.get("call")
        # active = the ~8 s "ringing" window the card animates for; after
        # that the card shows it as the last call.
        call = ({"number": _c["number"], "ts": _c["ts"],
                 "active": now - _c["ts"] < CALL_RING_S} if _c else None)
    if mode == "awake":
        umode = "awake"                       # "Listening…" + edge glow
    elif mode == "reject":
        umode = "reject" if now - mode_ts < 8.0 else "idle"
    elif last_good is not None and now - last_good.get("ts", 0) < 60.0:
        umode = "done"
    else:
        umode = "idle"
    ulast = None
    if last_good:
        ulast = {"cmd": last_good.get("cmd"),
                 "conf": last_good.get("conf"),
                 "latency": last_good.get("e2e_ms", last_good.get("infer_ms"))}
        ulast["slot"] = last_good.get("slot")
        # done-card display formats (V3: '0:30' / '8:00 AM' / '22°' / '60%')
        disp = {"TIMER": _timer_word, "ALARM": _alarm_word,
                "TEMPERATURE": _temp_word,
                "BRIGHTNESS": _bright_word}.get(last_good.get("cmd"))
        if disp:
            w = disp(last_good.get("slot") or "")
            if w:
                ulast["slot"] = w
        if last_good.get("slot_margin") is not None:
            ulast["slotMargin"] = last_good["slot_margin"]
        if last_good.get("slot_alt"):
            ulast["slotAlt"] = last_good["slot_alt"]
    t = dev["timer"]
    if t and not t["done"]:
        timer = {"remaining": t["remaining"],
                 "frac_left": round(1.0 - t["frac"], 4), "done": False}
    elif t:
        timer = {"remaining": 0, "frac_left": 0.0, "done": True}
    else:
        timer = None
    p = dev["temp"]
    tmode = (p["mode"] + " → " + str(p["target"]) + "°") \
        if p["target"] is not None else "idle"
    return {
        "mode": umode,
        "last": ulast,
        "lamp": {"on": dev["lights"]["on"], "level": dev["lights"]["level"],
                 "color": dev["lights"]["color"]},
        "thermo": p["current"],
        "tmode": tmode,
        "timer": timer,
        "alarms": dev["alarms"],
        "music": _ui_music(dev),
        "notes": [c for c, _ts in dev["chips"]],
        "reminders": dev["reminders"],
        "call": call,
        "weather": weather,
        "speaking": time.time() < STATE.get("speaking_until", 0.0),
        "slotPrompt": None, "candidates": None, "retries": None,
        "me3": STATE["me3"],
        # Server instance id (2026-11-07): the page reloads when it changes —
        # a restart resets turn/tool_seq/heard_seq to 0, and a tab kept open
        # across it de-duped against the OLD counters (missing/misplaced chips).
        "boot_id": _BOOT_ID,
        "chat_history": list(STATE["chat_history"]),
        "cmdlog": list(STATE["cmdlog"]),
        # getattr: _CHAT_LLM is None until the first /chat call (lazy load)
        "llm": {"provider": getattr(_CHAT_LLM, "provider", "off"),
                "model": getattr(_CHAT_LLM, "model", "-"),
                "base_url": getattr(_CHAT_LLM, "base_url", ""),
                "thinking": getattr(_CHAT_LLM, "thinking", "off")},
    }


# ---------------------------------------------------------------- typed chat
# The chat engine for POST /chat. Mirrors agent.py's LLM loop: same prompt,
# same tool modules, same schema normalization (Groq validates strictly),
# same fail-soft dispatch. Runs in a thread (ThreadingHTTPServer), so a slow
# LLM call never blocks the 1 Hz /ui_state polls or the voice agent's pushes.
# Voice turns share the SAME conversation memory as typed chat (the UI chat
# is a voice proxy — one pipeline, one context). A voice turn is detected as
# complete when its phase leaves "speaking" with a response present.
_VOICE_LAST = {"phase": None, "heard": None}
# Voice CoT round tracking (2026-11-06): the Pi agent's tap pushes the LLM's
# streamed text as 'thinking' and then CLEARS it ("" on LLMFullResponseEnd /
# FunctionCallResult). With a 1 Hz poll, a fast round (sub-1s) is missed
# entirely, and even a caught one vanishes from the CoT box on the next poll.
# Fix: the server keeps the LAST non-empty thinking per round in m3['cot']
# (survives the clear) and counts rounds (m3['cot_round']) — a new round
# starts on empty->non-empty (or a tool call). The UI keys its think row on
# cot_round, exactly like the typed path keys on the SSE stream.
_VOICE_ROUND = {"prev": "", "round": 0, "first": True, "t0": 0.0}

def _append_turn(heard: str, response: str, tools=None) -> None:
    """Append one completed turn (user text + assistant reply) to the shared
    conversation memory (full thread; the LLM window is sliced in _chat_run)."""
    try:
        if not (response or "").strip():
            return
        # heard may be empty (the boot greeting has no user utterance) —
        # the assistant entry is still appended so the greeting survives a
        # refresh (it used to vanish because the append required heard).
        with STATE["lock"]:
            # DEDUPE (2026-11-06): the same (heard, response) pair may only be
            # appended ONCE. The Pi's reply-tap used to force phase="speaking"
            # on every LLM token while the real phase was still "thinking", so
            # the phase bounced thinking->speaking per token and this handler
            # (which appends on every speaking->non-speaking transition) wrote
            # the SAME reply to history once per token — the UI then rendered
            # N identical assistant bubbles. With the tap fixed the phase is
            # clean, but this guard makes the history append idempotent either
            # way.
            _h = STATE["chat_history"]
            _prev = _h[-1] if _h else None
            if _prev and _prev.get("role") == "assistant" \
                    and _prev.get("content") == response[:4000] \
                    and (not heard or (len(_h) >= 2 and _h[-2].get("content") == heard[:400])):
                return
            if heard:
                STATE["chat_history"].append(
                    {"role": "user", "content": heard[:400]})
            _entry = {"role": "assistant", "content": response[:4000]}
            if tools:
                # UI-only chip payload (dropped from the LLM prompt window,
                # same as typed turns' "tools").
                _entry["tools"] = [dict(t) for t in tools]
            STATE["chat_history"].append(_entry)
    except Exception:
        pass

def _cmdlog(text: str, cls: str = "ok") -> None:
    """Append one command-log entry (newest last; the UI renders
    newest-first and rebuilds from this on refresh). Cap 50."""
    try:
        with STATE["lock"]:
            STATE["cmdlog"].append(
                {"t": time.strftime("%H:%M:%S"), "text": str(text)[:200],
                 "cls": cls})
    except Exception:
        pass

def _chat_load():
    """Lazily load prompt + tools + LLM config (env-driven, like agent.py).
    Fail-soft: a broken tool import must not crash /ui_state (the whole
    page polls it) — chat 503s, everything else keeps working."""
    global _CHAT_TOOLS, _CHAT_PROMPT, _CHAT_PROMPT_MOD, _CHAT_LLM
    # not None AND not the _NoLLM stub (a failed first load should be
    # retried on the next turn, not cached as "unavailable" forever)
    if _CHAT_LLM is not None and \
            getattr(_CHAT_LLM, "provider", "") != "unavailable":
        return
    try:
        _chat_load_impl()
    except Exception as e:
        print(f"chat engine unavailable: {e}", file=sys.stderr)
        if _CHAT_LLM is None:
            class _NoLLM:
                provider = "unavailable"
                model = "-"
                base_url = ""
                thinking = "off"
            _CHAT_LLM = _NoLLM()

def _chat_load_impl():
    global _CHAT_TOOLS, _CHAT_PROMPT, _CHAT_PROMPT_MOD, _CHAT_LLM
    here = Path(__file__).resolve().parent
    # Tool modules + prompt.py live in the me3/ dir (parent of bootsui/) —
    # same layout as the Pi (~/me3/bootsui/ + ~/me3/*.py). Fall back to the
    # bootsui/ dir itself (the laptop sandbox layout).
    for cand in (here.parent, here):
        if (cand / "prompt.py").exists() and (
                (cand / "me2_tools.py").exists()
                or (cand / "boots_ui_mcp.py").exists()):
            sys.path.insert(0, str(cand))
            break
    try:
        import me2_tools as _ui
        import spotify_tools as _sp
        import web_tools as _web
        import prompt as _prompt
    except ImportError:  # Pi layout (pre-rename generations)
        try:
            import boots_ui_mcp as _ui
        except ImportError:
            import me2_ui_mcp as _ui
        import spotify_tools as _sp
        import web_tools as _web
        import prompt as _prompt
    # File + shell tools (voice parity) — optional: older deployments may not
    # have them yet, and the chat path must still work without.
    _ft = _sh = None
    try:
        import files_tools as _ft
    except ImportError:
        pass
    try:
        import shell_tools as _sh
    except ImportError:
        pass

    def _norm_props(p):
        out = {}
        for k, v in (p or {}).items():
            if isinstance(v, str):
                out[k] = {"type": v}
            elif isinstance(v, (dict, bool)):
                out[k] = v
            else:
                out[k] = {"type": "string"}
        return out

    tools = []
    for mod in (_ui, _sp, _web, _ft, _sh):
        if mod is None:
            continue
        for (n, d, p) in mod.TOOL_SPECS:
            tools.append({
                "type": "function",
                "function": {
                    "name": n, "description": d,
                    "parameters": {"type": "object",
                                   "properties": _norm_props(p),
                                   "required": [k for k in p if k not in
                                                getattr(mod, "OPTIONAL_PARAMS", {}).get(n, ())]},
                },
            })

    _CHAT_TOOLS = tools
    _CHAT_PROMPT = _prompt.SYSTEM_PROMPT
    _CHAT_PROMPT_MOD = _prompt   # for system_prompt() (base + fresh clock)
    _apply_llm_cfg(_llm_cfg())

# ---------------------------------------------------------------------------
# LLM provider config — swappable at runtime via POST /llm (UI dropdown).
# Precedence: saved llm.json (user's UI choice) > env vars > Groq default.
# ---------------------------------------------------------------------------
_LLM_PRESETS = {
    "groq": {
        "label": "Groq · " + os.environ.get("LLM_MODEL", "qwen/qwen3.8-27b").split("/")[-1],
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.environ.get("LLM_API_KEY")
                  or os.environ.get("GROQ_API_KEY", ""),
        "model": os.environ.get("LLM_MODEL", "qwen/qwen3.8-27b"),
        # Groq Qwen3: reasoning_effort "none" = thinking off (saves tokens +
        # latency), "default" = model thinks (costs tokens against TPM/RPD).
        "thinking": "off",
    },
    "hpc": {
        # HPC/personal-server vLLM — same endpoint + key muji uses
        # (202.92.159.240:8001, Qwen3.8-27B). DGX3 (10.153.60.233) is
        # ASR/TTS only and needs SHARC VPN — not an LLM target.
        # Env-overridable: ME3_DGX3_LLM_BASE / ME3_DGX3_API_KEY /
        # ME3_DGX3_LLM_MODEL
        # ME3_HPC_* (2026-11-07, shared with agent.py's fallback) win over
        # the legacy ME3_DGX3_* names.
        "label": "HPC · vLLM",
        "base_url": os.environ.get("ME3_HPC_LLM_BASE") or os.environ.get(
            "ME3_DGX3_LLM_BASE", "http://202.92.159.240:8001/v1"),
        # key from the environment ONLY (/etc/me3/groq.env); no hardcoded
        # fallback (public repo). NOT ME3_DGX3_API_KEY: that is the DGX3
        # ASR/TTS key, a different server.
        "api_key": os.environ.get("ME3_HPC_API_KEY", ""),
        "model": os.environ.get("ME3_HPC_LLM_MODEL") or os.environ.get(
            "ME3_DGX3_LLM_MODEL", "Qwen/Qwen3.8-27B"),
        # HPC is self-hosted (no rate limit to save) and its Qwen3 returns
        # the reasoning stream in the 'reasoning' field — default ON so the
        # CoT think row has a real stream to show. Groq stays "off": its
        # Qwen3 emits NO reasoning tokens with reasoning_effort:"none",
        # and the 1000-OTPM tier makes thinking turns expensive.
        "thinking": "on",
    },
}
_LLM_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "llm.json")


def _llm_cfg() -> dict:
    """Current LLM config: llm.json (user's UI pick) > env > Groq default."""
    cfg = dict(_LLM_PRESETS["groq"])
    env_base = os.environ.get("LLM_BASE_URL", "")
    env_model = os.environ.get("LLM_MODEL", "")
    if env_base:
        cfg["base_url"] = env_base
    if env_model:
        cfg["model"] = env_model
    try:
        with open(_LLM_CFG_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            cfg.update({k: saved[k] for k in
                        ("base_url", "api_key", "model") if saved.get(k)})
            if saved.get("provider"):
                cfg["provider"] = saved["provider"]
            if saved.get("thinking") in ("on", "off"):
                cfg["thinking"] = saved["thinking"]
    except Exception:
        pass
    return cfg


def _apply_llm_cfg(cfg: dict):
    """Point the chat's LLM session at cfg (called at load + on /llm switch)."""
    global _CHAT_LLM
    _CHAT_LLM = _requests.Session()
    _CHAT_LLM.headers.update({
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json"})
    _CHAT_LLM.base_url = cfg["base_url"].rstrip("/")
    _CHAT_LLM.model = cfg["model"]
    _CHAT_LLM.provider = cfg.get("provider", "custom")
    _CHAT_LLM.thinking = cfg.get("thinking", "off")

_HPC_FB = {"s": None}
# "website/web page" anywhere, or a bare "site" ONLY right after a build
# verb ("build a site") -- "visit the site office" must not match.
_SITE_INTENT = re.compile(
    r"\b(web ?site|web ?page|landing page)\b"
    r"|\b(make|build|create|write|generate|design)\b\s+(me\s+|us\s+)?(a|an|my|the|our)?\s*(new\s+|simple\s+|small\s+)?site\b",
    re.I)


def _hpc_fallback_session():
    """requests.Session for the HPC preset (typed-chat fallback), or None."""
    if _requests is None:
        return None
    if _HPC_FB["s"] is None:
        p = _LLM_PRESETS["hpc"]
        if not (p.get("base_url") and p.get("api_key")):
            return None
        s = _requests.Session()
        s.headers.update({"Authorization": f"Bearer {p['api_key']}",
                          "Content-Type": "application/json"})
        s.base_url = p["base_url"].rstrip("/")
        s.model = p["model"]
        _HPC_FB["s"] = s
    return _HPC_FB["s"]


def _chat_run(msg: str, emit=None):
    """One typed turn: LLM -> tools (max 10 rounds), like agent.py. Returns
    (reply_text, [tool calls made]). emit(kind, payload) is the SSE hook —
    /chat_stream passes a callback that pushes {"thinking","tool_start",
    "tool_result","reply","done"} events to the browser as they happen
    (muji-style: the CoT box streams deltas, not a 1 Hz poll snapshot)."""
    _chat_load()
    # LLM context window: the LAST 5 turns from the full thread (the UI
    # keeps everything; the model only sees the tail). Capped to keep the
    # prompt small (voice-agent latency budget).
    with STATE["lock"]:
        # Prompt window: last 5 history entries, rebuilt with the REAL
        # tool-call records. Live-tested (2026-11-04): if prior tool turns
        # appear as plain-text confirmations, Qwen3.8 imitates them and
        # skips the tool call ("Lights are on." with no action); if they
        # appear as assistant.tool_calls + tool messages, it reliably
        # emits the tool call. UI-only fields ("tools" chip payload) are
        # dropped — Groq 400s on unknown message fields.
        tail = list(STATE["chat_history"])[-10:]  # 5 user+assistant turns
        prior = []
        for m in tail:
            if m["role"] == "assistant" and m.get("tool_calls"):
                prior.append({"role": "assistant",
                              "content": m.get("content") or None,
                              "tool_calls": m["tool_calls"]})
                for c in m["tool_calls"]:
                    prior.append({"role": "tool",
                                  "tool_call_id": c["id"],
                                  "content": c.get("result", "")})
            else:
                prior.append({"role": m["role"],
                              "content": m.get("content", "")})
    # Fresh clock each turn: the LLM has no time tool, so 'what time is it'
    # is answered from this line (the edge fast-path reads the clock itself).
    if _CHAT_PROMPT_MOD is not None:
        _sys = _CHAT_PROMPT_MOD.system_prompt()
    else:
        _sys = _CHAT_PROMPT or "You are Boots, a warm voice assistant."
    messages = [{"role": "system", "content": _sys}]
    messages.extend(prior)
    messages.append({"role": "user", "content": msg})
    calls = []
    reply = "(stopped after 10 tool rounds)"
    # Empty reply guard: an empty assistant entry in the prompt window is
    # invalid for strict OpenAI-compatible providers (and it made the UI
    # skip the bubble via the replyShown dedupe). Fall back to a short
    # spoken line instead of storing "" in history.
    llm_ms = 0   # total LLM round-trip time across all rounds
    # Unexecuted-action guard (2026-11-07): gpt-oss sometimes ENDS a website
    # turn with text claiming the site is built ("verified working") plus a
    # made-up URL, without ever calling write_website (reasoning literally
    # said "Now create website."). When the user asked for a site and the
    # turn tries to end without the call, the claim is discarded, the model
    # is told nothing was built, and the next round FORCES write_website.
    _wants_site = bool(_SITE_INTENT.search(msg))
    _nudged = False
    _force_tool = None
    for _ in range(10):  # research (2-3 search + 2 fetch) + write needs >5 (2026-11-07)
        body = {"model": _CHAT_LLM.model, "messages": messages,
                "tools": _CHAT_TOOLS, "temperature": 0.3,
                # Groq stays capped (replies are 1-2 sentences; without this
                # Groq assumes 2048 and 429s our 1000-OTPM tier). HPC is
                # self-hosted (no rate limit) and MUST get headroom: the
                # write_website tool call alone is ~1.5k+ tokens of JSON
                # (4-6 sections + sources), and thinking burns ~900 on top.
                # At 256 the stream died mid-args (finish=length, empty
                # args -> {} -> empty reply -> "didn't catch that" — the
                # 2026-11-05 "write me anything" failure, reproduced live).
                # (2026-11-07) Groq too: the 256 cap (free-tier OTPM era)
                # starved write_website — its ~1.5k-token argument JSON +
                # reasoning came back EMPTY -> "Done." and no site.
                "max_tokens": 16000}
        # Groq Qwen3 thinking switch. "off" -> reasoning_effort:"none"
        # (no reasoning tokens, faster, cheaper). "on" -> omit the param
        # (Groq default = model reasons). Groq-only: other providers
        # (vLLM) don't accept this param.
        # gpt-oss rejects "none" (HTTP 400: low|medium|high only) -> "low".
        if _CHAT_LLM.provider == "groq":
            _oss = "gpt-oss" in (_CHAT_LLM.model or "")
            if getattr(_CHAT_LLM, "thinking", "off") == "off":
                body["reasoning_effort"] = "low" if _oss else "none"
            elif _oss:
                body["reasoning_effort"] = "medium"
        # STREAMED: the CoT think row must show the reasoning as it
        # happens, not a blob that appears when the whole round finishes
        # (the non-streamed fetch only exposed it post-hoc). Verified
        # 2026-11-05: vLLM Qwen3.8 emits reasoning deltas in
        # delta["reasoning"] (Groq-style APIs: delta["reasoning_content"]);
        # tool_calls arrive as per-chunk fragments (index + name on the
        # first, arguments in pieces) and are reassembled below.
        if _force_tool:
            body["tool_choice"] = {"type": "function",
                                   "function": {"name": _force_tool}}
            _force_tool = None
        t_llm = time.time()
        body["stream"] = True
        # HPC FALLBACK (2026-11-07): Groq is primary; any failure to OPEN the
        # stream (429/5xx/401/timeout/network) re-sends this round to the
        # HPC vLLM. Not mid-stream: a stream that dies after tokens raises.
        try:
            r = _CHAT_LLM.post(
                _CHAT_LLM.base_url + "/chat/completions", json=body,
                timeout=(5, 60))
            r.raise_for_status()
        except Exception as _e:
            _fb = _hpc_fallback_session()
            if _fb is None or _fb.base_url == _CHAT_LLM.base_url:
                raise
            _cmdlog(f"LLM {_CHAT_LLM.provider} failed ({type(_e).__name__}) "
                    f"-> HPC fallback", "err")
            _b2 = dict(body, model=_fb.model, max_tokens=16000)
            _b2.pop("reasoning_effort", None)
            r = _fb.post(_fb.base_url + "/chat/completions", json=_b2,
                         timeout=(10, 120))
            r.raise_for_status()
        _think = ""
        _content = ""
        _tc = {}   # index -> {"id","name","args"}
        _finish = None
        _last_push = 0.0
        for _raw in r.iter_lines():
            if not _raw:
                continue
            line = _raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                _j = json.loads(payload)
            except json.JSONDecodeError:
                continue
            _choices = _j.get("choices") or []
            if not _choices:
                continue
            _d = _choices[0].get("delta") or {}
            _r = _d.get("reasoning") or _d.get("reasoning_content") or ""
            if _r:
                _think += _r
                # Push on EVERY reasoning delta (throttled to ~10 Hz): the
                # UI's 1 Hz poll then shows a growing stream, not a jump.
                # The SSE emit is per-delta (no throttle), CUMULATIVE —
                # the UI handler is idempotent (writes the full snapshot,
                # same as the poll's cotThink), so SSE + poll can't fight.
                if emit:
                    emit("thinking", {"text": _think})
                _now = time.time()
                if _now - _last_push > 0.1:
                    with STATE["lock"]:
                        STATE["me3"]["thinking"] = _think[:1500]
                    _last_push = _now
            _c = _d.get("content") or ""
            if _c:
                _content += _c
                # Qwen3.8 opens its content with "\n\n" (thinking-model
                # habit) — strip for the bubble + history so the reply
                # doesn't render with a leading blank line.
                _clean = _content.lstrip("\n")
                # Stream the in-flight reply too: during the 'thinking'
                # phase the UI renders m3.response as the typing bubble,
                # so pushing it here (same throttle) makes the bubble
                # type out live instead of popping in whole.
                with STATE["lock"]:
                    STATE["me3"]["response"] = _clean[:4000]
                if emit:
                    emit("reply", {"text": _clean})
            for _f in _d.get("tool_calls") or []:
                _idx = _f.get("index", 0)
                _e = _tc.setdefault(_idx, {"id": "", "name": "", "args": ""})
                if _f.get("id"):
                    _e["id"] = _f["id"]
                _fn = _f.get("function") or {}
                if _fn.get("name"):
                    _e["name"] += _fn["name"]
                if _fn.get("arguments"):
                    _e["args"] += _fn["arguments"]
            if _choices[0].get("finish_reason"):
                _finish = _choices[0]["finish_reason"]
        if _think:
            with STATE["lock"]:
                STATE["me3"]["thinking"] = _think[:1500]
        llm_ms += int((time.time() - t_llm) * 1000)
        ch = {"content": _content, "reasoning": _think}
        tcs = [{"id": _e["id"] or "call_%d" % _i,
                "type": "function",
                "function": {"name": _e["name"], "arguments": _e["args"]}}
               for _i, _e in sorted(_tc.items())]
        if tcs and _finish == "length":
            # Stream died mid-generation: a tool call with TRUNCATED
            # arguments (json.loads would fail -> {} -> the tool errors or
            # misfires, and the empty-content round fell through to the
            # "didn't catch that" fallback — the 2026-11-05 failure).
            # Don't execute a half-built call; tell the user honestly.
            reply = ("That took longer to build than I have room for — "
                     "try again and I'll keep it shorter.")
            break
        if not tcs:
            reply = (ch.get("content") or "").strip()
            # Drop Qwen's leading "\n\n" habit from the stored/spoken reply.
            reply = reply.lstrip("\n").strip()
            if (_wants_site and not _nudged
                    and not any(c["name"] == "write_website" for c in calls)):
                _nudged = True
                _force_tool = "write_website"
                _cmdlog("guard: website claimed without write_website -> forcing the call", "err")
                messages.append({"role": "assistant", "content": reply or "(no reply)"})
                messages.append({"role": "user", "content": (
                    "You have NOT called write_website — nothing was built and "
                    "any link you gave is not real. Call write_website now, "
                    "using what your research found (sections, stats, sources).")})
                if emit:
                    emit("reply", {"text": ""})   # drop the false claim from the bubble
                with STATE["lock"]:
                    STATE["me3"]["response"] = ""
                continue
            break
        messages.append({
            "role": "assistant", "content": ch.get("content") or "",
            "tool_calls": tcs})
        for tc in tcs:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            # Push the tool START into me3 state before executing: the UI's
            # 1 Hz poll renders a purple spinner chip for it (one chip per
            # tool_seq), and the RESULT push (same seq) flips that same chip
            # to the green pill — one object, purple -> green, live.
            seq = 0
            with STATE["lock"]:
                m3 = STATE["me3"]
                m3["tool_seq"] = m3.get("tool_seq", 0) + 1
                seq = m3["tool_seq"]
                m3["phase"] = "tool"
                m3["tool"] = name[:120]
                m3["tool_args"] = json.dumps(args)[:400]
                m3["tool_result"] = ""
                m3["tool_ms"] = 0
            if emit:
                emit("tool_start", {"name": name, "args": args, "seq": seq})
            t_tool = time.time()
            if name.startswith("boots_"):
                res = _ui_dispatch(name, args)
            elif name.startswith("spotify_"):
                res = _sp_dispatch(name, args)
            elif name in ("list_dir", "read_file", "write_file",
                          "edit_file", "search_files"):
                res = _ft_dispatch(name, args)
            elif name in ("run_command", "run_python"):
                res = _sh_dispatch(name, args)
            else:
                res = _web_dispatch(name, args)
            err = isinstance(res, str) and bool(res) and res[:7].lower() in ("errored", "error: ", "error —")
            with STATE["lock"]:
                m3 = STATE["me3"]
                m3["tool_result"] = str(res)[:200]
                m3["tool_ms"] = int((time.time() - t_tool) * 1000)
                m3["phase"] = "thinking"  # back to LLM until the turn ends
            if emit:
                emit("tool_result", {"name": name, "seq": seq,
                                      "result": str(res)[:200],
                                      "ms": int((time.time() - t_tool) * 1000),
                                      "ok": not err})
            # result + ms are returned to the UI: the tool chip needs both
            # to leave its spinner state (missing fields rendered as
            # "ok · undefined ms" and the chip looked stuck loading).
            # ok: explicit success flag so the UI stops guessing from the
            # result string (errors auto-open the detail block in amber).
            calls.append({"id": tc["id"], "name": name, "args": args,
                          "seq": seq, "result": str(res)[:200],
                          "ms": int((time.time() - t_tool) * 1000),
                          "ok": not err})
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": str(res)[:2000]})
    with STATE["lock"]:
        STATE["chat_history"].append({"role": "user", "content": msg[:400]})
        # tools ride on the assistant entry: the UI rebuilds each turn's
        # chips from chat_history on refresh (persistence) — without this
        # the chips only existed in the /chat HTTP response and vanished
        # on reload.
        if not (reply or "").strip():
            reply = "Done." if calls else "Sorry, I didn't catch that — try again?"
        entry = {"role": "assistant", "content": reply[:4000],
                 # UI chip payload (dropped before the LLM prompt — Groq
                 # 400s on unknown message fields).
                 "tools": [{"seq": c["seq"], "name": c["name"],
                            "args": c.get("args") or {},
                            "result": c["result"][:200], "ms": c["ms"]}
                           for c in calls]}
        # Real tool_call records ride on the entry too: the prompt builder
        # re-expands them into assistant.tool_calls + tool messages so the
        # model sees the CORRECT pattern (live-tested: text-only
        # confirmations make Qwen3.8 skip the tool call and imitate the
        # lie; real records make it call the tool reliably).
        if calls:
            entry["tool_calls"] = [{
                "id": c["id"], "type": "function",
                "function": {"name": c["name"],
                             "arguments": json.dumps(c.get("args") or {})},
                "result": c["result"][:200]} for c in calls]
        STATE["chat_history"].append(entry)
    # Latency breakdown for the UI readout: llm = all LLM round-trips,
    # tool = all tool execs, total = whole turn. asr/tts are voice-only
    # (a typed turn has no STT/TTS legs) so they stay None -> "–".
    lat = {"llm": llm_ms,
           "tool": sum(c.get("ms", 0) for c in calls) or None,
           "total": None}  # filled by the /chat handler (includes queue)
    return reply, calls, lat

def _ui_dispatch(name, args):
    try:
        import me2_tools as _m
        return _m.dispatch(name, **args)
    except Exception as e:
        return f"tool {name} errored: {e!r}"

def _sp_dispatch(name, args):
    try:
        import spotify_tools as _m
        return _m.dispatch(name, **args)
    except Exception as e:
        return f"tool {name} errored: {e!r}"

def _web_dispatch(name, args):
    try:
        import web_tools as _m
        return _m.dispatch(name, **args)
    except Exception as e:
        return f"tool {name} errored: {e!r}"

def _ft_dispatch(name, args):
    try:
        import files_tools as _m
        return _m.dispatch(name, **args)
    except Exception as e:
        return f"tool {name} errored: {e!r}"

def _sh_dispatch(name, args):
    try:
        import shell_tools as _m
        return _m.dispatch(name, **args)
    except Exception as e:
        return f"tool {name} errored: {e!r}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html", "/muji"):
            # The ME3 observability dashboard (muji-styled chat + inspector) is
            # the UI. Served from boots_ui.html next to this file when present.
            if _PAGE_HTML is not None:
                self._send(200, _PAGE_HTML, "text/html; charset=utf-8")
            else:
                self._send(404, b"UI not found (drop boots_ui.html next to boots_ui.py)", "text/plain")
        elif self.path == "/state":
            self._send(200, json.dumps(state_json()).encode(),
                       "application/json")
        elif self.path == "/ui_state":
            self._send(200, json.dumps(ui_state_json()).encode(),
                       "application/json")
        elif self.path.startswith("/site/"):
            # Serve a generated website (write_website tool) from the sites/ dir.
            name = self.path[len("/site/"):]
            # Security: only allow a single .html filename, no traversal.
            if (name and not name.startswith("/") and ".." not in name
                    and "/" not in name and name.endswith(".html")):
                p = os.path.join(SITES_DIR, os.path.basename(name))
                if os.path.isfile(p):
                    try:
                        with open(p, "rb") as fh:
                            data = fh.read()
                        self._send(200, data, "text/html; charset=utf-8")
                        return
                    except Exception:
                        pass
            self._send(404, b"site not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            obj = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b"bad json", "text/plain")
            return
        if self.path == "/spotify_kick":     # (2026-11-07) refresh the music card now
            _SPOT["kick"].set()
            self._send(200, b"ok", "text/plain")
            return
        if self.path == "/fire":
            with STATE["lock"]:
                outcome = apply_fire(obj)
                STATE["last"] = obj
                STATE["history"].appendleft(obj)
                STATE["total"] += 1
                if outcome in ("gated", "invalid"):   # not applied
                    STATE["mode"] = "reject"  # red RETRY card, decays in 8 s
                else:
                    STATE["last_good"] = obj
                    STATE["mode"] = "idle"
                STATE["mode_ts"] = time.time()
            self._send(200, f"ok:{outcome}".encode(), "text/plain")
        elif self.path == "/me3":
            # Live conversation state from the ME3 voice agent:
            # {"phase","heard","thinking","response","tool","active"}
            phase = obj.get("phase")
            if phase not in ("idle", "listening", "transcribing", "thinking",
                             "tool", "speaking", "nomic"):   # nomic: waiting for the EMEET (2026-11-07)
                self._send(400, b"bad phase", "text/plain")
                return
            _prev_tool = None
            _log_tool = None  # set once per NEW tool call (logged after the lock)
            with STATE["lock"]:
                m3 = STATE["me3"]
                _prev_tool = m3.get("tool") or ""
                # turn: monotonic per-voice-turn id (2026-11-06). Bumped when a
                # NEW heard text arrives. The UI keys its reply bubble on this
                # (not on m3.ts, which moves on EVERY push — a mid-turn ts bump
                # made the poll mint a fresh bubble per re-push, and the
                # heard-only boundary reset mid-stream did the same).
                # Greeting (2026-11-07): the boot welcome has no utterance —
                # clear the stale last-heard so it isn't paired with the
                # previous session's last words (duplicated HEARD bubble).
                if obj.get("greeting") and phase == "speaking":
                    m3["heard"] = ""
                    m3["turn"] = m3.get("turn", 0) + 1
                _new_heard = (obj.get("heard") or m3["heard"])[:300]
                # New-turn boundary = a 'listening' push with a fresh heard.
                # NOT 'any push with a different heard' (2026-11-06): the
                # agent's HEARTBEAT push carries the stale last-heard with
                # phase='idle', and a mid-idle poll would then re-fire the
                # turn bump + clear the previous turn's cot/latency (the Pi
                # stress caught this: heartbeat wiped the stored latency).
                # (the variable is `phase` — `_phase` was a typo that raised
                # NameError on ANY new-heard turn push; the handler's except
                # swallowed it as a generic "bad json" 400, so the Pi's
                # real voice turns were all 400ing while same-heard pushes
                # short-circuited past the bug and worked. 2026-11-06.)
                # new_turn (2026-11-07): the ASR pushes it once per transcript,
                # so a REPEATED utterance ("hey" twice) also starts a turn —
                # heard-text change alone can't see a repeat (the 2nd reply
                # overwrote the 1st bubble). Two pushes carry each transcript
                # (ASR + StatusIndicator); _turn_heard/_turn_ts dedupe them.
                _now = time.time()
                _same_recent = (_new_heard == m3.get("_turn_heard")
                                and _now - m3.get("_turn_ts", 0) < 3.0)
                _bump = False
                if obj.get("new_turn") and _new_heard and not _same_recent:
                    _bump = True
                elif (_new_heard and _new_heard != m3["heard"]
                        and _new_heard != m3.get("_turn_heard")
                        and phase in ("listening", "transcribing")):
                    _bump = True
                if _bump:
                    m3["_turn_heard"] = _new_heard
                    m3["_turn_ts"] = _now
                    m3["heard_seq"] = m3.get("heard_seq", 0) + 1  # voice-only HEARD key
                    m3["latency"] = None  # new turn: the old readout goes now, not mid-turn
                    m3["_turn_tools"] = []  # this turn's tool calls (for history)
                    m3["turn"] = m3.get("turn", 0) + 1
                    # New turn: drop the PREVIOUS turn's tool result so the
                    # UI's seq-keyed path can't re-flip the old chip green
                    # (the agent clears its own tool_result on a new
                    # utterance, but a crashed/late agent could leave it).
                    m3["tool_result"] = ""
                    m3["tool_ms"] = 0
                    # New turn: drop the PREVIOUS turn's CoT text too. The
                    # UI's CoT box reads m3.thinking||m3.cot and fires on
                    # change — a page refresh rebuilds the panel with an
                    # empty lastCotText, so a lingering m3.cot would
                    # re-fire the old turn's thinking into the fresh panel
                    # (the 'stale think' bug, 2026-11-06).
                    m3["cot"] = ""
                    m3["thinking"] = ""
                m3["phase"] = phase
                # heard: the transcript arrives via _push_heard_to_ui (a separate,
                # reliable POST); the agent's frame-pushes send empty heard, so we
                # PRESERVE the last non-empty heard. (obj.get(...) or m3[...]).
                m3["heard"] = (obj.get("heard") or m3["heard"])[:300]
                # thinking/response: the agent's REAL pushes (from
                # _me3_push/_me3_push_now/_me3_push_state) always include these
                # keys -- INCLUDING an explicit empty string on turn reset /
                # LLMFullResponseStart. We must HONOR empty (clear the bubble)
                # rather than keep the stale previous value. BUT the 2s
                # liveness heartbeat omits these keys entirely (it only sends
                # hb/active/phase), so we only touch thinking/response/tool when
                # the key is actually PRESENT in the payload (via a sentinel).
                _absent = object()
                if obj.get("thinking", _absent) is not _absent:
                    _nt = (obj.get("thinking") or "")[:1500]  # same cap as the typed path (was 300)
                    m3["thinking"] = _nt
                    # CoT round tracking (see _VOICE_ROUND above): the agent
                    # (2026-11-06) pushes its OWN round number (cot_round,
                    # bumped at LLMFullResponseStart) — authoritative; fall
                    # back to local empty->non-empty detection for older
                    # agents. The last non-empty text is kept in m3['cot'] so
                    # the 1 Hz poll can't miss a fast round and the row
                    # survives the agent's end-of-round clear.
                    if obj.get("cot_round", _absent) is not _absent \
                            and obj.get("cot_round"):
                        if int(obj["cot_round"]) != m3.get("cot_round", 0):
                            m3["cot_round"] = int(obj["cot_round"])
                            _VOICE_ROUND["t0"] = time.time()
                    # (no fallback round-guessing any more, 2026-11-07: a push
                    # without cot_round — e.g. StatusIndicator's — used to bump
                    # the round when reasoning text reappeared after a clear,
                    # which drew a duplicate think row in the CoT)
                    if _nt:
                        m3["cot"] = _nt
                    _VOICE_ROUND["prev"] = _nt
                # tool_result: the agent's FunctionCallResultFrame tap pushes
                # it in a DEDICATED field (2026-11-06) — content-sniffing the
                # response was rejected: the final reply can be IDENTICAL to
                # the tool result ("Lights are on, boo." is both), so only an
                # explicit field is unambiguous. This is what flips the voice
                # chip green + settles the card.
                # Progress update (2026-11-07): the agent spoke a short
                # mid-turn line ("Searching the web for …"). Numbered so the
                # UI draws each once, and kept in the turn's ordered step list
                # (with the tool calls) so a refresh redraws it in place.
                # Pre-tool text (agent 'preamble'): kept as an update step
                # BEFORE the tool call it introduced (that call's entry has
                # no result yet), so a refresh redraws it in order. No
                # progress_seq bump: live, it is already a bubble.
                if obj.get("preamble"):
                    _tl = m3.setdefault("_turn_tools", [])
                    _ix = len(_tl)
                    if _tl and _tl[-1].get("kind") != "progress" and not _tl[-1].get("result"):
                        _ix -= 1
                    _tl.insert(_ix, {"kind": "progress", "text": str(obj["preamble"])[:400]})
                if obj.get("progress"):
                    m3["progress_seq"] = m3.get("progress_seq", 0) + 1
                    m3["progress"] = str(obj["progress"])[:200]
                    m3.setdefault("_turn_tools", []).append(
                        {"kind": "progress", "text": m3["progress"],
                         "pseq": m3["progress_seq"]})
                if obj.get("tool_result", _absent) is not _absent:
                    _nres = (obj.get("tool_result") or "")[:200]
                    # tool_ms ONCE, on the result's first arrival, measured from
                    # the tool call's START (2026-11-07). It used to be "now -
                    # LLM round start", recomputed on EVERY push that carried
                    # the result -> 11670 ms shown for a 61 ms tool.
                    if _nres and _nres != m3.get("tool_result"):
                        _t0 = m3.get("_tool_t0") or 0
                        m3["tool_ms"] = int((time.time() - _t0) * 1000) if _t0 else 0
                        # latest TOOL entry (progress items share the list)
                        for _t in reversed(m3.get("_turn_tools") or []):
                            if _t.get("kind") != "progress":
                                _t.update(result=_nres, ms=m3["tool_ms"])
                                break
                    m3["tool_result"] = _nres
                if obj.get("response", _absent) is not _absent:
                    m3["response"] = (obj.get("response") or "")[:4000]
                if obj.get("tool", _absent) is not _absent:
                    _ntool = (obj.get("tool") or "")[:120]
                    _nargs = (obj.get("tool_args") or "")[:400] \
                        if obj.get("tool_args", _absent) is not _absent \
                        else m3.get("tool_args") or ""
                    # VOICE TOOL CHIPS (2026-11-06): the Pi agent pushes tool +
                    # tool_args but never a tool_seq (that was typed-chat only),
                    # so the UI's seq-keyed chip path never fired for voice
                    # turns — no chip, no green settle, no CoT tool row. Detect
                    # the START of a new tool call here (name+args differ from
                    # what's stored) and bump the SAME seq the typed path uses:
                    # result="" at start, the agent's later result push flips
                    # it green. (The agent never pushes a separate "tool done"
                    # event — the result arrives with the next LLM text push.)
                    _new_call = bool(_ntool) and (_ntool != _prev_tool
                                                  or _nargs != (m3.get("tool_args") or ""))
                    if _new_call:
                        m3["tool_seq"] = m3.get("tool_seq", 0) + 1
                        m3["tool_result"] = ""
                        m3["tool_ms"] = 0
                        m3["_tool_t0"] = time.time()
                        # Voice turns keep their tool calls (2026-11-07) so the
                        # history entry carries chips like typed turns do — a
                        # refresh used to draw the chip BELOW the reply.
                        m3.setdefault("_turn_tools", []).append(
                            {"seq": m3["tool_seq"], "name": _ntool,
                             "args": _nargs, "result": "", "ms": 0})
                        _log_tool = f"{_ntool} · {_nargs}" if _nargs else _ntool
                    m3["tool"] = _ntool
                    if _ntool:
                        m3["tool_args"] = _nargs
                # tool_args: what the LLM passed to the tool (e.g. query=...). Only
                # updated by a real push (the heartbeat omits these keys).
                if obj.get("tool_args", _absent) is not _absent:
                    m3["tool_args"] = (obj.get("tool_args") or "")[:400]
                # tool_result/tool_ms: typed chat fills these (the poll needs
                # them to close the chip's spinner). The voice agent never
                # sends them — clear on turn reset (tool key present, empty)
                # so a stale result can't leak into the next turn.
                if obj.get("tool", _absent) is not _absent:
                    if not obj.get("tool"):
                        m3["tool_result"] = ""
                        m3["tool_ms"] = 0
                        # (latency is NOT cleared here any more: every agent
                        # status push carries tool="", so this wiped the voice
                        # readout ~2.5 s after each turn. It resets at the
                        # new-turn boundary above instead.)
                # latency: the voice agent (Pi) can push per-leg timings
                # {"asr","llm","tool","tts","total"} — merged in when present
                # (heartbeat omits it). Today's agent doesn't send it, so
                # the readout stays "–" for voice turns until it does.
                if obj.get("latency", _absent) is not _absent:
                    lat = obj.get("latency")
                    if isinstance(lat, dict):
                        m3["latency"] = {
                            k: lat.get(k)
                            for k in ("asr", "llm", "tool", "tts", "total")
                            if isinstance(lat.get(k), (int, float))}
                    else:
                        m3["latency"] = None
                m3["mic"] = bool(obj.get("mic", False))
                # voice: the VOICE agent's LLM identity (provider/model), pushed
                # with every agent push (2026-11-06). The UI's provider-aware
                # sub-labels read THIS, not s.llm (the typed-chat LLM — a
                # different engine) and not the old hardcoded 'qwen3.8-27b ·
                # vllm'.
                v = obj.get("voice")
                if isinstance(v, dict) and v.get("model"):
                    m3["voice"] = {"provider": str(v.get("provider", "llm")),
                                   "model": str(v.get("model", "llm")),
                                   "gate": bool(v.get("gate", True))}
                # active = agent alive (any push, incl. the 2s heartbeat) OR a
                # non-idle phase. A heartbeat omits 'active', so treat its
                # presence (last_hb set below) as "alive".
                m3["active"] = bool(obj.get("hb") or obj.get("active") or phase != "idle")
                m3["last_hb"] = time.time()   # any push = agent is alive
                m3["ts"] = time.time()
            # Voice turns share the SAME conversation memory as typed chat
            # (the UI chat is a voice proxy — one pipeline, one context).
            # A turn completes when its phase leaves "speaking" with a
            # response present; tool calls are logged as they appear.
            # BUGFIX (2026-11-05): _tool was referenced before assignment when
            # the payload had no "tool" key (every heartbeat / phase push) ->
            # UnboundLocalError crashed the whole /me3 handler AFTER the state
            # update but BEFORE _cmdlog/_append_turn/self._send. Symptoms: the
            # agent's [me3push] "Remote end closed connection" spam, voice
            # turns never appended to chat history, and no command-log lines
            # for voice activity. Now: default empty + log only real tools.
            # ONE log line per tool CALL (2026-11-07): this used to log on every
            # push that carried the tool name -> 11 identical lines per call.
            if _log_tool:
                _cls = ("boots" if _log_tool.startswith("boots_")
                        else "spotify" if _log_tool.startswith("spotify_") else "web")
                _cmdlog(_log_tool[:120], _cls)
            _lp = _VOICE_LAST["phase"]
            _VOICE_LAST["phase"] = phase
            # final (2026-11-07): the agent flags the LLM round that produced
            # SPOKEN text — append that, so a tool turn's placeholder phrase
            # ("Playing music…") never becomes its own history turn. The
            # speaking->listening append below stays as a backstop (greeting,
            # older agents); _append_turn dedupes the repeat.
            if obj.get("final") and m3.get("response"):
                _append_turn(m3.get("heard") or "", m3.get("response") or "",
                             tools=list(m3.get("_turn_tools") or []))
            # Backstop append on speaking->non-speaking: GREETING ONLY now
            # (2026-11-07). Every real reply is flagged 'final' by the agent;
            # with spoken progress updates the phase also leaves 'speaking'
            # MID-turn, and this used to store pre-tool text ("Let me look
            # that up.") as its own turn -> duplicate HEARD after a refresh.
            # The greeting's closing push (greeting flag, phase leaves
            # 'speaking') stores it DIRECTLY — not via the transition: the
            # agent's 2 s heartbeat pushes 'listening' mid-greeting, so the
            # speaking->X transition had already passed without the flag.
            if obj.get("greeting") and phase != "speaking" and m3.get("response"):
                _append_turn(m3.get("heard") or "", m3.get("response") or "")
                _cmdlog(f"voice reply · {m3.get('heard', '')[:40]}", "reply")
            self._send(200, b"ok", "text/plain")
        elif self.path == "/me3mic":
            lvl = obj.get("mic_level", 0.0)
            try:
                lvl = max(0.0, min(1.0, float(lvl)))
            except Exception:
                lvl = 0.0
            with STATE["lock"]:
                STATE["me3"]["mic_level"] = lvl
                STATE["me3"]["last_hb"] = time.time()  # mic activity => alive
            self._send(200, b"ok", "text/plain")
        elif self.path == "/weather":
            # check_weather tool result (web_tools posts it, 2026-11-07): the
            # WEATHER card shows the place actually asked about, not only the
            # home fetch. Body: {"place","temp","condition","code"}.
            try:
                # half-up (round() is banker's: 14.5 -> 14 while Boots said 14.5)
                import math as _math
                temp = int(_math.floor(float(obj.get("temp")) + 0.5))
            except (TypeError, ValueError):
                self._send(400, b"bad temp", "text/plain")
                return
            code = obj.get("code")
            icon, cond = WMO.get(int(code), ("⛅", "")) if isinstance(code, (int, float)) else ("⛅", "")
            with STATE["lock"]:
                STATE["weather_asked"] = {
                    "icon": icon, "temp": temp,
                    "condition": str(obj.get("condition") or cond)[:40],
                    "place": str(obj.get("place") or "")[:60],
                    "ts": time.time()}
            self._send(200, b"ok", "text/plain")
        elif self.path == "/location":
            # Browser-shared GPS (the dashboard's "use my location" pin). The
            # Pi has no built-in GPS, so the dashboard's geolocation feeds it
            # here; we persist it to ~/.me3_location.json which the voice
            # agent's check_weather tool reads (GPS fallback for weather).
            lat = obj.get("lat")
            lon = obj.get("lon")
            try:
                lat = None if lat is None else float(lat)
                lon = None if lon is None else float(lon)
            except (TypeError, ValueError):
                self._send(400, b"bad coords", "text/plain")
                return
            if lat is None or lon is None:
                self._send(400, b"missing lat/lon", "text/plain")
                return
            import os as _os, json as _json
            gps_file = _os.path.join(_os.path.expanduser("~"), ".me3_location.json")
            try:
                with open(gps_file, "w", encoding="utf-8") as f:
                    _json.dump({"lat": lat, "lon": lon,
                                "name": obj.get("name") or "",
                                "ts": time.time()}, f)
            except Exception as e:
                self._send(500, f"save failed: {e!r}".encode(), "text/plain")
                return
            self._send(200, b"ok", "text/plain")
        elif self.path == "/state":
            mode = obj.get("mode")
            if mode in ("idle", "awake"):
                with STATE["lock"]:
                    STATE["mode"] = mode
                    STATE["mode_ts"] = time.time()
                    m = DEV["music"]
                    if mode == "awake":
                        # Wake duck: silence the music so the ack beep and
                        # the command capture are clean (pi_demo holds the
                        # M1A mic; music on the same device would bleed in).
                        if m["playing"] and not m["paused"]:
                            if _music_pause():
                                _DUCK["awake"] = True
                    elif _DUCK["awake"]:
                        # Idle after a wake: resume the ducked music — but only
                        # if the pause was the WAKE-DUCK, not a deliberate user
                        # "pause" (user_paused). A user pause must STAY paused
                        # until "resume music". If a TTS reply is still playing,
                        # defer to its monitor (it resumes when the reply ends).
                        if m.get("user_paused"):
                            _DUCK["awake"] = False  # user owns the pause now
                        elif _TTS["proc"] is not None and _TTS["proc"].poll() is None:
                            _DUCK["defer"] = True
                        elif m["paused"] and _M["proc"] is None:
                            if _music_resume():
                                _DUCK["awake"] = False
                            else:
                                m["paused"] = False
                                _DUCK["awake"] = False
            self._send(200, b"ok", "text/plain")
        elif self.path == "/reminders":
            # Toggle a reminder's done state by index (UI checkbox).
            # Body: {"index": <int>, "done": <bool>}
            idx = obj.get("index")
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                self._send(400, b"bad index", "text/plain")
                return
            with STATE["lock"]:
                if 0 <= idx < len(DEV["reminders"]):
                    DEV["reminders"][idx]["done"] = bool(obj.get("done", True))
                    self._send(200, b"ok", "text/plain")
                else:
                    self._send(404, b"no such reminder", "text/plain")
        elif self.path == "/llm":
            # Switch the chat's LLM provider (UI dropdown). Body:
            # {"provider": "groq"|"hpc"} or a full custom
            # {"base_url","api_key","model"}. Persists to llm.json.
            prov = (obj.get("provider") or "").strip()
            if prov in _LLM_PRESETS:
                cfg = dict(_LLM_PRESETS[prov])
                cfg["provider"] = prov
            elif obj.get("base_url") and obj.get("model"):
                cfg = {"provider": "custom",
                       "base_url": obj["base_url"],
                       "api_key": obj.get("api_key", ""),
                       "model": obj["model"]}
            else:
                self._send(400, b"bad provider", "text/plain")
                return
            # Optional thinking toggle (Groq Qwen3 only): {"thinking":"on"|"off"}
            if obj.get("thinking") in ("on", "off"):
                cfg["thinking"] = obj["thinking"]
            _apply_llm_cfg(cfg)
            # Never persist a PRESET's key into llm.json (it lives in
            # /etc/me3/groq.env; agent.py + _llm_cfg fall back to env when
            # the saved key is empty). Custom endpoints keep theirs.
            _saved = dict(cfg)
            if prov in _LLM_PRESETS:
                _saved["api_key"] = ""
            try:
                with open(_LLM_CFG_PATH, "w", encoding="utf-8") as f:
                    json.dump(_saved, f)
            except Exception:
                pass  # switch still applies for this process
            self._send(200, json.dumps({"ok": True, "provider":
                                        cfg.get("provider"),
                                        "model": cfg.get("model")}).encode(),
                       "application/json")
        elif self.path == "/chat_stream":
            # SSE twin of /chat — the muji pattern (agent.py yields
            # sse("thinking", delta) per chunk; app.js appends each to a
            # live burst). The 1 Hz /ui_state poll is a snapshot clock;
            # SSE is the REAL stream: thinking deltas land in the CoT
            # box the moment the model emits them, tool chips flip
            # start->result live, and the reply types out per delta.
            # /chat stays (stress suite + fallback); this is additive.
            msg = (obj.get("message") or "").strip()
            if not msg or len(msg) > 1000:
                self._send(400, b"bad message", "text/plain")
                return
            if _requests is None:
                self._send(503, b"chat backend unavailable (no requests lib)",
                           "text/plain")
                return
            if getattr(_CHAT_LLM, "provider", "") == "unavailable":
                self._send(503, json.dumps(
                    {"error": "chat engine unavailable (tool import failed)"}
                    ).encode(), "application/json")
                return
            t0 = time.time()
            with STATE["lock"]:
                m3 = STATE["me3"]
                m3["phase"] = "thinking"
                m3["latency"] = None
                m3["heard"] = msg[:400]
                m3["turn"] = m3.get("turn", 0) + 1  # typed turn = fresh id
                m3["thinking"] = ""
                m3["response"] = ""
                m3["tool"] = ""
                m3["tool_args"] = ""
                m3["ts"] = time.time()
            turn_ts = m3["ts"]
            q = queue.Queue()

            def _emit(kind, payload):
                q.put((kind, payload))

            def _worker():
                try:
                    reply, calls, lat = _chat_run(msg, emit=_emit)
                    with STATE["lock"]:
                        m3 = STATE["me3"]
                        m3["phase"] = "idle"
                        m3["response"] = (reply or "")[:4000]
                        if calls:
                            m3["tool"] = calls[-1]["name"][:120]
                            m3["tool_args"] = (
                                json.dumps(calls[-1].get("args") or {})[:400])
                            m3["tool_result"] = calls[-1].get("result", "ok")[:200]
                            m3["tool_ms"] = calls[-1].get("ms", 0)
                        else:
                            m3["tool_result"] = ""
                            m3["tool_ms"] = 0
                        lat = dict(lat)
                        lat["total"] = int((time.time() - t0) * 1000)
                        m3["latency"] = lat
                        m3["ts"] = time.time()
                    # Command log (same as /chat): persisted server-side,
                    # the UI rebuilds from it on refresh.
                    for c in calls:
                        _cls = ("boots" if c["name"].startswith("boots_")
                                else "spotify" if c["name"].startswith("spotify_") else "web")
                        _cmdlog(f"{c['name']} · {c.get('result', 'ok')[:80]}", _cls)
                    _cmdlog(f"chat reply · {int((time.time() - t0) * 1000)} ms", "reply")
                    q.put(("done", {"reply": reply or "", "calls": calls,
                                    "lat": lat, "ts": turn_ts}))
                except Exception as e:
                    _cmdlog("chat error: " + repr(e), "err")
                    with STATE["lock"]:
                        STATE["me3"]["phase"] = "idle"
                    q.put(("error", {"error": repr(e)}))

            threading.Thread(target=_worker, daemon=True).start()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while True:
                    try:
                        kind, payload = q.get(timeout=15)
                    except Exception:
                        # Keepalive comment: a tool round can run 60 s with
                        # no deltas — without this the browser/proxy would
                        # see a silent pipe. The UI ignores comment lines.
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(("event: %s\n" % kind).encode())
                    self.wfile.write(b"data: " +
                                     json.dumps(payload).encode() + b"\n\n")
                    self.wfile.flush()
                    if kind in ("done", "error"):
                        break
            except (BrokenPipeError, ConnectionResetError):
                pass  # tab closed mid-turn; the worker thread finishes on its own
        elif self.path == "/chat":
            # Typed conversation (no STT/TTS). ADDITIVE endpoint: the voice
            # agent's /me3, /me3mic, /fire, /ui_state paths are untouched.
            # Runs the same LLM + tool layer as agent.py (Groq qwen3.8-27b,
            # prompt.py, the three tool modules next to this file's parent).
            msg = (obj.get("message") or "").strip()
            if not msg or len(msg) > 1000:
                self._send(400, b"bad message", "text/plain")
                return
            if _requests is None:
                self._send(503, b"chat backend unavailable (no requests lib)",
                           "text/plain")
                return
            if getattr(_CHAT_LLM, "provider", "") == "unavailable":
                self._send(503, json.dumps(
                    {"error": "chat engine unavailable (tool import failed)"}
                    ).encode(), "application/json")
                return
            t0 = time.time()
            with STATE["lock"]:
                m3 = STATE["me3"]
                m3["phase"] = "thinking"
                # clear the previous turn's readout so stale numbers can't
                # linger while this turn is in flight
                m3["latency"] = None
                m3["heard"] = msg[:400]  # typed text shows as the HEARD bubble
                m3["turn"] = m3.get("turn", 0) + 1  # typed turn = fresh id
                m3["thinking"] = ""
                m3["response"] = ""
                m3["tool"] = ""
                m3["tool_args"] = ""
                m3["ts"] = time.time()
            try:
                reply, calls, lat = _chat_run(msg)
            except Exception as e:
                # Rate-limit headers: Groq tells us exactly which limit
                # tripped and when it resets — log them so a 429 is a
                # fact (tokens vs requests, reset time), not a guess.
                rl = {}
                resp = getattr(e, "response", None)
                if resp is not None:
                    for h in ("x-ratelimit-remaining-requests",
                              "x-ratelimit-remaining-tokens",
                              "x-ratelimit-reset-requests",
                              "x-ratelimit-reset-tokens",
                              "retry-after"):
                        v = resp.headers.get(h)
                        if v is not None:
                            rl[h] = v
                detail = " | " + " ".join(
                    f"{k}={v}" for k, v in sorted(rl.items())) if rl else ""
                _cmdlog("chat error: " + repr(e) + detail, "err")
                # Reset phase: a failed turn used to leave the UI frozen on
                # "thinking" (step bar/ring) until the next successful turn.
                with STATE["lock"]:
                    STATE["me3"]["phase"] = "idle"
                # JSON error body: the UI parses .json() on every /chat
                # response; a plain-text 502 used to surface as a
                # SyntaxError in the command log, hiding the real cause.
                self._send(502, json.dumps(
                    {"error": repr(e), "ratelimit": rl}).encode(),
                    "application/json")
                return
            with STATE["lock"]:
                m3 = STATE["me3"]
                # Chat has no TTS leg — the turn is done the moment the
                # reply text exists. Setting "speaking" here used to freeze
                # the UI on that phase (only the voice agent pushes the
                # done-speaking transition on the Pi), so rest on "idle".
                m3["phase"] = "idle"
                m3["response"] = (reply or "")[:4000]
                if calls:
                    # The per-call loop already pushed this state (with
                    # tool_seq); re-assert it so a poll racing the loop
                    # still sees the final chip as green, not mid-purple.
                    m3["tool"] = calls[-1]["name"][:120]
                    m3["tool_args"] = (
                        json.dumps(calls[-1].get("args") or {})[:400])
                    m3["tool_result"] = calls[-1].get("result", "ok")[:200]
                    m3["tool_ms"] = calls[-1].get("ms", 0)
                else:
                    m3["tool_result"] = ""
                    m3["tool_ms"] = 0
                lat["total"] = int((time.time() - t0) * 1000)
                m3["latency"] = lat
                m3["ts"] = time.time()
            # Command log (persisted server-side; the UI rebuilds from it).
            for c in calls:
                _cls = ("boots" if c["name"].startswith("boots_")
                        else "spotify" if c["name"].startswith("spotify_") else "web")
                _cmdlog(f"{c['name']} · {c.get('result', 'ok')[:80]}", _cls)
            _cmdlog(f"chat reply · {int((time.time() - t0) * 1000)} ms", "reply")
            self._send(200, json.dumps(
                {"reply": reply, "tools": calls,
                 "ms": int((time.time() - t0) * 1000)}).encode(),
                "application/json")
        else:
            self._send(404, b"not found", "text/plain")




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".",
                    help="Pi payload dir (for meta.json / wake_meta.json)")
    ap.add_argument("--port", type=int, default=8330)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    global _PAGE_HTML, MUSIC_DIR
    load_meta(Path(args.dir))
    MUSIC_DIR = Path(args.dir) / "music"
    _music_init()
    threading.Thread(target=_spotify_poll_loop, daemon=True, name="spotify-card").start()
    # The UI page (muji-styled dashboard). Look next to the script first
    # (boots_ui/ is where it lives on the Pi), then in --dir.
    for cand in (Path(__file__).with_name("boots_ui.html"),
                 Path(args.dir) / "boots_ui.html"):
        if cand.exists():
            _PAGE_HTML = cand.read_bytes()
            print(f"UI page: {cand}", flush=True)
            break
    _chat_load()  # eager: /ui_state reports llm state from the first poll
    threading.Thread(target=_weather_loop, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ME2 UI: http://{args.host}:{args.port}  (package dir {args.dir})",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
