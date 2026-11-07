#!/usr/bin/env python3
# Boots UI  (was "ME2 UI") â€” LLM -> Boots UI command backend.
#
# The 10 real ME2/Boots-UI commands (spec: ME2/src/commands.py "V3 phrasebook",
# single source of truth) are exposed here as LLM function-calling tools. Each
# tool POSTs a fired command to the Boots UI server (boots_ui.py) on :8330
# (POST /fire  {"cmd": <CMD>, "slot": <value>, "conf": 1.0}).
#
# The Boots UI server is a SEPARATE component on the Pi (ME2 stack: pi_demo.py
# wake loop + boots_ui.py :8330). ME3 does NOT need it running to stay robust:
# if :8330 is unreachable it returns a clean spoken "Boots UI backend is
# offline" message instead of raising (fail-soft), so a dead backend never
# adds to voice latency or crashes the agent.
#
# NOTE (D11): pipecat 1.12's MCPService is incompatible with the installed `mcp`
# package on this Python, so agent.py uses a DIRECT OpenAI function-calling
# layer. This file provides the canonical tool specs + a real dispatch().
#
# The parametric commands use a FIXED 3-value V3 vocabulary (the spec's "3 locked
# values" per class) â€” see V3 below. Tools expose those as enum choices so the
# LLM only ever sends a key the Boots UI server understands.
#
import json
import os
import urllib.request

# Where the Boots UI server lives. Overridable via env so ME3 works whether the
# Boots UI server is on the same Pi, another host, or behind the SHARC VPN.
BOOTS_UI_HOST = os.environ.get("BOOTS_UI_HOST", "127.0.0.1")
BOOTS_UI_PORT = int(os.environ.get("BOOTS_UI_PORT", "8330"))
BOOTS_UI_TIMEOUT = float(os.environ.get("BOOTS_UI_TIMEOUT", "1.0"))  # < voice latency budget

# Offline message the LLM will speak if the Boots UI backend is unreachable.
OFFLINE_MSG = ("The Boots UI backend is offline right now, so I couldn't run "
               "that command. Please make sure the device UI server is running.")

# V3 slot vocabulary (single source of truth: ME2/pi/boots_ui.py lines 70-74).
V3 = {
    "TIMER":       ["10 seconds", "30 seconds", "1 minute"],
    "ALARM":       ["6 AM", "8 AM", "9 PM"],
    "TEMPERATURE": ["18 degrees", "22 degrees", "26 degrees"],
    "BRIGHTNESS":  ["20 percent", "60 percent", "100 percent"],
    "COLOR":       ["red", "blue", "green"],
}



# ---- canonical tool definitions (10 real Boots UI intents -> /fire cmd + slot) ----
# (name, description, params)  -- params is the OpenAI JSON-schema "properties".
# Enum params are constrained to the V3 vocabulary so the LLM only sends valid keys.
TOOL_SPECS = [
    ("boots_play_music",    "Play music on the device.", {}),
    ("boots_pause",         "Pause the music.", {}),
    ("boots_resume",        "Resume the music.", {}),
    ("boots_next",          "Skip to the next song.", {}),
    ("boots_stop",          "Stop the music.", {}),
    ("boots_volume_up",     "Turn the volume up.", {}),
    ("boots_volume_down",   "Turn the volume down.", {}),
    ("boots_set_volume",    "Set the volume to a level (0-100).",
                            {"level": {"type": "integer", "minimum": 0, "maximum": 100}}),
    ("boots_lights",        "Turn the lights on or off.",
                            {"action": {"type": "string", "enum": ["on", "off"]}}),
    ("boots_set_bright",    "Set the light brightness to any level from 0 to 100 percent.",
                            {"level": {"type": "string"}}),
    ("boots_set_color",     "Set the light color (any color name).",
                            {"color": {"type": "string"}}),
    ("boots_set_timer",     "Set a countdown timer for any duration.",
                            {"duration": {"type": "string"}}),
    ("boots_set_alarm",     "Set an alarm for any time of day.",
                            {"time": {"type": "string"}}),
    ("boots_set_temperature", "Set the room temperature to any target.",
                            {"degrees": {"type": "string"}}),
    ("boots_set_reminder",  "Set a reminder.",
                            {"text": {"type": "string"}}),
    ("boots_make_call",     "Place a phone call.",
                            {"number": {"type": "string"}}),
]

# Map each LLM tool name -> the Boots UI /fire command it fires.
TOOL_TO_FIRE = {
    "boots_play_music":      ("PLAY_MUSIC",      None),
    "boots_pause":           ("PAUSE",           None),
    "boots_resume":          ("RESUME",          None),
    "boots_next":            ("NEXT",            None),
    "boots_stop":            ("STOP",            None),
    "boots_volume_up":       ("VOLUME_UP",       None),
    "boots_volume_down":     ("VOLUME_DOWN",     None),
    "boots_set_volume":      ("SET_VOLUME",      "level"),
    "boots_lights":          ("_LIGHTS",         "action"),   # on/off -> LIGHT_ON/LIGHT_OFF
    "boots_set_bright":      ("BRIGHTNESS",      "level"),
    "boots_set_color":       ("COLOR",           "color"),
    "boots_set_timer":       ("TIMER",           "duration"),
    "boots_set_alarm":       ("ALARM",           "time"),
    "boots_set_temperature": ("TEMPERATURE",     "degrees"),
    "boots_set_reminder":    ("CREATE_REMINDER", "text"),
    "boots_make_call":       ("CALL",            "number"),
}




import re as _re


def _pct(slot):
    '''free-form -> int 0-100 (first number, sign included), clamped. None if no number.'''
    m = _re.search(r'-?\d+', '' if slot is None else str(slot))
    if not m: return None
    return max(0, min(100, int(m.group())))


def _num(slot):
    # `'' if slot is None` — NOT `slot or ''`: a legit 0 is falsy and
    # `0 or ''` -> '' -> None, which lied about "volume 0".
    m = _re.search(r'-?\d+', '' if slot is None else str(slot))
    return int(m.group()) if m else None


def _confirm(cmd, slot, requested=None):
    '''Value-specific, speakable confirmation the LLM reads back. `slot` is the
    APPLIED value when the server reported one (fired:<value>).'''
    try:
        if cmd == 'TIMER' and str(slot).isdigit():
            return f'Timer set for {_human_secs(slot)}.'
        if cmd == 'ALARM' and _re.fullmatch(r'\d{2}:\d{2}', str(slot)):
            h, mm = (int(x) for x in str(slot).split(':'))
            return f"Alarm set for {h % 12 or 12}:{mm:02d} {'AM' if h < 12 else 'PM'}."
        if cmd == 'TEMPERATURE':
            d, r = _num(slot), _num(requested)
            if d is not None and r is not None and r != d:
                return f"Temperature set to {d} degrees, the {'highest' if r > d else 'lowest'} it goes."
        # Music: natural confirmations (were "PLAY_MUSIC done." etc).
        if cmd == 'PLAY_MUSIC':
            return f'Playing {slot}.' if slot else 'Playing music.'
        if cmd == 'NEXT':
            return f'Next song: {slot}.' if slot else 'Skipping to the next song.'
        if cmd == 'PAUSE':
            return 'Music paused.'
        if cmd == 'RESUME':
            return 'Music resumed.'
        if cmd == 'STOP':
            return 'Music stopped.'
        if cmd in ('VOLUME_UP', 'VOLUME_DOWN'):
            v = _num(slot)
            word = 'up' if cmd == 'VOLUME_UP' else 'down'
            return f'Volume {word} to {v} percent.' if v is not None else f'Volume {word}.'
        if cmd == 'SET_VOLUME':
            v, r = _num(slot), _num(requested)
            if v is not None and r is not None and r < v:
                return f'Volume set to {v} percent, the lowest I go so you can still hear me.'
            if v is not None and r is not None and r > v:
                return f'Volume set to {v} percent, the maximum.'
        if cmd == 'BRIGHTNESS':
            p = _pct(slot)
            return f'Brightness set to {p} percent.' if p is not None else 'Brightness updated.'
        if cmd == 'TEMPERATURE':
            d = _num(slot)
            return f'Temperature set to {d} degrees.' if d is not None else 'Temperature updated.'
        if cmd == 'SET_VOLUME':
            v = _num(slot)
            # `is not None` — a legit 0 is falsy; `if v` would say "updated"
            # for "volume 0" and lie about the value. Clamp to 0-100 so the
            # spoken value matches what the device actually does (the server
            # clamps too — an unclamped confirm said "250 percent" for 250).
            v = max(0, min(100, v)) if v is not None else None
            return f'Volume set to {v} percent.' if v is not None else 'Volume updated.'
        if cmd == 'COLOR':
            c = (slot or '').strip()
            return f'Light color set to {c}.' if c else 'Light color updated.'
        if cmd == 'TIMER':
            c = (slot or '').strip()
            return f'Timer set for {c}.' if c else 'Timer set.'
        if cmd == 'ALARM':
            c = (slot or '').strip()
            return f'Alarm set for {c}.' if c else 'Alarm set.'
        if cmd == 'LIGHT_ON':
            return 'Lights on.'
        if cmd == 'LIGHT_OFF':
            return 'Lights off.'
        if cmd == 'CREATE_REMINDER':
            c = (slot or '').strip()
            return f'Reminder set: {c}.' if c else 'Reminder set.'
        if cmd == 'CALL':
            c = (slot or '').strip()
            return f'Calling {c}.' if c else 'Calling.'
    except Exception:
        pass
    return f'{cmd} done.'


def _fire(cmd: str, slot=None) -> str:
    """POST one command to the Boots UI server (:8330/fire). Returns outcome.

    Fail-soft: on any network/HTTP error returns the OFFLINE_MSG (the LLM will
    speak it) instead of raising.
    """
    body = {"cmd": cmd, "slot": slot, "conf": 1.0}
    url = f"http://{BOOTS_UI_HOST}:{BOOTS_UI_PORT}/fire"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=BOOTS_UI_TIMEOUT) as r:
            resp = r.read().decode("utf-8", "replace")
    except Exception:
        return OFFLINE_MSG
    # Confirm what the device ACTUALLY did (2026-11-07 stress test): the
    # server answers "ok:fired:<applied value>" or "ok:invalid". The old
    # confirmation echoed the REQUEST, so "Light color set to chartreuse",
    # "Timer set for 1 hour" (1 s), "Alarm set for noon" (nothing) were lies.
    outcome = resp[3:] if resp.startswith("ok:") else resp
    if outcome == "invalid":
        return _ask_again(cmd, slot)
    if outcome == "nothing":
        # (2026-11-07) the local player had nothing to act on. Spotify is the
        # usual music source now (the LLM often picks the local tool for it):
        # hand the command to Spotify; if Spotify has nothing either, say so.
        sp_tool = {"PAUSE": "spotify_pause", "RESUME": "spotify_resume",
                   "NEXT": "spotify_next", "STOP": "spotify_pause"}.get(cmd)
        if sp_tool:
            try:
                import spotify_tools
                msg = spotify_tools.dispatch(sp_tool)
                if not _re.match(r"(Spotify isn't|Nothing is playing|Spotify didn't|Spotify auth)", msg):
                    return msg
            except Exception:
                pass
        return {
            "PAUSE": "No local music is playing, so there was nothing to pause here. If it's Spotify, I can pause that.",
            "RESUME": "There's no local music paused. If you mean Spotify, I can resume that.",
            "STOP": "No local music is playing. If it's Spotify, I can pause that.",
            "NEXT": "No local music is playing. If it's Spotify, I can skip the Spotify track.",
        }.get(cmd, "There was nothing to do.")
    if outcome == "nomedia":
        return ("I can't play local music: there are no song files on the device. "
                "I can play it on Spotify instead if you tell me what to play.")
    applied = outcome[6:] if outcome.startswith("fired:") else None
    return _confirm(cmd, applied if applied else slot, requested=slot)


def _ask_again(cmd, slot):
    """Spoken line when the device could not read the value (nothing changed)."""
    s = str(slot or "").strip()
    return {
        "BRIGHTNESS": f"I couldn't read '{s}' as a brightness. Give me a percent, like 40 percent.",
        "COLOR": f"I don't know the color '{s}'. Try a common color name, like teal or orange.",
        "TEMPERATURE": f"I couldn't read '{s}' as a temperature. Say a number of degrees, like 22.",
        "TIMER": f"I couldn't tell how long '{s}' is. Say something like 5 minutes.",
        "ALARM": f"I couldn't read '{s}' as a time. Say something like 7:30 AM.",
        "CREATE_REMINDER": "What should the reminder say?",
        "CALL": "Who should I call? I need a number or a name.",
        "SET_VOLUME": f"I couldn't read '{s}' as a volume. Say a number from 10 to 100.",
    }.get(cmd, "I couldn't do that. Could you say it another way?")


def _human_secs(n):
    """3600 -> '1 hour', 90 -> '1 minute 30 seconds', 120 -> '2 minutes'."""
    n = int(n)
    h, rem = divmod(n, 3600)
    m, s = divmod(rem, 60)
    parts = [f"{v} {u}{'' if v == 1 else 's'}" for v, u in ((h, "hour"), (m, "minute"), (s, "second")) if v]
    return " ".join(parts) or "0 seconds"


def dispatch(tool: str, **kw) -> str:
    """Execute a Boots UI command by firing it at :8330. Returns a string the
    LLM can speak back to the user.

    `tool` is the tool name (e.g. "boots_lights"); `kw` holds the tool's params
    (e.g. action="on"). The first param is named `tool` (not `action`) so it
    never collides with a tool param named `action` (boots_lights).
    """
    mapping = TOOL_TO_FIRE.get(tool)
    if mapping is None:
        return f"I don't know how to do '{tool}'."
    cmd, slot_key = mapping
    slot = kw.get(slot_key) if slot_key else None

    if cmd == "_LIGHTS":
        a = (slot or "").strip().lower()
        return _fire("LIGHT_ON" if a == "on" else "LIGHT_OFF")
    return _fire(cmd, slot=slot)


# NOTE: the standalone FastMCP server that used to live here was removed —
# the canonical MCP server is mcp_server.py (one process, all tools).
# This module is consumed via its TOOL_SPECS + dispatch() by both the agent
# (direct function-calling, D11) and mcp_server.py (MCP facade).
