#!/usr/bin/env python3
# Boots UI  (was "ME2 UI") — LLM -> Boots UI command backend.
#
# The 10 real ME2/Boots-UI commands (spec: ME2/src/commands.py "V3 phrasebook",
# single source of truth) are exposed here as LLM function-calling tools. Each
# tool POSTs a fired command to the Boots UI server (me2_ui.py) on :8330
# (POST /fire  {"cmd": <CMD>, "slot": <value>, "conf": 1.0}).
#
# The Boots UI server is a SEPARATE component on the Pi (ME2 stack: pi_demo.py
# wake loop + me2_ui.py :8330). ME3 does NOT need it running to stay robust:
# if :8330 is unreachable it returns a clean spoken "Boots UI backend is
# offline" message instead of raising (fail-soft), so a dead backend never
# adds to voice latency or crashes the agent.
#
# NOTE (D11): pipecat 1.12's MCPService is incompatible with the installed `mcp`
# package on this Python, so agent.py uses a DIRECT OpenAI function-calling
# layer. This file provides the canonical tool specs + a real dispatch().
#
# The parametric commands use a FIXED 3-value V3 vocabulary (the spec's "3 locked
# values" per class) — see V3 below. Tools expose those as enum choices so the
# LLM only ever sends a key the Boots UI server understands.
#
import json
import os
import sys
import urllib.request

# Where the Boots UI server lives. Overridable via env so ME3 works whether the
# Boots UI server is on the same Pi, another host, or behind the SHARC VPN.
BOOTS_UI_HOST = os.environ.get("ME2_UI_HOST", "127.0.0.1")
BOOTS_UI_PORT = int(os.environ.get("ME2_UI_PORT", "8330"))
BOOTS_UI_TIMEOUT = float(os.environ.get("ME2_UI_TIMEOUT", "1.0"))  # < voice latency budget

# Offline message the LLM will speak if the Boots UI backend is unreachable.
OFFLINE_MSG = ("The Boots UI backend is offline right now, so I couldn't run "
               "that command. Please make sure the device UI server is running.")

# V3 slot vocabulary (single source of truth: ME2/pi/me2_ui.py lines 70-74).
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
    ("boots_set_bright",    "Set the light brightness.",
                            {"level": {"type": "string", "enum": V3["BRIGHTNESS"]}}),
    ("boots_set_color",     "Set the light color.",
                            {"color": {"type": "string", "enum": V3["COLOR"]}}),
    ("boots_set_timer",     "Set a countdown timer.",
                            {"duration": {"type": "string", "enum": V3["TIMER"]}}),
    ("boots_set_alarm",     "Set an alarm.",
                            {"time": {"type": "string", "enum": V3["ALARM"]}}),
    ("boots_set_temperature", "Set the room temperature.",
                            {"degrees": {"type": "string", "enum": V3["TEMPERATURE"]}}),
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
            r.read()
        return f"OK: {cmd}"
    except Exception:
        return OFFLINE_MSG


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


# Standalone MCP wrapper (kept for reference; unused by agent.py, which uses the
# direct function-calling layer).
try:
    from mcp.server.fastmcp import FastMCP
    HAVE_MCP = True
except Exception:
    HAVE_MCP = False

mcp = None
if HAVE_MCP:
    mcp = FastMCP("boots-ui")

    # mcp 1.30-compatible wrapper: build a properly-typed function per tool so
    # FastMCP can introspect its signature (no *args / no leading-underscore
    # params). Every Boots UI tool has zero or one simple slot param.
    def _register_tool(name, desc, params):
        if not params:
            def fn():
                """_doc_"""
                return dispatch(name)
            fn.__name__ = name
            fn.__doc__ = desc
            mcp.add_tool(fn)
        else:
            (pname, pschema), = params.items()
            def fn(value: str):
                """_doc_"""
                return dispatch(name, **{pname: value})
            fn.__name__ = name
            fn.__doc__ = desc
            mcp.add_tool(fn)

    for (name, desc, params) in TOOL_SPECS:
        _register_tool(name, desc, params)


if __name__ == "__main__":
    if HAVE_MCP and mcp:
        mcp.run()
    else:
        print("FastMCP not available; this file is used via agent.py's function-calling layer.",
              file=sys.stderr)
