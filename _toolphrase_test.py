import json, time, urllib.request

# Replicate the EXACT logic the deployed agent.py now uses for a pure
# function-call turn (no LLM text tokens):
#   1) FunctionCallsStartedFrame -> _RECENT_TEXT['thinking'] = _tool_phrase(tool)
#      and _RECENT_TEXT['reply'] = _tool_phrase(tool)  (if no streamed text)
#   2) _me3_push_now posts thinking + response + tool + tool_args
#   3) FunctionCallResultFrame -> reply updated with the tool result
# Then POST to the live UI and read back /ui_state to prove the bubbles get text.

TOOL_PHRASE = {
    "boots_set_timer": "Setting a timer",
    "boots_set_alarm": "Setting an alarm",
    "spotify_play_search": "Searching Spotify and playing",
}

RECENT = {"transcript": "set a timer for one minute", "reply": None,
          "tool": "boots_set_timer", "tool_args": "duration='1 minute'",
          "thinking": None}


def tool_phrase(tool_name):
    if not tool_name:
        return ""
    base = TOOL_PHRASE.get(tool_name, tool_name)
    args = RECENT.get("tool_args")
    return f"{base}{(' — ' + args) if args else ''}…"


# --- FunctionCallsStartedFrame handling (mirror agent.py) ---
primary = RECENT["tool"]
thinking_buf = []  # empty: pure function-call turn, no LLM text
if not (thinking_buf and "".join(thinking_buf).strip()):
    RECENT["thinking"] = tool_phrase(primary)
if not (RECENT.get("reply") or "").strip():
    RECENT["reply"] = tool_phrase(primary)

# --- _me3_push_now body (mirror agent.py) ---
thinking = ("".join(thinking_buf) or RECENT.get("thinking") or "")
phase = "tool"
body = {
    "phase": phase,
    "heard": (RECENT.get("transcript") or "")[:300],
    "thinking": thinking[:400],
    "response": (RECENT.get("reply") or "")[:400],
    "tool": (RECENT.get("tool") or "")[:120],
    "tool_args": (RECENT.get("tool_args") or "")[:400],
    "mic": False, "active": True,
}

print("AGENT WOULD POST:")
print("  phase     =", repr(body["phase"]))
print("  heard     =", repr(body["heard"]))
print("  thinking  =", repr(body["thinking"]))
print("  response  =", repr(body["response"]))
print("  tool      =", repr(body["tool"]))
print("  tool_args =", repr(body["tool_args"]))


def post(b):
    req = urllib.request.Request(
        "http://127.0.0.1:8330/me3", data=json.dumps(b).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=2) as r:
        r.read()


post(body)
time.sleep(0.6)
with urllib.request.urlopen("http://127.0.0.1:8330/ui_state", timeout=2) as r:
    m = json.load(r)["me3"]

print("\nUI /ui_state.me3 AFTER POST:")
print("  phase     =", repr(m.get("phase")))
print("  thinking  =", repr(m.get("thinking")))
print("  response  =", repr(m.get("response")))
print("  tool      =", repr(m.get("tool")))
print("  tool_args =", repr(m.get("tool_args")))

ok = (m.get("thinking")  # thinking bubble non-blank
      and m.get("response")  # assistant bubble has text
      and "timer" in m.get("thinking", "")
      and "timer" in m.get("response", ""))
print("\nTOOL_TURN_BUBBLES:", "PASS" if ok else "FAIL")

# reset the live UI to listening
post({"phase": "listening", "heard": "", "thinking": "", "response": "",
      "tool": "", "tool_args": "", "mic": True, "active": True})
print("reset to listening")
