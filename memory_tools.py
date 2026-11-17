#!/usr/bin/env python3
# Boots -- lasting notes about the user (2026-11-18).
#
# The conversation itself is trimmed and lost on restart; facts the user
# wants kept ("call me Jan", "I'm vegetarian", "my thesis is on solar
# dryers") are saved here and added to the system prompt of every call,
# voice and typed chat alike. Same contract as the other tool modules:
# TOOL_SPECS + dispatch(), short speakable results.
#
# File: BOOTS_MEMORY (default ~/.boots_memory.json), a JSON list of
# {"fact": str, "t": epoch}. Capped so the prompt cost stays small
# (MAX_FACTS facts, PROMPT_CHARS characters in the prompt).
import json
import os
import re
import time

MEMORY_PATH = os.environ.get("BOOTS_MEMORY") or os.path.join(os.path.expanduser("~"), ".boots_memory.json")
MAX_FACTS = 40
MAX_FACT_CHARS = 200
PROMPT_CHARS = 1200

# Never keep secrets, even if asked: the file is plain text and goes to the LLM.
_SECRET = re.compile(r"\b(password|passcode|pin|otp|one[- ]time code|api key|token|secret|"
                     r"credit card|card number|cvv|bank account|account number)\b", re.I)


# "Remember that I like OPM" -- seen live 2026-11-18: Qwen replied "Got it,
# I'll remember" WITHOUT calling remember, so nothing was saved. Requests
# like this force the remember call on the turn's first round (agent.py and
# the dashboard chat). Questions ("do you remember...", "what do you
# remember") and to-dos ("remember to call mom at 5", a reminder) don't.
_REMEMBER_ASK = re.compile(r"\b(remember|don'?t forget|keep in mind|take note|make a note|note that)\b", re.I)
_REMEMBER_NOT = re.compile(r"^\s*(do|did|can|could|what|who|when|where|how|why|which)\b|"
                           r"\bremember (to|when|me)\b|\bremind\b|\?\s*$|"
                           r"\b(can.?t|cannot|couldn.?t|don.?t|didn.?t) remember\b", re.I)


def wants_remember(text: str) -> bool:
    t = text or ""
    return bool(_REMEMBER_ASK.search(t)) and not _REMEMBER_NOT.search(t)


def _load():
    try:
        with open(MEMORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return [d for d in data if isinstance(d, dict) and d.get("fact")] if isinstance(data, list) else []
    except Exception:
        return []


def _save(facts):
    tmp = MEMORY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=0)
    os.replace(tmp, MEMORY_PATH)


def prompt_block() -> str:
    """The saved facts as a system-prompt block, newest kept when over the cap."""
    facts = _load()
    if not facts:
        return ""
    lines, used = [], 0
    for d in reversed(facts):
        line = "- " + d["fact"]
        if used + len(line) > PROMPT_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    return ("\n\nWHAT YOU KNOW ABOUT THE USER (saved with remember; use it naturally, don't "
            "recite it):\n" + "\n".join(reversed(lines)))


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip()).rstrip(".")


def _tool_remember(fact=""):
    f = _norm(fact)[:MAX_FACT_CHARS]
    if not f:
        return "Tell me what to remember."
    if _SECRET.search(f):
        return "Not saved: I don't keep passwords, codes, keys or card and account numbers."
    facts = _load()
    low = f.lower()
    if any(d["fact"].lower() == low for d in facts):
        return f"I already remember that: {f}."
    facts.append({"fact": f, "t": time.time()})
    dropped = len(facts) - MAX_FACTS
    if dropped > 0:
        facts = facts[dropped:]
    _save(facts)
    return f"Saved: {f}." + (" (I dropped the oldest note to make room.)" if dropped > 0 else "")


def _tool_forget(fact=""):
    q = _norm(fact).lower()
    facts = _load()
    if not facts:
        return "I don't have anything saved about you."
    if q in ("all", "everything", "all of it"):
        _save([])
        return f"Done, I forgot all {len(facts)} saved notes."
    words = [w for w in re.findall(r"\w+", q) if len(w) > 2]
    keep, gone = [], []
    for d in facts:
        low = d["fact"].lower()
        hit = q and (q in low or (words and sum(w in low for w in words) >= max(1, (len(words) + 1) // 2)))
        (gone if hit else keep).append(d)
    if not gone:
        return f"I don't have a note about {fact}."
    _save(keep)
    return "Forgot: " + "; ".join(d["fact"] for d in gone[:3]) + ("." if len(gone) <= 3 else f", and {len(gone) - 3} more.")


def _tool_list_memories():
    facts = _load()
    if not facts:
        return "I don't have anything saved about you yet."
    shown = "; ".join(d["fact"] for d in facts[-12:])
    return f"I remember {len(facts)} thing{'s' if len(facts) != 1 else ''}: {shown}."


TOOL_SPECS = [
    ("remember", "Save a lasting fact about the user or something they ask you to remember "
                 "(name, preferences, ongoing projects). One short sentence.",
     {"fact": "string"}),
    ("forget", "Delete a saved note about the user (fact='all' deletes every note).",
     {"fact": "string"}),
    ("list_memories", "Say what you have saved about the user.", {}),
]

NAMES = {n for n, _d, _p in TOOL_SPECS}

_DISPATCH = {
    "remember": lambda **kw: _tool_remember(kw.get("fact", "")),
    "forget": lambda **kw: _tool_forget(kw.get("fact", "")),
    "list_memories": lambda **kw: _tool_list_memories(),
}


def dispatch(tool: str, **kw) -> str:
    fn = _DISPATCH.get(tool)
    if fn is None:
        return f"I don't know how to do '{tool}'."
    try:
        return fn(**kw)
    except Exception as e:
        return f"Something went wrong with my notes: {type(e).__name__}."
