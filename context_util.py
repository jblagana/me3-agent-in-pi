#!/usr/bin/env python3
# Boots -- conversation window helpers shared by agent.py and the dashboard
# chat (2026-11-18).
#
# The window used to be "the last 24 messages". One website build is about
# 12 messages (searches, fetches, the build), so a single build pushed the
# earlier conversation out. Now the window is the last N USER TURNS, and the
# bulky parts of older turns (long tool results, the HTML inside a
# write_website/build_app call) are shortened, so many turns fit in fewer
# tokens than the old 24 messages often took.
#
# Messages are OpenAI-style dicts. Pairing rules the provider enforces are
# kept: an assistant message with tool_calls is never separated from its
# tool results, and the window always starts on a user message.
import json

OLD_RESULT_CHARS = 400     # tool results in turns older than the last KEEP_FULL
OLD_ARGS_CHARS = 600       # tool-call arguments in those turns
KEEP_FULL = 2              # newest user turns kept word for word
SHORT_NOTE = " …(older result, shortened)"


def _role(m):
    return (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) or ""


def _is_note(m, note_prefix):
    return bool(note_prefix) and isinstance(m.get("content"), str) and m["content"].startswith(note_prefix)


def user_turn_starts(msgs, note_prefix=""):
    """Indexes of real user messages (background job notes don't count)."""
    return [i for i, m in enumerate(msgs)
            if isinstance(m, dict) and _role(m) == "user" and not _is_note(m, note_prefix)]


def window_start(msgs, max_turns, note_prefix=""):
    """Index where the last `max_turns` user turns begin (0 = keep all)."""
    starts = user_turn_starts(msgs, note_prefix)
    if max_turns <= 0 or len(starts) <= max_turns:
        return 0
    return starts[-max_turns]


def shorten_old(msgs, keep_full=KEEP_FULL, note_prefix=""):
    """Shorten tool results and big tool-call arguments in every turn before
    the newest `keep_full` user turns. Edits the dicts in place and returns
    how many were shortened."""
    starts = user_turn_starts(msgs, note_prefix)
    if len(starts) <= keep_full:
        return 0
    cut = starts[-keep_full]
    n = 0
    for m in msgs[:cut]:
        if not isinstance(m, dict):
            continue
        r = _role(m)
        c = m.get("content")
        if r == "tool" and isinstance(c, str) and len(c) > OLD_RESULT_CHARS + len(SHORT_NOTE):
            m["content"] = c[:OLD_RESULT_CHARS] + SHORT_NOTE
            n += 1
        elif r == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function") if isinstance(tc, dict) else None
                args = fn.get("arguments") if isinstance(fn, dict) else None
                if isinstance(args, str) and len(args) > OLD_ARGS_CHARS:
                    # Keep it valid JSON: the short fields survive, long
                    # strings (HTML, section bodies) are cut.
                    try:
                        obj = json.loads(args)
                        if isinstance(obj, dict):
                            obj = {k: (v[:120] + "…" if isinstance(v, str) and len(v) > 120 else
                                       ("[…]" if isinstance(v, (list, dict)) else v))
                                   for k, v in obj.items()}
                            fn["arguments"] = json.dumps(obj, ensure_ascii=False)
                        else:
                            fn["arguments"] = "{}"
                    except Exception:
                        fn["arguments"] = "{}"
                    n += 1
    return n


# ---------------------------------------------------------------------------
# Stray </think> (2026-11-18). With reasoning off, Qwen3.8 sometimes writes
# its answer, then "</think>", then the same answer again; the voice spoke it
# twice and the chat showed the tag. Rule (same for voice and chat): text
# before the first </think> is kept if it has words; everything after it is
# dropped. If nothing came before the tag, the tag is removed and the text
# after it is kept. <think> opening tags are removed. Works on a stream:
# a tag split across chunks ("</th" + "ink>") is held back until complete.
# ---------------------------------------------------------------------------
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def _words(s):
    import re
    return len(re.findall(r"[^\W\d_]{2,}", s or ""))


def _partial_tag_suffix(s):
    """Length of a trailing piece of s that could be the start of a tag."""
    for k in range(min(len(s), len(THINK_CLOSE) - 1), 0, -1):
        tail = s[-k:]
        if THINK_CLOSE.startswith(tail) or THINK_OPEN.startswith(tail):
            return k
    return 0


class ThinkFilter:
    """Feed reply text chunk by chunk; returns the text to show/speak."""

    def __init__(self):
        self.said = ""
        self.pending = ""
        self.off = False          # True once text after </think> is being dropped
        self.dropped = False      # True if a tag was seen at all

    def feed(self, chunk):
        if self.off:
            if chunk:
                self.dropped = True
            return ""
        s = self.pending + (chunk or "")
        self.pending = ""
        out = ""
        while s and not self.off:
            if THINK_OPEN in s:
                s = s.replace(THINK_OPEN, "")
                self.dropped = True
            i = s.find(THINK_CLOSE)
            if i < 0:
                k = _partial_tag_suffix(s)
                if k:
                    self.pending, s = s[-k:], s[:-k]
                out += s
                self.said += s
                break
            self.dropped = True
            head, s = s[:i], s[i + len(THINK_CLOSE):]
            out += head
            self.said += head
            if _words(self.said) >= 2:
                self.off = True
        return out


def clean_reply(text):
    """Whole-text version of ThinkFilter (the chat bubble and history)."""
    f = ThinkFilter()
    out = f.feed(text)
    if not f.off and f.pending and not (THINK_CLOSE.startswith(f.pending) or THINK_OPEN.startswith(f.pending)):
        out += f.pending
    return out
