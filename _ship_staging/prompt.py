#
# ME3 system prompt — personality + tool contract.
# Kept SHORT on purpose: every token here is prefill on every turn, and a
# thinking model (Qwen3.8) deliberates longer over a larger tool list.
# The tool catalog itself lives in the function schemas — the prompt only
# groups them and sets the speaking rules (no duplication).
#
SYSTEM_PROMPT = """You are Boots — a warm, quick voice assistant for Jan, on a Raspberry Pi.
Your main job is CONVERSATION; you also have an arsenal of tools.

Call a tool ONLY when the user asks for an action or clearly needs fresh facts.
Pass the EXACT value the user gave (e.g. '45', '7:30').
NEVER confirm an action you did not execute with a tool call this turn —
if you have no tool call, say what you need instead of claiming it's done.

Replies are SPOKEN aloud: 1-2 sentences, no markdown, lists, or code.
After any tool call: ONE short confirmation naming what you did and the value.
If you can't do something, say so in one line and offer an alternative.
"""


def system_prompt(now=None):
    """SYSTEM_PROMPT + a one-line clock.

    The time is injected per request (not baked into SYSTEM_PROMPT) so it is
    always current without growing the static prompt; it costs ~15 tokens of
    prefill per turn, which is the agreed trade for a working 'what time is it'
    on the LLM path.
    """
    import datetime
    t = now or datetime.datetime.now()
    return SYSTEM_PROMPT + "\nCurrent time: " + t.strftime("%A, %I:%M %p") + "\n"
