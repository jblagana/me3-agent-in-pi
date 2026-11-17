"""Context window and saved notes (2026-11-18).

1. memory   remember saves one short fact, ignores repeats, refuses secrets,
            keeps at most MAX_FACTS; forget removes by words or 'all';
            the prompt block lists the facts and stays under its cap.
2. window   the window counts USER TURNS (background job notes don't
            count), so a tool-heavy turn no longer pushes older turns out.
3. shorten  turns older than the last two get their tool results and big
            tool-call arguments cut; arguments stay valid JSON; the newest
            two turns are untouched.
4. agent    _trim_llm_context keeps 8 turns, never drops the current
            request, never starts on a tool message; the saved notes reach
            the system instruction; memory tools are registered.

Offline: no network, no LLM calls.

    cd ~/me3 && .venv/bin/python tests/test_context.py
"""
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
TMP = tempfile.mkdtemp(prefix="boots-ctx-test-")
os.environ["BOOTS_MEMORY"] = os.path.join(TMP, "memory.json")
os.environ["BOOTS_LIBRARY"] = os.path.join(TMP, "library")

import memory_tools as mem    # noqa: E402
import context_util as cu     # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def memory_case():
    out = mem.dispatch("remember", fact="Call the user Jan.")
    check("memory: saved", out.startswith("Saved") and "Call the user Jan" in mem.prompt_block(), out)
    again = mem.dispatch("remember", fact="call the user jan")
    check("memory: repeat ignored", "already remember" in again and len(mem._load()) == 1, again)
    sec = mem.dispatch("remember", fact="My bank PIN is 4321")
    check("memory: secrets refused", sec.startswith("Not saved") and len(mem._load()) == 1, sec)
    mem.dispatch("remember", fact="The user's thesis is about solar dryers for mangoes")
    gone = mem.dispatch("forget", fact="thesis solar dryers")
    check("memory: forget by words", "Forgot" in gone and "thesis" not in mem.prompt_block(), gone)
    for i in range(mem.MAX_FACTS + 5):
        mem.dispatch("remember", fact=f"Test fact number {i} about something")
    facts = mem._load()
    check("memory: capped", len(facts) == mem.MAX_FACTS and facts[-1]["fact"].endswith(f"{mem.MAX_FACTS + 4} about something"))
    blk = mem.prompt_block()
    check("memory: prompt block capped", len(blk) < mem.PROMPT_CHARS + 200 and "newest" not in blk
          and f"number {mem.MAX_FACTS + 4}" in blk, str(len(blk)))
    yes = ["Remember that I like OPM music.", "Please remember my sister is called Ana",
           "Don't forget I'm vegetarian", "keep in mind that I work nights"]
    no = ["Do you remember my name?", "What do you remember about me", "remember to call mom at 5",
          "Remind me to drink water", "I can't remember the song title", "play some music"]
    check("memory: remember requests detected", all(mem.wants_remember(t) for t in yes),
          str([t for t in yes if not mem.wants_remember(t)]))
    check("memory: questions and to-dos are not", not any(mem.wants_remember(t) for t in no[:5]) and not mem.wants_remember(no[5]),
          str([t for t in no if mem.wants_remember(t)]))
    out = mem.dispatch("forget", fact="all")
    check("memory: forget all", "forgot all" in out and mem.prompt_block() == "", out)


def tool_turn(q, n_tools=5, big=3000, name="fetch_url"):
    msgs = [{"role": "user", "content": q}]
    for k in range(n_tools):
        cid = f"c{len(q)}{k}"
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": name, "arguments": json.dumps({"html": "<p>" + "x" * big + "</p>", "name": "site"})}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": "R" * big})
    msgs.append({"role": "assistant", "content": f"answer to {q}"})
    return msgs


def window_case():
    msgs = []
    for i in range(10):
        msgs += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    msgs += tool_turn("build me a website", n_tools=6)
    msgs.insert(5, {"role": "user", "content": "[Background job update, not said by the user] done"})
    start = cu.window_start(msgs, 8, "[Background job update")
    kept = msgs[start:]
    users = [m["content"] for m in kept if m["role"] == "user" and not m["content"].startswith("[Background")]
    check("window: last 8 user turns kept", users[0] == "q3" and users[-1] == "build me a website"
          and len(users) == 8, str(users))
    check("window: starts on a user message", kept[0]["role"] == "user")


def shorten_case():
    msgs = tool_turn("old one") + tool_turn("middle") + tool_turn("newest")
    n = cu.shorten_old(msgs)
    old = msgs[:len(tool_turn("old one"))]
    new = msgs[len(old):]
    check("shorten: old tool results cut", n > 0 and all(len(m["content"]) < 500 for m in old if m["role"] == "tool"))
    args = [tc["function"]["arguments"] for m in old if m.get("tool_calls") for tc in m["tool_calls"]]
    ok_json = all(isinstance(json.loads(a), dict) for a in args)
    check("shorten: old arguments short and still valid JSON",
          ok_json and all(len(a) < 400 for a in args) and json.loads(args[0])["name"] == "site", args[0][:80])
    check("shorten: newest two turns untouched",
          all(len(m["content"]) == 3000 for m in new if m["role"] == "tool"))


def agent_case():
    try:
        os.environ.setdefault("ME3_BENCH", "1")
        os.environ.setdefault("ME3_NO_STATUS", "1")
        import agent
    except Exception as e:
        print(f"SKIP agent hooks (agent.py not importable here: {type(e).__name__})")
        return
    msgs = [{"role": "system", "content": "S"}]
    for i in range(12):
        msgs += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    msgs += tool_turn("research this", n_tools=12)
    ctx = types.SimpleNamespace(messages=msgs)
    agent._trim_llm_context(ctx, agent.GroqResilientLLM.MAX_CONTEXT_MESSAGES,
                            agent.GroqResilientLLM.MAX_CONTEXT_TURNS)
    users = [m["content"] for m in ctx.messages if m["role"] == "user"]
    check("agent: 8 turns kept through a long tool turn",
          len(users) == 8 and users[-1] == "research this" and users[0] == "q5", str(users))
    check("agent: window starts on a user message", ctx.messages[1]["role"] == "user")
    # a single turn bigger than the hard cap keeps its own request
    big = [{"role": "system", "content": "S"}] + tool_turn("huge job", n_tools=40, big=10)
    ctx2 = types.SimpleNamespace(messages=big)
    agent._trim_llm_context(ctx2, 60, 8)
    check("agent: current request never dropped",
          any(m.get("role") == "user" and m.get("content") == "huge job" for m in ctx2.messages))
    mem.dispatch("remember", fact="The user likes OPM music")
    llm = types.SimpleNamespace(_settings=types.SimpleNamespace(system_instruction=""))
    agent._stamp_system_and_strip(types.SimpleNamespace(messages=[{"role": "user", "content": "hi"}]), llm)
    check("agent: saved notes in the system instruction",
          "likes OPM music" in llm._settings.system_instruction)
    g = agent._memory_guard_choice(types.SimpleNamespace(messages=[
        {"role": "user", "content": "Remember that I like OPM music"}]))
    check("agent: remember request forces the remember call", g and g["function"]["name"] == "remember", str(g))
    g2 = agent._memory_guard_choice(types.SimpleNamespace(messages=[
        {"role": "user", "content": "Remember that I like OPM music"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function",
         "function": {"name": "remember", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "Saved"}]))
    check("agent: not forced again after the call", g2 is None, str(g2))
    import asyncio
    asyncio.new_event_loop().run_until_complete(stream_case(agent))
    asyncio.new_event_loop().run_until_complete(reset_case(agent))
    names = {s.name for s in agent._function_schemas()}
    check("agent: memory tools registered", mem.NAMES <= names, str(mem.NAMES - names))
    mem.dispatch("forget", fact="all")


def think_case():
    cases = [("You like OPM music.\n</think>\n\nYou like OPM music.", "You like OPM music."),
             ("</think>\n\nHello there friend", "Hello there friend"),
             ("<think></think>Hi you all", "Hi you all"),
             ("a < b is true", "a < b is true")]
    bad = [(t, cu.clean_reply(t).strip()) for t, w in cases if cu.clean_reply(t).strip() != w]
    check("think: whole replies cleaned", not bad, str(bad))
    f = cu.ThinkFilter()
    out = "".join(f.feed(x) for x in ["You like ", "OPM.</th", "ink>\n\nYou like", " OPM."])
    check("think: tag split across chunks", out == "You like OPM." and f.off, repr(out))


async def stream_case(agent):
    """Real openai chunk objects through the voice stream tap."""
    try:
        from openai.types.chat import ChatCompletionChunk
    except Exception as e:
        print(f"SKIP think: voice stream (no openai types: {e!r})")
        return
    parts = ["You like ", "OPM music.", "\n</think>", "\n\nYou like OPM music."]

    async def gen():
        for i, p in enumerate(parts):
            yield ChatCompletionChunk.model_validate({
                "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]})
    got = []
    async for ch in agent._tap_reasoning_stream(gen()):
        got.append(ch.choices[0].delta.content or "")
    check("think: voice stream speaks the answer once", "".join(got).strip() == "You like OPM music.", repr(got))


async def reset_case(agent):
    """Restart button: a new reset.json empties the voice agent's history."""
    import asyncio
    agent.RESET_JSON_PATH = os.path.join(TMP, "reset.json")
    with open(agent.RESET_JSON_PATH, "w") as f:
        json.dump({"t": 1.0}, f)            # an old reset: must not clear
    ctx = types.SimpleNamespace(messages=[{"role": "user", "content": "hi"},
                                          {"role": "assistant", "content": "hello"}])
    agent._CONTEXT_REF["ctx"] = ctx
    task = asyncio.ensure_future(agent._reset_watcher(poll_s=0.05))
    await asyncio.sleep(0.2)
    kept = len(ctx.messages)
    with open(agent.RESET_JSON_PATH, "w") as f:
        json.dump({"t": 2.0}, f)
    await asyncio.sleep(0.2)
    task.cancel()
    check("reset: old reset ignored, new one clears the voice history",
          kept == 2 and ctx.messages == [], f"{kept} {ctx.messages}")


def dashboard_case():
    src = open(os.path.join(os.path.dirname(HERE), "bootsui", "boots_ui.py"), encoding="utf-8").read()
    check("dashboard: memory tools routed and loaded",
          '("remember", "forget", "list_memories")' in src and src.count('_strip_think(') >= 4 and "_mem.prompt_block()" in src
          and 'elif self.path == "/reset":' in src and 'STATE["chat_history"].clear()' in src
          and '[-16:]' in src and 'str(res)[:1500]' in src)


if __name__ == "__main__":
    memory_case()
    window_case()
    shorten_case()
    think_case()
    dashboard_case()
    agent_case()
    ok = all(RESULTS)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.stdout.flush()
    os._exit(0 if ok else 1)
