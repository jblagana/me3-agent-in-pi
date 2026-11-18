"""Tool routing: send only the tools a turn needs (2026-11-18).

1. groups   every tool belongs to a group; the always-sent core is small.
2. pick     requests map to the right groups ("play X" -> music, "lights on"
            -> home, chit-chat -> none); follow-ups keep the previous
            message's groups; files-only mode adds the library.
3. load     load_tools("music") parses group names and aliases.
4. agent    _route_tools swaps a subset into the context for the request and
            hands back the full set to restore; a forced tool is always in
            it; load_tools adds groups for the rest of the turn and a new
            user message resets them; ME3_TOOL_ROUTER-style off = no swap.
5. dashboard  the typed chat routes body["tools"] and dispatches load_tools.

Offline: no network, no LLM calls.

    cd ~/me3 && .venv/bin/python tests/test_router.py
"""
import asyncio
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
TMP = tempfile.mkdtemp(prefix="boots-router-test-")
os.environ["BOOTS_LIBRARY"] = os.path.join(TMP, "library")
os.environ["BOOTS_MEMORY"] = os.path.join(TMP, "memory.json")

import tool_router as tr        # noqa: E402
import me2_tools, spotify_tools, web_tools, files_tools, shell_tools  # noqa: E402,E401
import library_tools, memory_tools  # noqa: E402,E401

RESULTS = []
MODS = [me2_tools, spotify_tools, web_tools, files_tools, shell_tools, library_tools, memory_tools]


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def groups_case():
    g = tr.build_groups(MODS)
    every = {n for m in MODS for n, _d, _p in m.TOOL_SPECS}
    covered = set().union(*g.values())
    check("groups: every tool is in a group", every <= covered, str(every - covered))
    check("groups: core is small and has notes + web search",
          {"remember", "web_search", "check_weather", "load_tools"} <= g["core"] and len(g["core"]) <= 8,
          str(sorted(g["core"])))
    check("groups: heavy builders are not core", "write_website" not in g["core"] and "write_website" in g["build"])
    return g


def pick_case(g):
    cases = {
        "play Leaves by Ben and Ben": {"music"},
        "turn on the lights": {"home"},
        "set a timer for 10 minutes": {"home"},
        "how are you today": set(),
        "who won the NBA finals": set(),
        "make me a calculator app": {"build"},
        "read my notes file": {"files"},
        "how much disk space is left": {"shell"},
        "what does my thesis say about drying time": {"library"},
    }
    bad = {t: sorted(tr.pick_groups([t])) for t, want in cases.items()
           if not want <= tr.pick_groups([t]) or (not want and tr.pick_groups([t]))}
    check("pick: requests map to the right groups", not bad, str(bad))
    check("pick: follow-up keeps the previous request's group",
          "home" in tr.pick_groups(["a bit more", "make the lights brighter"]))
    check("pick: a tool used recently keeps its group",
          "music" in tr.pick_groups(["thanks"], used_tools={"spotify_play_search"}, groups_map=g))
    check("pick: files-only mode adds the library", "library" in tr.pick_groups(["hi"], files_mode=True))
    n_all = len(set().union(*g.values()))
    n_light = len(tr.tool_names(tr.pick_groups(["turn on the lights"]), g))
    check("pick: a lights request sends well under half the tools", n_light < n_all / 2, f"{n_light}/{n_all}")


def load_case():
    check("load: names and aliases", tr.parse_groups("Music, lights") == {"music", "home"})
    check("load: all", tr.parse_groups("all") == set(tr.KEYWORDS))
    check("load: unknown group explained", "No such tool group" in tr.load_tools_result("weather stuff"))


async def agent_case():
    try:
        os.environ.setdefault("ME3_BENCH", "1")
        os.environ.setdefault("ME3_NO_STATUS", "1")
        import agent
        from pipecat.processors.aggregators.llm_context import LLMContext
    except Exception as e:
        print(f"SKIP agent hooks (agent.py not importable here: {type(e).__name__})")
        return
    schemas = agent._function_schemas()
    names = {s.name for s in schemas}
    check("agent: load_tools registered", "load_tools" in names)
    ctx = LLMContext(messages=[{"role": "user", "content": "turn on the lights"}], tools=schemas)
    full = len(ctx.tools.standard_tools)
    old = agent._route_tools(ctx)
    sent = {s.name for s in ctx.tools.standard_tools}
    check("agent: routed subset swapped in", old is not None and "boots_lights" in sent
          and "write_website" not in sent and len(sent) < full / 2, f"{len(sent)}/{full}")
    ctx.set_tools(old)
    check("agent: full set restored", len(ctx.tools.standard_tools) == full)
    old = agent._route_tools(ctx, {"type": "function", "function": {"name": "write_website"}})
    check("agent: forced tool always sent", "write_website" in {s.name for s in ctx.tools.standard_tools})
    ctx.set_tools(old)

    async def cb(r):
        pass
    await agent._load_tools_handler(types.SimpleNamespace(arguments={"groups": "music"}, result_callback=cb))
    old = agent._route_tools(ctx)
    check("agent: load_tools adds the group this turn",
          "spotify_play_search" in {s.name for s in ctx.tools.standard_tools})
    ctx.set_tools(old)
    ctx2 = LLMContext(messages=[{"role": "user", "content": "how are you"}], tools=schemas)
    old = agent._route_tools(ctx2)
    sent = {s.name for s in ctx2.tools.standard_tools}
    check("agent: a new message resets loaded groups; chit-chat sends only core",
          "spotify_play_search" not in sent and sent <= tr.build_groups(MODS)["core"] | {"load_tools"},
          str(sorted(sent)))
    ctx2.set_tools(old)


def loop_case():
    import context_util as cu
    call = lambda n: {"role": "assistant", "content": None,
                      "tool_calls": [{"id": "x", "type": "function", "function": {"name": n, "arguments": "{}"}}]}
    res = {"role": "tool", "tool_call_id": "x", "content": "ok"}
    turn = [{"role": "user", "content": "It is too dark in here."}]
    check("loop: 3 runs of an action tool trip the guard",
          cu.looping_tool(turn + [call("boots_set_bright"), res] * 3) == "boots_set_bright")
    check("loop: 2 runs do not", cu.looping_tool(turn + [call("boots_set_bright"), res] * 2) is None)
    check("loop: research tools are exempt", cu.looping_tool(turn + [call("web_search"), res] * 5) is None)
    old = [{"role": "user", "content": "lights"}] + [call("boots_lights"), res] * 3
    check("loop: calls from an earlier turn don't count",
          cu.looping_tool(old + [{"role": "assistant", "content": "Done."}] + turn + [call("boots_lights"), res]) is None)


def dashboard_case():
    root = os.path.dirname(HERE)
    src = open(os.path.join(root, "bootsui", "boots_ui.py"), encoding="utf-8").read()
    check("dashboard: routes tools and handles load_tools",
          "_tr.pick_groups(" in src and 'elif name == "load_tools":' in src
          and 'body["tools"] = [t for t in _CHAT_TOOLS' in src)
    check("dashboard: loop guard and empty-reply retry",
          'body["tool_choice"] = "none"' in src and "_cu4.EMPTY_NUDGE" in src and "_empty_retried = True" in src)
    asrc = open(os.path.join(root, "agent.py"), encoding="utf-8").read()
    check("agent: loop guard turns tools off and restores", 'context.set_tool_choice("none")' in asrc
          and "context.set_tool_choice(_loop_tc[0])" in asrc)


if __name__ == "__main__":
    g = groups_case()
    pick_case(g)
    load_case()
    loop_case()
    dashboard_case()
    asyncio.new_event_loop().run_until_complete(agent_case())
    ok = all(RESULTS)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.stdout.flush()
    os._exit(0 if ok else 1)
