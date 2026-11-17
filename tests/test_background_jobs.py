"""Website/app builds run in the background (2026-11-18).

1. register  write_website / build_app / publish_website register as Pipecat
             async calls (not cancelled by talking); web_search stays normal.
2. handler   the job runs through the MCP bridge with the long timeout, a
             second call while it runs returns "already running" at once, and
             the running list is empty afterwards.
3. start     the tool start releases the turn (final_seen) and costs no LLM
             call: the spoken line is fixed text.
4. flow      real Pipecat assistant aggregator: job starts, the user asks
             something else, the job finishes -> Pipecat writes the result as a
             developer message and asks for ONE LLM run. Boots turns it into a
             labelled message Qwen accepts, keeps it, adds the job rules to the
             system prompt, and never merges it with the user's words.
5. quiet     a job that finishes before anything else is said settles in
             place as a normal tool result (no labelled message, one LLM run,
             same as a build today).

Offline: no microphone, no network, no LLM calls.

    cd ~/me3 && .venv/bin/python tests/test_background_jobs.py
"""
import asyncio
import json
import os
import sys
import types

os.environ.setdefault("ME3_BENCH", "1")
os.environ.setdefault("ME3_NO_STATUS", "1")
os.environ.setdefault("ME3_BOOTSUI", "http://127.0.0.1:9")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent  # noqa: E402

from pipecat.frames.frames import (EndFrame, FunctionCallFromLLM,  # noqa: E402
                                   FunctionCallInProgressFrame, FunctionCallResultFrame,
                                   FunctionCallsStartedFrame, LLMContextFrame)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineTask  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair)
from pipecat.processors.frame_processor import FrameProcessor  # noqa: E402
from pipecat.services.openai.llm import OpenAILLMService  # noqa: E402

RESULTS = []


def report(name, ok, detail):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:<9}{detail}")


def register_case():
    llm = OpenAILLMService(api_key="offline-test", model="offline")
    schemas = {s.name: s for s in agent._function_schemas()}
    for n in ("write_website", "build_app", "publish_website", "web_search"):
        llm.register_function(n, schemas[n].handler)
    bg = all(llm._function_is_async(n) for n in ("write_website", "build_app", "publish_website"))
    normal = not llm._function_is_async("web_search")
    report("register", bg and normal, f"builds are background jobs={bg}  web_search unchanged={normal}")


async def handler_case():
    seen = {}

    async def fake_call(tool, args, timeout=60):
        seen["timeout"] = timeout
        await asyncio.sleep(0.3)
        return "Built and published at https://example.invalid/site"

    agent._MCP.call = fake_call
    got = []

    async def cb(result, **_):
        got.append(result)

    def params():
        return types.SimpleNamespace(function_name="write_website",
                                     arguments={"description": "milk tea"}, result_callback=cb)

    job = asyncio.ensure_future(agent._web_bg_body(params()))
    await asyncio.sleep(0.1)
    running = "write_website" in agent._BG_JOBS
    await agent._web_bg_body(params())          # second call while running
    dup = len(got) == 1 and "already running" in got[0]
    await job
    done = len(got) == 2 and got[1].startswith("Built") and not agent._BG_JOBS
    long_to = seen.get("timeout", 0) > 60
    report("handler", running and dup and done and long_to,
           f"tracked while running={running}  2nd call refused={dup}  "
           f"result delivered + cleared={done}  timeout>60s={long_to}")


async def updates_case():
    """Unprompted stage updates: spoken only in a quiet moment, never while
    the user talks; silent stages only feed the status; one long-job nudge."""
    import tempfile
    web = agent.web
    web.JOB_DIR = tempfile.mkdtemp(prefix="me3-jobs-test-")
    agent.BG_UPDATE_GAP_S, agent.BG_NUDGE_S = 0.5, 4.0
    agent._SPEAK_GATE.update(user=False, bot=False)
    agent._TURN_INFLIGHT["until"] = 0.0
    spoken = []
    agent._speak_progress = lambda text, force=False: spoken.append(text)
    obs = {}

    async def fake_tool(tool, args, timeout=60):
        web._stage(tool, "images", "finding pictures and laying out the page")
        await asyncio.sleep(1.5)
        obs["silent_stage"] = list(spoken)
        obs["status1"] = agent._bg_running_text()
        agent._SPEAK_GATE["user"] = True                  # user starts talking
        web._stage(tool, "publishing", "publishing it to GitHub Pages",
                   say="Quick update: the website's built, and I'm publishing it now.")
        await asyncio.sleep(2.0)
        obs["while_talking"] = list(spoken)
        obs["status2"] = agent._bg_running_text()
        agent._SPEAK_GATE["user"] = False                 # quiet again
        await asyncio.sleep(1.5)
        obs["after"] = list(spoken)
        await asyncio.sleep(2.5)                          # past the nudge time
        obs["nudge"] = list(spoken)
        return "Built and published at https://example.invalid/site"

    agent._MCP.call = fake_tool
    got = []

    async def cb(result, **_):
        got.append(result)

    await agent._web_bg_body(types.SimpleNamespace(
        function_name="write_website", arguments={}, result_callback=cb))
    silent = obs["silent_stage"] == [] and "finding pictures" in obs["status1"]
    held = obs["while_talking"] == [] and "publishing" in obs["status2"]
    said = obs["after"] == ["Quick update: the website's built, and I'm publishing it now."]
    nudge = len(obs["nudge"]) == 2 and obs["nudge"][1].startswith("Still working on the website")
    clean = not agent._BG_STATUS and not agent._BG_JOBS and bool(got)
    report("updates", silent and held and said and nudge and clean,
           f"silent stage not spoken={silent}  held while you talk={held}  "
           f"said when quiet={said}  one long-job nudge={nudge}  cleaned up={clean}")


async def start_case():
    st = agent.StatusIndicator(state={"bot_speaking": False, "mic_level": 0.0})
    task = PipelineTask(Pipeline([st]))
    run = asyncio.ensure_future(PipelineRunner(handle_sigint=False).run(task))
    await asyncio.sleep(0.5)
    spoken = []
    agent._speak_progress = lambda text, force=False: spoken.append(text)
    agent._BG_SPOKEN.update(last=0.0, n=0)

    async def start(tool, model_text=""):
        agent._VOICE_T["final_seen"] = False
        agent._RECENT_TEXT["reply"] = model_text
        n = len(spoken)
        await task.queue_frame(FunctionCallsStartedFrame(function_calls=[FunctionCallFromLLM(
            function_name=tool, tool_call_id="w0", arguments={}, context=None)]))
        await asyncio.sleep(0.3)
        return spoken[n:], agent._VOICE_T.get("final_seen") is True

    first, rel1 = await start("write_website")
    again, rel2 = await start("build_app")
    own, rel3 = await start("write_website", "Sure, I'll build that milk tea site for you.")
    await task.queue_frame(EndFrame())
    await asyncio.wait({run}, timeout=3)
    released = rel1 and rel2 and rel3
    full = len(first) == 1 and "website" in first[0] and len(first[0]) > 30
    short = len(again) == 1 and again[0] in agent._BG_START_SHORT and again[0] != first[0]
    skipped = own == []
    report("start", released and full and short and skipped,
           f"turn released={released}  first line full={full} {first[:1]}  "
           f"repeat is short + different={short} {again[:1]}  "
           f"model's own words -> no canned line={skipped}")


class _UpRec(FrameProcessor):
    """Before the assistant aggregator: counts the LLM runs it asks for."""

    def __init__(self, runs):
        super().__init__(name="LLMStandIn")
        self.runs = runs

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            self.runs.append(1)
        await self.push_frame(frame, direction)


async def _flow(user_speaks):
    runs = []
    ctx = LLMContext()
    ctx.add_message({"role": "user", "content": "Build me a website about milk tea"})
    pair = LLMContextAggregatorPair(ctx)
    asst = pair[1] if isinstance(pair, tuple) else pair.assistant()
    task = PipelineTask(Pipeline([_UpRec(runs), asst]))
    run = asyncio.ensure_future(PipelineRunner(handle_sigint=False).run(task))
    await asyncio.sleep(0.5)
    await task.queue_frame(FunctionCallInProgressFrame(
        function_name="write_website", tool_call_id="w1", arguments={"description": "milk tea"},
        cancel_on_interruption=False))
    await asyncio.sleep(0.3)
    if user_speaks:
        ctx.add_message({"role": "user", "content": "What's the weather today?"})
        ctx.add_message({"role": "assistant", "content": "It's 31 degrees and sunny."})
        ctx.add_message({"role": "user", "content": "nice"})
    await task.queue_frame(FunctionCallResultFrame(
        function_name="write_website", tool_call_id="w1", arguments={"description": "milk tea"},
        result="Built and published at https://example.invalid/site", run_llm=None))
    await asyncio.sleep(0.5)
    await task.queue_frame(EndFrame())
    await asyncio.wait({run}, timeout=3)
    return ctx, runs


async def flow_case():
    ctx, runs = await _flow(user_speaks=True)
    had_dev = any(m.get("role") == "developer" for m in ctx.messages if isinstance(m, dict))
    llm = types.SimpleNamespace(_settings=types.SimpleNamespace(system_instruction=""))
    agent._stamp_system_and_strip(ctx, llm)
    agent._merge_user_fragments(ctx)
    notes = [m for m in ctx.messages if agent._is_bg_note(m)]
    kept = len(notes) == 1 and "write_website finished" in notes[0]["content"] \
        and "example.invalid" in notes[0]["content"]
    no_dev = not any(m.get("role") == "developer" for m in ctx.messages if isinstance(m, dict))
    rules = "BACKGROUND JOBS" in llm._settings.system_instruction
    not_merged = any(m.get("content") == "nice" for m in ctx.messages if isinstance(m, dict))
    one_run = len(runs) == 1
    # a later ordinary call keeps the note and the rules (it is still in context)
    report("flow", had_dev and kept and no_dev and rules and not_merged and one_run,
           f"pipecat wrote developer msg={had_dev}  labelled note kept={kept}  "
           f"no developer left={no_dev}  rules in system={rules}  "
           f"user words separate={not_merged}  LLM runs={len(runs)}")


async def quiet_case():
    ctx, runs = await _flow(user_speaks=False)
    llm = types.SimpleNamespace(_settings=types.SimpleNamespace(system_instruction=""))
    agent._stamp_system_and_strip(ctx, llm)
    tool_msg = [m for m in ctx.messages if isinstance(m, dict) and m.get("role") == "tool"]
    in_place = bool(tool_msg) and "example.invalid" in json.dumps(tool_msg[-1].get("content"))
    no_note = not any(agent._is_bg_note(m) for m in ctx.messages)
    no_rules = "BACKGROUND JOBS" not in llm._settings.system_instruction
    report("quiet", in_place and no_note and no_rules and len(runs) == 1,
           f"settled as tool result={in_place}  no note={no_note}  "
           f"no extra prompt={no_rules}  LLM runs={len(runs)}")


async def main():
    register_case()
    await handler_case()
    await flow_case()
    await quiet_case()
    await updates_case()           # stubs _speak_progress
    await start_case()             # stubs _speak_progress
    ok = all(RESULTS)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    code = loop.run_until_complete(main())
    sys.stdout.flush()
    os._exit(code)
