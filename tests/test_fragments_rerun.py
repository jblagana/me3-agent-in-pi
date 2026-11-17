"""Speech fragments join into one request; a cut-off reply re-runs (2026-11-17).

1. merge     "Take down" / "by" / "demon hunters." back to back -> one user
             message; pieces said across a reply stay separate; tool messages
             untouched.
2. note      a developer note at the end of the context (what Pipecat's
             empty-user-turn recovery adds) is folded into the system
             instruction for that call and removed from the context.
3. rerun     real Pipecat aggregator: a turn answers, then a turn with no
             words interrupts the reply. With Boots' config the LLM runs again
             (and the note says to carry on, not ask to repeat); with the old
             config (empty_user_turn=None) it does not.

Offline: no microphone, no network, no LLM calls.

    cd ~/me3 && .venv/bin/python tests/test_fragments_rerun.py
"""
import asyncio
import os
import sys
import types

os.environ.setdefault("ME3_BENCH", "1")
os.environ.setdefault("ME3_NO_STATUS", "1")
os.environ.setdefault("ME3_BOOTSUI", "http://127.0.0.1:9")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent  # noqa: E402

from pipecat.frames.frames import (EndFrame, InterimTranscriptionFrame,  # noqa: E402
                                   LLMContextFrame, TranscriptionFrame,
                                   VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineTask  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair, LLMUserAggregatorParams)
from pipecat.processors.frame_processor import FrameProcessor  # noqa: E402
from pipecat.turns.empty_user_turn import EmptyUserTurnConfig  # noqa: E402
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (  # noqa: E402
    MinWordsUserTurnStartStrategy)
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (  # noqa: E402
    SpeechTimeoutUserTurnStopStrategy)
from pipecat.turns.user_turn_strategies import UserTurnStrategies  # noqa: E402


def _ctx(msgs):
    c = LLMContext()
    c.messages.extend(msgs)
    return c


def merge_case():
    tool_call = {"role": "assistant", "tool_calls": [{"id": "x", "type": "function",
                 "function": {"name": "spotify_play_search", "arguments": "{}"}}]}
    tool_res = {"role": "tool", "tool_call_id": "x", "content": "Ask: TWICE or HUNTR/X?"}
    c = _ctx([{"role": "user", "content": "Let's play the takedown music by Diva"},
              tool_call, tool_res,
              {"role": "user", "content": "Take down"},
              {"role": "user", "content": "by"},
              {"role": "user", "content": "demon hunters."}])
    agent._merge_user_fragments(c)
    m = c.messages
    joined = m[-1] == {"role": "user", "content": "Take down by demon hunters."}
    tools_kept = m[1] is tool_call and m[2] is tool_res and len(m) == 4

    c2 = _ctx([{"role": "user", "content": "What can you do?"},
               {"role": "assistant", "content": "Lots of things [I was cut off here]"},
               {"role": "user", "content": "Play music"}])
    agent._merge_user_fragments(c2)
    across_reply = len(c2.messages) == 3

    ok = joined and tools_kept and across_reply
    print(f"{'PASS' if ok else 'FAIL'}  merge   fragments joined={joined}  tool msgs kept={tools_kept}  "
          f"separate across a reply={across_reply}")
    return ok


def note_case():
    llm = types.SimpleNamespace(_settings=types.SimpleNamespace(system_instruction=""))
    c = _ctx([{"role": "user", "content": "Take down by demon hunters."},
              {"role": "developer", "content": agent.EMPTY_TURN_PROMPT}])
    agent._stamp_system_and_strip(c, llm)
    folded = llm._settings.system_instruction.endswith(agent.EMPTY_TURN_PROMPT)
    removed = [m.get("role") for m in c.messages] == ["user"]
    # an ordinary call carries no note
    c2 = _ctx([{"role": "user", "content": "hi"}])
    agent._stamp_system_and_strip(c2, llm)
    clean = agent.EMPTY_TURN_PROMPT not in llm._settings.system_instruction
    ok = folded and removed and clean
    print(f"{'PASS' if ok else 'FAIL'}  note    folded into system={folded}  removed from context={removed}  "
          f"next call clean={clean}")
    return ok


class _Sink(FrameProcessor):
    """After the aggregator: counts the LLM runs it asks for."""

    def __init__(self, runs):
        super().__init__(name="LLMStandIn")
        self.runs = runs

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            self.runs.append([dict(m) for m in frame.context.messages])
        await self.push_frame(frame, direction)


async def _rerun(empty_cfg):
    runs = []
    pair = LLMContextAggregatorPair(
        LLMContext(), user_params=LLMUserAggregatorParams(
            empty_user_turn=empty_cfg, user_turn_stop_timeout=0.6,
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=1)],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)])))
    user_agg = pair[0] if isinstance(pair, tuple) else pair.user()
    task = PipelineTask(Pipeline([user_agg, _Sink(runs)]))
    run = asyncio.ensure_future(PipelineRunner(handle_sigint=False).run(task))
    await asyncio.sleep(0.5)

    async def q(frame, wait=0.3):
        await task.queue_frame(frame)
        await asyncio.sleep(wait)

    # the request: a real turn, the LLM is asked to answer it
    await q(VADUserStartedSpeakingFrame())
    await q(InterimTranscriptionFrame("demon", "user", "t"))
    await q(TranscriptionFrame("demon hunters.", "user", "t"))
    await q(VADUserStoppedSpeakingFrame(), 0.5)
    first = len(runs)
    # while the LLM works: a turn starts on words that never become new text
    # (the live failure), interrupting the reply, then ends empty
    await q(InterimTranscriptionFrame("demon hunters", "user", "t"), 1.5)
    await task.queue_frame(EndFrame())
    await asyncio.wait({run}, timeout=3)
    return first, runs


async def rerun_case():
    first, runs = await _rerun(EmptyUserTurnConfig(interrupted_prompt=agent.EMPTY_TURN_PROMPT,
                                                   idle_prompt=None, max_consecutive_recoveries=1))
    reran = first == 1 and len(runs) == 2
    note = bool(runs) and runs[-1][-1].get("role") == "developer" \
        and runs[-1][-1].get("content") == agent.EMPTY_TURN_PROMPT
    old_first, old_runs = await _rerun(None)
    old_silent = old_first == 1 and len(old_runs) == 1
    ok = reran and note and old_silent
    print(f"{'PASS' if ok else 'FAIL'}  rerun   new config re-runs once={reran}  with carry-on note={note}  "
          f"old config stays silent={old_silent}  (runs: {len(runs)} vs {len(old_runs)})")
    return ok


async def main():
    a = merge_case()
    b = note_case()
    c = await rerun_case()
    ok = a and b and c
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    code = loop.run_until_complete(main())
    sys.stdout.flush()
    os._exit(code)
