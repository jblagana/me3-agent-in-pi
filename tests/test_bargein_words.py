"""Barge-in on words, not sound (2026-11-16).

While Boots is speaking, a voice-activity blip with no words (room noise) must
NOT interrupt it; a one-word partial transcript must not either; a two-word
one must. Also: a VAD blip must not wipe the dashboard's reply text
(StatusIndicator now resets on the real UserStartedSpeakingFrame).

Offline: no microphone, no network.

    cd ~/me3 && .venv/bin/python tests/test_bargein_words.py
"""
import asyncio
import os
import sys

os.environ.setdefault("ME3_BENCH", "1")
os.environ.setdefault("ME3_NO_STATUS", "1")
os.environ.setdefault("ME3_BOOTSUI", "http://127.0.0.1:9")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent  # noqa: E402

from pipecat.frames.frames import (BotStartedSpeakingFrame, EndFrame,  # noqa: E402
                                   InterimTranscriptionFrame, InterruptionFrame,
                                   UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineTask  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair, LLMUserAggregatorParams)
from pipecat.processors.frame_processor import FrameProcessor  # noqa: E402
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (  # noqa: E402
    MinWordsUserTurnStartStrategy)
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (  # noqa: E402
    SpeechTimeoutUserTurnStopStrategy)
from pipecat.turns.user_turn_strategies import UserTurnStrategies  # noqa: E402

EVENTS = []


class _Upstream(FrameProcessor):
    """Sits before the aggregator: records interruptions/turn starts sent up."""

    def __init__(self):
        super().__init__(name="UpstreamRecorder")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            EVENTS.append(type(frame).__name__)
        await self.push_frame(frame, direction)


def _step(label):
    EVENTS.append(f"-- {label}")


async def _aggregator_case():
    pair = LLMContextAggregatorPair(
        LLMContext(), user_params=LLMUserAggregatorParams(
            empty_user_turn=None,
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=2)],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)])))
    user_agg = pair[0] if isinstance(pair, tuple) else pair.user()
    task = PipelineTask(Pipeline([_Upstream(), user_agg]))
    run = asyncio.ensure_future(PipelineRunner(handle_sigint=False).run(task))
    await asyncio.sleep(0.5)

    async def send(label, frame):
        _step(label)
        await task.queue_frame(frame)
        await asyncio.sleep(0.4)

    await send("bot starts speaking", BotStartedSpeakingFrame())
    await send("noise: VAD start, no words", VADUserStartedSpeakingFrame())
    await send("partial transcript 'stop' (1 word)",
               InterimTranscriptionFrame("stop", "user", "t"))
    await send("partial transcript 'stop talking' (2 words)",
               InterimTranscriptionFrame("stop talking", "user", "t"))
    await task.queue_frame(EndFrame())
    await asyncio.wait({run}, timeout=3)

    def fired_after(label):
        i = EVENTS.index(f"-- {label}")
        nxt = next((j for j in range(i + 1, len(EVENTS)) if EVENTS[j].startswith("-- ")),
                   len(EVENTS))
        return "InterruptionFrame" in EVENTS[i + 1:nxt]

    noise = fired_after("noise: VAD start, no words")
    one = fired_after("partial transcript 'stop' (1 word)")
    two = fired_after("partial transcript 'stop talking' (2 words)")
    ok = (not noise) and (not one) and two
    print(f"{'PASS' if ok else 'FAIL'}  aggregator   noise interrupts={noise}  "
          f"1 word interrupts={one}  2 words interrupt={two}")
    return ok


async def _status_case():
    st = agent.StatusIndicator(state={"bot_speaking": True, "mic_level": 0.0})
    task = PipelineTask(Pipeline([st]))
    run = asyncio.ensure_future(PipelineRunner(handle_sigint=False).run(task))
    await asyncio.sleep(0.5)
    agent._RECENT_TEXT["reply"] = "The capital of France is Paris."
    await task.queue_frame(VADUserStartedSpeakingFrame())
    await asyncio.sleep(0.3)
    kept = agent._RECENT_TEXT.get("reply") == "The capital of France is Paris."
    await task.queue_frame(UserStartedSpeakingFrame())
    await asyncio.sleep(0.3)
    cleared = agent._RECENT_TEXT.get("reply") is None
    await task.queue_frame(EndFrame())
    await asyncio.wait({run}, timeout=3)
    ok = kept and cleared
    print(f"{'PASS' if ok else 'FAIL'}  status UI    VAD blip keeps reply={kept}  "
          f"real turn start clears it={cleared}")
    return ok


async def main():
    a = await _aggregator_case()
    b = await _status_case()
    print("ALL PASS" if (a and b) else "SOME FAILED")
    return 0 if (a and b) else 1


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    code = loop.run_until_complete(main())
    sys.stdout.flush()
    os._exit(code)
