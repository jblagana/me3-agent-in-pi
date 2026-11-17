"""Stale interim after the turn ends must not start a new turn (2026-11-17).

Seen live on 2026-11-16 20:05:48: after the final "demon hunters." the turn
ended and the LLM started, then 0.19 s later an interim transcript of the same
words reached the user aggregator. The min-words start strategy (min_words=1
while Boots is silent) took it as a new user turn, broadcast an interruption,
cancelled the LLM, and the song request was lost.

DeepgramBootsSTT now drops interims between VAD stop and the next VAD start.

Cases:
  control  the stale interim sent straight to the aggregator DOES start a turn
           (the failure mode exists in this Pipecat version)
  fixed    the same interim sent through DeepgramBootsSTT does NOT
  fresh    after a new VAD start, an interim through DeepgramBootsSTT still
           starts a turn (barge-in / normal speech unaffected)

Offline: no microphone, no network (Deepgram is never connected).

    cd ~/me3 && .venv/bin/python tests/test_stale_interim.py
"""
import asyncio
import os
import sys

os.environ.setdefault("ME3_BENCH", "1")
os.environ.setdefault("ME3_NO_STATUS", "1")
os.environ.setdefault("ME3_BOOTSUI", "http://127.0.0.1:9")
os.environ.setdefault("DEEPGRAM_API_KEY", "offline-test")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipecat.frames.frames import (EndFrame, InterimTranscriptionFrame,  # noqa: E402
                                   InterruptionFrame, TranscriptionFrame,
                                   UserStartedSpeakingFrame, VADUserStartedSpeakingFrame,
                                   VADUserStoppedSpeakingFrame)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineTask  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair, LLMUserAggregatorParams)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402
from pipecat.services.deepgram.stt import DeepgramSTTService  # noqa: E402
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (  # noqa: E402
    MinWordsUserTurnStartStrategy)
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (  # noqa: E402
    SpeechTimeoutUserTurnStopStrategy)
from pipecat.turns.user_turn_strategies import UserTurnStrategies  # noqa: E402

import deepgram_voice  # noqa: E402

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


async def _noop_base(self, frame, direction):
    """Stand-in for DeepgramSTTService.process_frame: never touch the network."""
    return


async def main():
    # Keep the Boots subclass logic (VAD flags, push_frame filter) but stub the
    # Deepgram base so nothing connects.
    DeepgramSTTService.process_frame = _noop_base
    stt = deepgram_voice.make_stt()

    pair = LLMContextAggregatorPair(
        LLMContext(), user_params=LLMUserAggregatorParams(
            empty_user_turn=None,
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=1)],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)])))
    user_agg = pair[0] if isinstance(pair, tuple) else pair.user()
    task = PipelineTask(Pipeline([_Upstream(), user_agg]))
    run = asyncio.ensure_future(PipelineRunner(handle_sigint=False).run(task))
    await asyncio.sleep(0.5)
    stt._next = user_agg            # what the bypass walk finds in the real pipeline

    async def vad(start):
        f = VADUserStartedSpeakingFrame() if start else VADUserStoppedSpeakingFrame()
        await stt.process_frame(f, FrameDirection.DOWNSTREAM)   # Boots flags
        await task.queue_frame(type(f)())                      # aggregator sees VAD too
        await asyncio.sleep(0.3)

    async def via_stt(frame):
        await stt.push_frame(frame, FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.4)

    async def one_turn(words):
        await vad(True)
        await via_stt(InterimTranscriptionFrame(words.split()[0], "user", "t"))
        await via_stt(TranscriptionFrame(words, "user", "t"))
        await vad(False)
        await asyncio.sleep(0.4)        # turn stops, LLM would start here

    def started(since):
        return any(e in ("InterruptionFrame", "UserStartedSpeakingFrame")
                   for e in EVENTS[since:])

    # control: stale interim straight into the aggregator
    await one_turn("demon hunters.")
    n = len(EVENTS)
    await task.queue_frame(InterimTranscriptionFrame("demon hunters", "user", "t"))
    await asyncio.sleep(0.4)
    control = started(n)

    # fixed: the same stale interim through DeepgramBootsSTT
    await one_turn("demon hunters.")
    n = len(EVENTS)
    await via_stt(InterimTranscriptionFrame("demon hunters", "user", "t"))
    fixed = not started(n)

    # fresh speech after a new VAD start still starts a turn
    n = len(EVENTS)
    await vad(True)
    await via_stt(InterimTranscriptionFrame("play it", "user", "t"))
    fresh = started(n)

    await task.queue_frame(EndFrame())
    await asyncio.wait({run}, timeout=3)

    ok = control and fixed and fresh
    print(f"{'PASS' if control else 'FAIL'}  control  stale interim straight to aggregator starts a turn={control}")
    print(f"{'PASS' if fixed else 'FAIL'}  fixed    stale interim via DeepgramBootsSTT ignored={fixed}")
    print(f"{'PASS' if fresh else 'FAIL'}  fresh    interim after new VAD start starts a turn={fresh}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    code = loop.run_until_complete(main())
    sys.stdout.flush()
    os._exit(code)
