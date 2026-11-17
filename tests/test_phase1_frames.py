"""Phase 1 check: do Boots' custom processors pass DATA frames?

Pipecat creates a processor's data-frame task only when the base
FrameProcessor.process_frame sees the StartFrame. A subclass that skips
super().process_frame() still passes system frames (audio, VAD) but parks
every data frame (transcripts, TTSSpeakFrame, EndFrame) in a queue nobody reads.

Runs offline: no microphone, no network, no API calls.

    cd ~/me3 && ME3_BENCH=1 ME3_NO_STATUS=1 .venv/bin/python tests/test_phase1_frames.py

Exit code 0 = every processor passed every frame.
"""
import asyncio
import os
import sys

os.environ.setdefault("ME3_BENCH", "1")       # agent.py: skip the EMEET wait
os.environ.setdefault("ME3_NO_STATUS", "1")
os.environ.setdefault("ME3_BOOTSUI", "http://127.0.0.1:9")   # UI pushes go nowhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent  # noqa: E402

from pipecat.frames.frames import (EndFrame, InputAudioRawFrame,  # noqa: E402
                                   TranscriptionFrame, TTSSpeakFrame)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineTask  # noqa: E402
from pipecat.processors.frame_processor import FrameProcessor  # noqa: E402

TIMEOUT_S = 5.0


class _Sink(FrameProcessor):
    def __init__(self):
        super().__init__(name="Sink")
        self.seen = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.seen.append(type(frame).__name__)
        await self.push_frame(frame, direction)


def _frames():
    return [
        InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1),
        TranscriptionFrame("turn on the lights", "user", "2026-11-16T00:00:00Z"),
        TTSSpeakFrame("Still working on it."),
    ]


async def _check(name, make):
    proc, sink = make(), _Sink()
    task = PipelineTask(Pipeline([proc, sink]))
    await task.queue_frames(_frames() + [EndFrame()])
    runner = PipelineRunner(handle_sigint=False)
    run = asyncio.ensure_future(runner.run(task))
    done, _ = await asyncio.wait({run}, timeout=TIMEOUT_S)
    ended = run in done
    if not ended:
        # A processor that skips super().process_frame() also ignores
        # CancelFrame, so a polite cancel can hang too: give it 2 s, then
        # tear the runner down directly.
        # asyncio.wait, not wait_for: wait_for blocks until a cancelled call
        # actually finishes, and here it never does.
        await asyncio.wait({asyncio.ensure_future(task.cancel())}, timeout=2.0)
        run.cancel()
        await asyncio.wait({run}, timeout=2.0)
    want = ["InputAudioRawFrame", "TranscriptionFrame", "TTSSpeakFrame"]
    got = {w: (w in sink.seen) for w in want}
    ok = all(got.values()) and ended
    marks = "  ".join(f"{w.replace('Frame', '')}={'yes' if v else 'NO'}" for w, v in got.items())
    print(f"{'PASS' if ok else 'FAIL'}  {name:<16} {marks}  "
          f"pipeline_ended={'yes' if ended else 'NO (EndFrame stuck)'}")
    return ok


async def main():
    state = {"bot_speaking": False, "mic_level": 0.0}
    cases = [
        ("InputGate", lambda: agent.InputGate(state=dict(state))),
        ("StatusIndicator", lambda: agent.StatusIndicator(state=dict(state))),
        ("ErrorApologyTTS", lambda: agent.ErrorApologyTTS(None, None, dict(state))),
    ]
    results = [await _check(n, m) for n, m in cases]
    print("ALL PASS" if all(results) else "SOME FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    code = loop.run_until_complete(main())
    sys.stdout.flush()
    # Hard exit: a stuck pipeline leaves tasks that would block a clean
    # interpreter shutdown.
    os._exit(code)
