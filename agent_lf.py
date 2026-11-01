#
# ME3 â€” interruptible voice agent (Pipecat 1.12)
#
#   Mic (EMEET M1A) -> [Silero VAD, barge-in] -> whisper.cpp (OpenAI-ASR @127.0.0.1:8080)
#                 -> Qwen3.8-27B LLM (vLLM, direct OpenAI function-calling for ME2 UI tools)
#                 -> Piper local TTS -> Speaker
#
# Barge-in (interruption) is built into Pipecat (allow_interruptions=True): while the bot
# speaks, the VAD detects the user and the TTS is cancelled.
#
# Run (as root; needs /etc/me3/llm.env + whisper-server on :8080):
#   /home/jan/me3/.venv/bin/python -m me3.agent
#
import os
import sys

from loguru import logger


def _load_llm_env():
    try:
        with open("/etc/me3/llm.env") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except Exception as e:
        logger.warning(f"could not load /etc/me3/llm.env: {e!r}")

_load_llm_env()

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://202.92.159.240:8001/v1")
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "not-needed")
LLM_MODEL    = os.environ.get("LLM_MODEL", "Qwen/Qwen3.8-27B")
ASR_BASE_URL = os.environ.get("ME3_ASR_BASE_URL", "http://127.0.0.1:8080/v1")
ME3_ASR_MODEL = os.environ.get("ME3_ASR_MODEL", "whisper-1")
MIC_DEV = int(os.environ.get("ME3_MIC_DEVICE", "1"))
SPK_DEV = int(os.environ.get("ME3_SPEAKER_DEVICE", "0"))

logger.info("ðŸŽ™ ME3 starting...")
import pyaudio

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    LLMRunFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair, LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transports.local.audio import (
    LocalAudioInputTransport, LocalAudioOutputTransport, LocalAudioTransportParams,
)

# make sibling modules importable regardless of how agent.py is launched
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import me2_ui_mcp as ui
from piper_tts import PiperLocalTTSService
from prompt import SYSTEM_PROMPT

logger.info("âœ… components imported")


def _openai_tools():
    out = []
    for (n, d, p) in ui.TOOL_SPECS:
        out.append({"type": "function", "function": {
            "name": n, "description": d,
            "parameters": {"type": "object", "properties": p, "required": list(p.keys())}}})
    return out


async def run_bot():
    pa = pyaudio.PyAudio()
    stt = OpenAISTTService(api_key=LLM_API_KEY, base_url=ASR_BASE_URL, model=ME3_ASR_MODEL)
    llm = OpenAILLMService(
        api_key=LLM_API_KEY, model=LLM_MODEL,
        settings=OpenAILLMService.Settings(system_instruction=SYSTEM_PROMPT, temperature=0.4),
        params={"base_url": LLM_BASE_URL},
    )
    # attach ME2 UI tools to the LLM (direct function-calling)
    llm.tools = _openai_tools()
    tts = PiperLocalTTSService()

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()))

    in_params = LocalAudioTransportParams(audio_in_enabled=True, audio_in_sample_rate=16000,
                                          audio_in_channels=1, input_device_index=MIC_DEV)
    out_params = LocalAudioTransportParams(audio_out_enabled=True, audio_out_sample_rate=16000,
                                           audio_out_channels=1, output_device_index=SPK_DEV)
    in_transport = LocalAudioInputTransport(pa, in_params)
    out_transport = LocalAudioOutputTransport(pa, out_params)

    pipeline = Pipeline([
        in_transport, stt, user_aggregator, llm, tts, out_transport, assistant_aggregator,
    ])
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True, allow_interruptions=True))

    # kick off the greeting
    context.add_message({"role": "user", "content": "Greet Jan in one short sentence and say you're listening."})
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)
    pa.terminate()


async def main():
    logger.info("starting local audio (mic #%s -> speaker #%s)", MIC_DEV, SPK_DEV)
    await run_bot()


if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nME3 stopped.")

