#
# ME3 â€” Local ASR via whisper.cpp (Pipecat STTService)
# Uses our compiled whisper.cpp (base.en) â€” no cloud, no whisper-server HTTP.
# Buffers the user's speech (delimited by VAD), then transcribes locally.
#
import asyncio
import ctypes
import os

import numpy as np

from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.openai.stt import BaseWhisperSTTService
from pipecat.transcriptions.language import Language

WHISPER_MODEL = os.environ.get("ME3_ASR_MODEL_PATH", "/home/jan/whisper.cpp/models/ggml-base.en.bin")
WHISPER_LIB = os.environ.get("ME3_ASR_LIB", "/home/jan/whisper.cpp/build/bin/libwhisper.so")

# Quiet-room hardening: ignore VAD-triggered turns whose audio RMS is below this
# threshold (ambient hiss / mic self-noise), so a faint trigger never becomes a transcript.
NOISE_FLOOR_RMS = float(os.environ.get("ME3_NOISE_FLOOR_RMS", "0.012"))


class WhisperCPPSTTService(BaseWhisperSTTService):
    """Local ASR backed by whisper.cpp (ggml-base.en)."""

    def __init__(self, **kwargs):
        super().__init__(
            model="whisper.cpp", api_key="local", base_url="local",
            language=Language.EN,
            http_client=None,
            **{k: v for k, v in kwargs.items() if k not in ("model", "api_key", "base_url", "http_client")},
        )
        self._lib = None
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_lock = asyncio.Lock()
        self._load()

    def _load(self):
        try:
            self._lib = ctypes.CDLL(WHISPER_LIB)
            logger.info(f"[asr-whisper.cpp] lib loaded: {WHISPER_LIB}")
        except Exception as e:
            logger.error(f"[asr-whisper.cpp] failed to load lib {WHISPER_LIB}: {e!r}")

    # whisper.cpp C API (ctypes)
    def _transcribe_sync(self, samples: np.ndarray) -> str:
        # whisper.cpp >= 1.9 (unified API): whisper_init_from_file loads model +
        # context in one call; no separate backend_init (auto-init).
        import ctypes as C

        class _FullParams(C.Structure):
            _fields_ = [("b", C.c_char * 2048)]

        L = self._lib
        L.whisper_init_from_file.argtypes = [C.c_char_p]
        L.whisper_init_from_file.restype = C.c_void_p
        L.whisper_full_default_params.argtypes = [C.c_int]
        L.whisper_full_default_params.restype = _FullParams
        L.whisper_full.argtypes = [C.c_void_p, C.POINTER(_FullParams),
                                  C.POINTER(C.c_float), C.c_int]
        L.whisper_full.restype = C.c_int
        L.whisper_full_n_segments.argtypes = [C.c_void_p]
        L.whisper_full_n_segments.restype = C.c_int
        L.whisper_full_get_segment_text.argtypes = [C.c_void_p, C.c_int]
        L.whisper_full_get_segment_text.restype = C.c_char_p
        L.whisper_free.argtypes = [C.c_void_p]
        L.whisper_free.restype = None

        ctx = L.whisper_init_from_file(WHISPER_MODEL.encode())
        if not ctx:
            raise RuntimeError("whisper context init failed")
        try:
            fp = L.whisper_full_default_params(0)  # 0 = WHISPER_SAMPLING_GREEDY
            n = len(samples)
            L.whisper_full(ctx, C.byref(fp),
                           samples.ctypes.data_as(C.POINTER(C.c_float)), n)
            out = []
            for i in range(L.whisper_full_n_segments(ctx)):
                out.append(L.whisper_full_get_segment_text(ctx, i).decode(errors="ignore"))
            return " ".join(out).strip()
        finally:
            L.whisper_free(ctx)

    # -- STTService interface -------------------------------------------------
    async def process_frame(self, frame: Frame, direction=None):
        from pipecat.frames.frames import (
            UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
            InputAudioRawFrame, VADUserStoppedSpeakingFrame,
            InterruptionFrame, VADUserStoppedSpeakingFrame as _V,
        )
        if isinstance(frame, UserStartedSpeakingFrame):
            self._buf = np.zeros(0, dtype=np.float32)
            await self.push_frame(frame)
            return
        if isinstance(frame, (InputAudioRawFrame,)):
            try:
                a = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
                self._buf = np.concatenate([self._buf, a])
            except Exception:
                pass
            await self.push_frame(frame)
            return
        if isinstance(frame, (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            audio = self._buf
            self._buf = np.zeros(0, dtype=np.float32)
            # quiet-room noise-floor gate: skip turns that were just ambient hiss
            rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
            if rms < NOISE_FLOOR_RMS:
                logger.debug(f"[asr] turn below noise floor (rms={rms:.5f} < {NOISE_FLOOR_RMS}); skipping")
                await self.push_frame(frame)
                return
            if len(audio) > 16000 * 0.2 and self._lib is not None:
                try:
                    text = await asyncio.to_thread(self._transcribe_sync, audio)
                    if text:
                        await self.push_frame(TranscriptionFrame(text, "", "local"))
                except Exception as e:
                    logger.error(f"[asr] transcription failed: {e!r}")
            await self.push_frame(frame)
            return
        # On barge-in, pipecat cancels the turn BEFORE the stop-speaking frame
        # reaches us, so the buffered audio would be lost -> [BLANK_AUDIO].
        # Flush here so the user's speech is still transcribed.
        if isinstance(frame, InterruptionFrame) and len(self._buf) > 16000 * 0.2 and self._lib is not None:
            audio = self._buf
            self._buf = np.zeros(0, dtype=np.float32)
            try:
                text = await asyncio.to_thread(self._transcribe_sync, audio)
                if text:
                    await self.push_frame(TranscriptionFrame(text, "", "local"))
            except Exception as e:
                logger.error(f"[asr] transcription on interruption failed: {e!r}")
        await self.push_frame(frame)
