"""TTS fallback: edge-tts down -> Piper Ryan on the Pi (2026-11-18).

1. pieces   text is split at clauses/sentences with Roger-like pauses; a
            short tag after a comma (", Jan.") stays attached.
2. voice    piper_voice.synthesize gives 48 kHz int16 audio of a sane length.
3. stream   the streaming path (replies) falls back to Piper when edge-tts
            fails before any audio, and never calls Qwen3.
4. whole    _synthesize (welcome path) falls back to Piper too.

No network: edge-tts is made to fail. Needs the Ryan model on the Pi.

    cd ~/me3 && .venv/bin/python tests/test_tts_fallback.py
"""
import asyncio
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("ME3_BENCH", "1")

import piper_voice as pv   # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def pieces_case():
    p = pv.pieces("Lights are on now, Jan. Want me to bump up the brightness?")
    check("pieces: short tag stays attached", p[0][0] == "Lights are on now, Jan." and p[0][1] == pv.SENT_S, str(p))
    p = pv.pieces("Sure. I set a timer for ten minutes, and I'll tell you when it's done.")
    check("pieces: sentence and comma pauses",
          [x[1] for x in p] == [pv.SENT_S, pv.COMMA_S, 0.0], str(p))
    check("pieces: empty text", pv.pieces("") == [])


def voice_case():
    if not pv.available():
        print("SKIP voice (model not installed here)")
        return False
    raw = pv.synthesize("It's twenty eight degrees right now.", 48000)
    secs = len(raw) / 2 / 48000
    check("voice: 48 kHz audio of a sane length", 1.0 < secs < 4.0, f"{secs:.2f}s")
    return True


async def service_case():
    try:
        import remote_voice as rv
        from pipecat.frames.frames import TTSAudioRawFrame
    except Exception as e:
        print(f"SKIP service (remote_voice not importable here: {type(e).__name__})")
        return
    qwen_calls = []

    async def dead_edge(text):
        raise ConnectionError("edge down (test)")
        yield  # pragma: no cover

    rv._edge_chunks = dead_edge
    rv._EdgeSynth.synthesize = staticmethod(lambda *a, **k: (_ for _ in ()).throw(ConnectionError("edge down")))
    orig_q = rv.Qwen3TTSService._synthesize
    rv.Qwen3TTSService._synthesize = lambda self, t, sr=16000: qwen_calls.append(t) or b""
    try:
        tts = rv.BootsTTSService()
        frames = [f async for f in tts.run_tts("Sure, the lights are on now.", "ctx1")]
        audio = sum(len(f.audio) for f in frames if isinstance(f, TTSAudioRawFrame))
        check("stream: edge fails -> Piper speaks", audio > 48000 * 2 * 0.8 and not qwen_calls,
              f"{audio} B, qwen {qwen_calls}")
        tts._edge_ok = True                    # welcome path, edge "up" but failing
        raw = tts._synthesize("Hi, I'm Boots.", 48000)
        check("whole: _synthesize falls back to Piper", len(raw) > 48000 and not qwen_calls, f"{len(raw)} B")
    finally:
        rv.Qwen3TTSService._synthesize = orig_q


if __name__ == "__main__":
    pieces_case()
    if voice_case():
        asyncio.new_event_loop().run_until_complete(service_case())
    ok = all(RESULTS)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.stdout.flush()
    os._exit(0 if ok else 1)
