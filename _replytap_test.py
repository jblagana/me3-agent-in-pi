# Drive the REAL ReplyTap processor (imported from the deployed agent.py) with
# synthetic LLM frames and verify the reply lands in the live UI /ui_state.me3.
# Proves the ReplyTap fix works end-to-end without needing the mic.
import asyncio, json, time, urllib.request, importlib.util, os

os.environ["ME3_MIC_DEV"] = "9"
os.environ["ME3_SPEAKER_DEV"] = "9"
spec = importlib.util.spec_from_file_location("me3agent", "/home/jan/me3/agent.py")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    print("IMPORT NOTE:", repr(e)[:120])

ReplyTap = mod.ReplyTap
from pipecat.frames.frames import (LLMFullResponseStartFrame, LLMTextFrame,
                                   LLMFullResponseEndFrame)

REPLY = "Hey Jan, I am here and ready to help."

async def main():
    tap = ReplyTap()
    tokens = [REPLY[i:i+6] for i in range(0, len(REPLY), 6)]
    await tap.process_frame(LLMFullResponseStartFrame(), None)
    for t in tokens:
        await tap.process_frame(LLMTextFrame(text=t), None)
    await tap.process_frame(LLMFullResponseEndFrame(), None)
    await asyncio.sleep(1.5)
    with urllib.request.urlopen("http://127.0.0.1:8330/ui_state", timeout=2) as r:
        m = json.load(r)["me3"]
    print("AFTER LLM-REPLY TURN:")
    for k in ("phase", "thinking", "response", "tool"):
        print("  %s = %r" % (k, m.get(k)))
    ok = (m.get("response") == REPLY and m.get("thinking") == "")
    print("REPLY_TAP_TEXT_TURN:", "PASS" if ok else "FAIL")
    # reset UI to listening
    body = {"phase": "listening", "thinking": "", "response": "", "tool": "",
            "tool_args": "", "heard": "", "active": True}
    req = urllib.request.Request("http://127.0.0.1:8330/me3",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=2) as r:
        r.read()
    print("reset to listening")

asyncio.run(main())
