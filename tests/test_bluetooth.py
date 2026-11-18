"""Bluetooth speaker (2026-11-18).

1. mixer   two senders are summed with clipping; one sender passes unchanged;
           a sender's buffer keeps at most MAX_BUF and drops its OLDEST audio.
2. bridge  a phone's 44.1 kHz stereo (or mono) becomes 48 kHz stereo.
3. wiring  dashboard routes, the voice tool and the router keyword exist.

Offline: no Bluetooth hardware, no network.

    cd ~/me3 && .venv/bin/python tests/test_bluetooth.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import music_receiver as rx      # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def frame(v):
    return np.full(rx.FRAME // 2, v, "<i2").tobytes()


def mixer_case():
    one = frame(1000)
    check("mixer: one sender unchanged", rx.mix([one]) == one)
    m = np.frombuffer(rx.mix([frame(1000), frame(2000)]), "<i2")
    check("mixer: two senders summed", (m == 3000).all())
    m = np.frombuffer(rx.mix([frame(30000), frame(30000)]), "<i2")
    check("mixer: clipped, not wrapped", (m == 32767).all())
    s = rx.Sender("x")
    for i in range(40):                      # 40 frames = 0.8 s, cap is 0.4 s
        s.push(frame(i))
    check("mixer: buffer capped", len(s.buf) <= rx.MAX_BUF, len(s.buf))
    first = np.frombuffer(s.take(), "<i2")[0]
    check("mixer: oldest audio dropped, newest kept", first >= 40 - rx.MAX_BUF // rx.FRAME - 1, first)
    s2 = rx.Sender("y")
    s2.push(frame(5)[:100])
    check("mixer: partial frame waits", s2.take() is None)


def bridge_case():
    try:
        import soxr
        import bt_audio_bridge as br
    except Exception as e:
        print(f"SKIP bridge ({type(e).__name__}: soxr not installed here)")
        return
    rs = soxr.ResampleStream(44100, 48000, 2, dtype="int16")
    t = np.arange(44100) / 44100
    tone = (np.sin(2 * np.pi * 440 * t) * 8000).astype("<i2")
    raw = np.repeat(tone, 2).tobytes()
    out = b"".join(br.to_48k_stereo(raw[i:i + 8820], 2, rs) for i in range(0, len(raw), 8820))
    n = len(out) // 4
    check("bridge: 1 s at 44.1 kHz -> about 48000 stereo frames", 47000 < n <= 48000, n)
    mono = br.to_48k_stereo(tone[:4800].tobytes(), 1, None)
    check("bridge: mono duplicated to stereo", len(mono) == 4800 * 4)


def wiring_case():
    src = open(os.path.join(ROOT, "bootsui", "boots_ui.py"), encoding="utf-8").read()
    html = open(os.path.join(ROOT, "bootsui", "boots_ui.html"), encoding="utf-8").read()
    check("wiring: dashboard routes", 'self.path == "/bt_pair"' in src and 'self.path == "/bt_status"' in src)
    check("wiring: dashboard button", 'id="btn-bt"' in html and "btRefresh" in html)
    me2 = open(os.path.join(ROOT, "me2_tools.py"), encoding="utf-8-sig").read()
    check("wiring: voice tool", '"boots_bluetooth_pair"' in me2 and "def _bt_pair" in me2)
    import tool_router as tr
    check("wiring: router sends it for 'pair my phone'", "home" in tr.pick_groups(["pair my phone with bluetooth"]))


if __name__ == "__main__":
    mixer_case()
    bridge_case()
    wiring_case()
    ok = all(RESULTS)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.stdout.flush()
    os._exit(0 if ok else 1)
