"""Boots Bluetooth speaker: phone audio -> music receiver (2026-11-18).

BlueALSA (bluealsa -p a2dp-sink) turns a connected phone/laptop into a PCM
we can read. This bridge watches for that PCM, reads it with
`bluealsa-cli open <pcm>` (raw S16LE at the phone's rate, usually 44.1 kHz),
resamples to 48 kHz stereo and sends it to tools/music_receiver.py on
127.0.0.1:8795, the same way the laptop sender does. The receiver mixes it into
the music FIFO, so Boots still ducks the music while it talks.

Pairing is tools/bt_agent.py's job; this only moves audio.
"""
import os
import re
import socket
import subprocess
import time

import numpy as np
import soxr

RX = ("127.0.0.1", int(os.environ.get("ME3_MUSIC_RX_PORT", "8795")))
CLI = "bluealsa-cli"
POLL_S = 1.0
READ = 4096


def sink_pcm():
    """D-Bus path of a connected phone's A2DP stream, or None."""
    try:
        out = subprocess.run([CLI, "list-pcms"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    for line in out.split():
        if "/a2dpsnk/" in line and line.endswith("/source"):
            return line.strip()
    return None


def pcm_format(path):
    """(rate, channels) from `bluealsa-cli info`."""
    rate, ch = 44100, 2
    try:
        out = subprocess.run([CLI, "info", path], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"(?:Rate|Sampling):\s*(\d+)", out)
        if m:
            rate = int(m.group(1))
        m = re.search(r"Channels:\s*(\d+)", out)
        if m:
            ch = int(m.group(1))
    except Exception:
        pass
    return rate, ch


def to_48k_stereo(raw, ch, rs):
    x = np.frombuffer(raw, "<i2")
    if ch == 1:
        x = np.repeat(x, 2)
    x = x.reshape(-1, 2)
    if rs is not None:
        x = rs.resample_chunk(x)
    return np.ascontiguousarray(x, dtype="<i2").tobytes()


def stream(path):
    rate, ch = pcm_format(path)
    print(f"phone connected: {path} ({rate} Hz, {ch} ch)", flush=True)
    rs = soxr.ResampleStream(rate, 48000, 2, dtype="int16") if rate != 48000 else None
    sock = socket.create_connection(RX, timeout=5)
    sock.settimeout(None)
    proc = subprocess.Popen([CLI, "open", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame = 2 * ch
    pending = b""
    try:
        while True:
            data = proc.stdout.read1(READ) if hasattr(proc.stdout, "read1") else proc.stdout.read(READ)
            if not data:
                break
            data = pending + data
            cut = len(data) - len(data) % frame
            pending = data[cut:]
            if cut:
                sock.sendall(to_48k_stereo(data[:cut], ch, rs))
    finally:
        proc.kill()
        sock.close()
        print("phone stream ended", flush=True)


def main():
    print("bluetooth audio bridge: waiting for a phone", flush=True)
    while True:
        path = sink_pcm()
        if path:
            try:
                stream(path)
            except Exception as e:
                print(f"bridge error: {e!r}", flush=True)
                time.sleep(2)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
