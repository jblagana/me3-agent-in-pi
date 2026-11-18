"""Boots music receiver (2026-11-07; mixes several senders since 2026-11-18).

Senders connect over TCP and send raw S16LE 48 kHz stereo PCM:
  * tools/laptop_audio_sender.py  -- the laptop's audio (Spotify app etc.)
  * tools/bt_audio_bridge.py      -- a phone/laptop playing to Boots over Bluetooth

Until 2026-11-18 only ONE sender was served at a time, so a connected laptop
sender (which stays connected even when silent) would lock Bluetooth out. Now
every sender gets a small buffer and a mixer thread sums them, 20 ms at a time,
into the music FIFO that agent.py's _SpotifyPipe reads (mixed into the EMEET
output, ducked while Boots talks).

Lag stays bounded: each sender's buffer holds at most MAX_BUF_S; when the agent
isn't reading, the FIFO write blocks, the buffers fill and their OLDEST audio
is dropped, so playback resumes near "now" instead of replaying a backlog.
"""
import os
import socket
import threading
import time

import numpy as np

FIFO = os.environ.get("ME3_SPOTIFY_FIFO", "/run/me3-spotify/spotify.pcm")
PORT = int(os.environ.get("ME3_MUSIC_RX_PORT", "8795"))
FRAME = 48000 * 2 * 2 // 50          # 20 ms of 48 kHz stereo s16 = 3840 bytes
MAX_BUF_S = 0.4
MAX_BUF = int(48000 * 2 * 2 * MAX_BUF_S) // FRAME * FRAME


class Sender:
    def __init__(self, addr):
        self.addr = addr
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.alive = True

    def push(self, data):
        with self.lock:
            self.buf += data
            over = len(self.buf) - MAX_BUF
            if over > 0:                       # drop the oldest audio, whole samples
                del self.buf[:over + (-over) % 4]

    def take(self):
        with self.lock:
            if len(self.buf) < FRAME:
                return None
            out = bytes(self.buf[:FRAME])
            del self.buf[:FRAME]
            return out


SENDERS = []
SENDERS_LOCK = threading.Lock()


def mix(frames):
    """Sum S16LE frames with clipping (one frame -> returned unchanged)."""
    if len(frames) == 1:
        return frames[0]
    acc = np.zeros(FRAME // 2, np.int32)
    for f in frames:
        acc += np.frombuffer(f, "<i2")
    return np.clip(acc, -32768, 32767).astype("<i2").tobytes()


def mixer():
    fifo = None
    while True:
        with SENDERS_LOCK:
            live = list(SENDERS)
        frames = [f for f in (s.take() for s in live) if f]
        if not frames:
            time.sleep(0.005)
            continue
        try:
            if fifo is None:
                if not os.path.exists(FIFO):
                    os.mkfifo(FIFO, 0o660)
                # O_RDWR: opening never blocks, even before the agent starts reading
                fifo = open(FIFO, "r+b", buffering=0)
            fifo.write(mix(frames))            # blocks when the agent isn't reading
        except Exception as e:
            print(f"fifo error: {e!r}", flush=True)
            try:
                fifo and fifo.close()
            except Exception:
                pass
            fifo = None
            time.sleep(0.5)


def handle(conn, addr):
    s = Sender(addr[0])
    with SENDERS_LOCK:
        SENDERS.append(s)
    print(f"sender connected from {addr[0]} ({len(SENDERS)} now)", flush=True)
    try:
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        pending = b""
        while True:
            data = conn.recv(19200)
            if not data:
                break
            data = pending + data
            cut = len(data) - len(data) % 4      # keep whole stereo samples
            pending = data[cut:]
            if cut:
                s.push(data[:cut])
    except Exception as e:
        print(f"receiver error: {e!r}", flush=True)
    finally:
        with SENDERS_LOCK:
            SENDERS.remove(s)
        conn.close()
        print(f"sender {addr[0]} disconnected", flush=True)


def serve():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(4)
    print(f"music receiver on :{PORT} -> {FIFO} (mixing senders)", flush=True)
    threading.Thread(target=mixer, daemon=True, name="mixer").start()
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=handle, args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    serve()
