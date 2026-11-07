"""Boots music receiver (2026-11-07).

Accepts ONE TCP sender at a time (tools/laptop_audio_sender.py) and copies its
raw S16LE 48 kHz stereo PCM into the music FIFO that agent.py's _SpotifyPipe
reads and mixes into the EMEET output (ducked while Boots talks).

Lag stays bounded by small kernel buffers (64 KB socket + 64 KB FIFO, about
0.7 s of 48 kHz stereo audio): when the agent isn't reading, the writes block,
TCP back-pressure reaches the laptop, and its capture simply skips audio
instead of queueing it.
"""
import os
import socket

FIFO = os.environ.get("ME3_SPOTIFY_FIFO", "/run/me3-spotify/spotify.pcm")
PORT = int(os.environ.get("ME3_MUSIC_RX_PORT", "8795"))


def serve():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(1)
    print(f"music receiver on :{PORT} -> {FIFO}", flush=True)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        print(f"sender connected from {addr[0]}", flush=True)
        try:
            if not os.path.exists(FIFO):
                os.mkfifo(FIFO, 0o660)
            # O_RDWR: opening never blocks, even before the agent starts reading
            with open(FIFO, "r+b", buffering=0) as fifo:
                while True:
                    data = conn.recv(19200)
                    if not data:
                        break
                    fifo.write(data)          # blocks when the agent isn't reading
        except Exception as e:
            print(f"receiver error: {e!r}", flush=True)
        finally:
            conn.close()
            print("sender disconnected", flush=True)


if __name__ == "__main__":
    serve()
