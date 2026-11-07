"""Send this Windows laptop's audio output to Boots' speaker (2026-11-07).

Spotify plays on the laptop (official app, so no librespot DRM problem); this
captures what Windows is playing (WASAPI loopback of the default output) and
streams it as raw S16LE PCM over TCP to the Pi's music receiver
(tools/music_receiver.py), which feeds the agent's music mixer -> EMEET,
ducked while Boots talks. Captures ALL laptop audio, not only Spotify.

The capture is reopened when the default output changes or no audio arrives
for REOPEN_IDLE_S: a loopback stream opened while nothing was playing (or
before a Bluetooth headset re-initialised) can stay silent forever.

Run (needs PyAudioWPatch):
    %USERPROFILE%\\boots-audio-venv\\Scripts\\pythonw.exe laptop_audio_sender.py [pi-host] [port]
"""
import socket
import sys
import time

import pyaudiowpatch as pyaudio

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.137.216"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8795
CHUNK = 960            # 20 ms at 48 kHz
REOPEN_IDLE_S = 3.0    # no audio for this long -> reopen the capture
CHECK_DEVICE_S = 5.0   # how often to look for a changed default output


def stream_once(pa, sock):
    """Capture the current default output until it goes stale or changes."""
    dev = pa.get_default_wasapi_loopback()
    sr, ch = int(dev["defaultSampleRate"]), int(dev["maxInputChannels"])
    if sr != 48000 or ch != 2:
        print(f"note: '{dev['name']}' is {sr} Hz / {ch} ch; the Pi expects 48000 Hz stereo", flush=True)
    stream = pa.open(format=pyaudio.paInt16, channels=ch, rate=sr, input=True,
                     input_device_index=dev["index"], frames_per_buffer=CHUNK)
    print(f"capturing '{dev['name']}'", flush=True)
    last_audio = last_check = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if stream.get_read_available() >= CHUNK:
                sock.sendall(stream.read(CHUNK, exception_on_overflow=False))
                last_audio = now
                continue
            if now - last_audio > REOPEN_IDLE_S:
                return                                   # silent too long: reopen
            if now - last_check > CHECK_DEVICE_S:
                last_check = now
                if pa.get_default_wasapi_loopback()["name"] != dev["name"]:
                    print("default output changed", flush=True)
                    return
            time.sleep(0.01)
    finally:
        stream.close()


def main():
    while True:
        try:
            sock = socket.create_connection((HOST, PORT), timeout=5)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(None)
            print(f"connected to {HOST}:{PORT}", flush=True)
            try:
                while True:
                    pa = pyaudio.PyAudio()   # fresh instance: picks up device changes
                    try:
                        stream_once(pa, sock)
                    finally:
                        pa.terminate()
            finally:
                sock.close()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"connection/audio problem ({e!r}); retrying in 2 s", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
