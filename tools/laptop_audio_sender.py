"""Send this Windows laptop's audio output to Boots' speaker (2026-11-07).

Spotify plays on the laptop (official app, so no librespot DRM problem); this
captures what Windows is playing (WASAPI loopback of the default output) and
streams it as raw S16LE PCM over TCP to the Pi's music receiver
(tools/music_receiver.py), which feeds the agent's music mixer -> EMEET,
ducked while Boots talks. Captures ALL laptop audio, not only Spotify.

Run (needs PyAudioWPatch):
    %USERPROFILE%\\boots-audio-venv\\Scripts\\python.exe laptop_audio_sender.py [pi-host] [port]
"""
import socket
import sys
import time

import pyaudiowpatch as pyaudio

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.137.216"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8795
CHUNK = 960  # 20 ms at 48 kHz


def main():
    pa = pyaudio.PyAudio()
    while True:
        try:
            dev = pa.get_default_wasapi_loopback()
            sr, ch = int(dev["defaultSampleRate"]), int(dev["maxInputChannels"])
            if sr != 48000 or ch != 2:
                print(f"note: output device is {sr} Hz / {ch} ch; the Pi expects 48000 Hz stereo")
            sock = socket.create_connection((HOST, PORT), timeout=5)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(None)
            print(f"streaming '{dev['name']}' ({sr} Hz, {ch} ch) -> {HOST}:{PORT}  (Ctrl+C to stop)")
            stream = pa.open(format=pyaudio.paInt16, channels=ch, rate=sr, input=True,
                             input_device_index=dev["index"], frames_per_buffer=CHUNK)
            try:
                while True:
                    sock.sendall(stream.read(CHUNK, exception_on_overflow=False))
            finally:
                stream.close()
                sock.close()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"connection/audio problem ({e!r}); retrying in 2 s")
            time.sleep(2)
    pa.terminate()


if __name__ == "__main__":
    main()
