#!/bin/bash
# Start whisper.cpp's OpenAI-compatible server on 127.0.0.1:8080 (if not running).
set -uo pipefail
cd /home/jan/whisper.cpp/build/bin
if curl -s --max-time 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
  echo "whisper-server already running"
  exit 0
fi
echo "starting whisper-server (base.en) on 127.0.0.1:8080 ..."
nohup ./whisper-server \
  --model /home/jan/whisper.cpp/models/ggml-base.en.bin \
  --host 127.0.0.1 --port 8080 \
  --language en \
  > /home/jan/me3/whisper-server.log 2>&1 &
echo "  pid $!"
# wait for it
for i in $(seq 1 30); do
  sleep 1
  if curl -s --max-time 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
    echo "  ✅ whisper-server up after ${i}s"
    echo "  models: $(curl -s http://127.0.0.1:8080/v1/models | head -c 200)"
    exit 0
  fi
done
echo "  ⚠ whisper-server did not come up; log tail:"
tail -15 /home/jan/me3/whisper-server.log
exit 1
