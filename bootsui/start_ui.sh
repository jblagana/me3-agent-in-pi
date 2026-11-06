#!/usr/bin/env bash
# Manual start of the Boots UI server (the service is me2-ui.service).
# All keys live in /etc/me3/groq.env (root:jan 640) -- never in this repo.
cd "$(dirname "$0")"
set -a
[ -r /etc/me3/groq.env ] && . /etc/me3/groq.env
set +a
export ME3_BOOTSUI="${ME3_BOOTSUI:-http://127.0.0.1:8330}"
exec /home/jan/me3/.venv/bin/python /home/jan/me3/bootsui/boots_ui.py --port 8330
