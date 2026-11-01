#!/usr/bin/env bash
# ME3 launcher (durable; ~/me3/startme3.sh).
# Voice I/O: ME3_VOICE=remote (default) -> DGX3 Qwen2-Audio ASR / Qwen3-TTS
# over the UP SHARC VPN. ME3_VOICE=local -> whisper.cpp + Piper (offline fallback).
export ME3_VOICE="${ME3_VOICE:-remote}"
export ME3_NOISE_FLOOR_RMS="${ME3_NOISE_FLOOR_RMS:-0.006}"
cd /home/jan/me3
exec .venv/bin/python agent.py