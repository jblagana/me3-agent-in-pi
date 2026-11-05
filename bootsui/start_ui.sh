#!/usr/bin/env bash
cd "$(dirname "$0")"
export LLM_BASE_URL="https://api.groq.com/openai/v1"
export LLM_API_KEY="${GROQ_API_KEY:-}"
export LLM_MODEL="qwen/qwen3.8-27b"
export ME3_DGX3_LLM_BASE="http://202.92.159.240:8001/v1"
export ME3_DGX3_API_KEY="***REMOVED-HPC-KEY***"
export ME3_DGX3_LLM_MODEL="Qwen/Qwen3.8-27B"
export ME3_BOOTSUI="http://127.0.0.1:8330"
# Optional: TAVILY_API_KEY (else web_search uses the Wikipedia fallback)
exec /home/jan/me3/.venv/bin/python /home/jan/me3/bootsui/boots_ui.py --port 8330
