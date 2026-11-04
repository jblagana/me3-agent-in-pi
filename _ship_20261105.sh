#!/usr/bin/env bash
# ============================================================================
# ME3 SHIP SCRIPT — sandbox -> Pi (jan@192.168.1.3)
# STATUS: STAGED. DO NOT RUN until the boss says ship.
#
# Run from the Pi as user `jan` (or via: scp this script to the Pi, then
# `bash ~/me3/_ship_20261105.sh`). It is idempotent-ish: re-running after a
# partial ship is safe (backups are re-taken with fresh timestamps).
#
# Pre-conditions (already verified 2026-11-05, see ALIGN_TO_PI.md):
#   - Pi venv (py3.13) has requests, numpy, loguru, mcp, edge_tts, miniaudio
#   - All 7 sandbox files py_compile clean on the Pi venv
#   - me2_tools.py exposes the same API agent.py consumes (TOOL_SPECS, dispatch)
#   - Pi spotify token HAS refresh_token (auto-refresh works)
#   - keyring is NOT on the Pi -> web_search falls back to Wikipedia unless
#     TAVILY_API_KEY is exported (optional)
# ============================================================================
set -euo pipefail
cd /home/jan/me3

TS=$(date +%Y%m%d_%H%M%S)

echo "== 1) safety: git snapshot + tarball of CURRENT Pi state =="
git add -A
git commit -q -m "pre-ship snapshot $TS" || echo "(nothing to commit)"
git tag -f "pre_ship_$TS"
tar czf ~/me3_backup_pre_ship_$TS.tar.gz --exclude=me3/.venv --exclude=me3/agent.log -C ~ me3
echo "backup: ~/me3_backup_pre_ship_$TS.tar.gz + tag pre_ship_$TS"

echo "== 2) copy the 7 ship files (staged in _ship_staging/) =="
S=~/me3/_ship_staging
cp -v $S/me2_tools.py        ~/me3/me2_tools.py
cp -v $S/spotify_tools.py    ~/me3/spotify_tools.py
cp -v $S/web_tools.py        ~/me3/web_tools.py
cp -v $S/prompt.py           ~/me3/prompt.py
cp -v $S/boots_ui.py         ~/me3/bootsui/boots_ui.py
cp -v $S/boots_ui.html       ~/me3/bootsui/boots_ui.html
cp -v $S/boots_music_worker.py ~/me3/bootsui/boots_music_worker.py

echo "== 3) point agent.py at me2_tools (was me2_ui_mcp) =="
grep -q '^import me2_tools as ui$' agent.py || \
  sed -i 's/^import me2_ui_mcp as ui$/import me2_tools as ui/' agent.py
.venv/bin/python -m py_compile agent.py && echo "agent.py compiles"

echo "== 4) create start_ui.sh (Pi launcher, same env as laptop) =="
cat > ~/me3/bootsui/start_ui.sh <<'LAUNCH'
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
LAUNCH
chmod +x ~/me3/bootsui/start_ui.sh

echo "== 5) retarget the UI service: me2_ui.py -> boots_ui.py =="
cp me2-ui.service me2-ui.service.bak_$TS
sed -i 's|/home/jan/me3/bootsui/me2_ui.py --dir /home/jan/me3/bootsui --port 8330|/home/jan/me3/bootsui/boots_ui.py --port 8330|' me2-ui.service
sudo cp me2-ui.service /etc/systemd/system/me2-ui.service
sudo systemctl daemon-reload

echo "== 6) restart UI + voice agent =="
sudo systemctl restart me2-ui.service
pkill -f 'python agent.py' || true   # me3-agent.service auto-restarts it
sleep 5
ss -tlnp | grep 8330 || { echo "FATAL: :8330 not listening"; exit 1; }

echo "== 7) smoke tests =="
curl -sf http://127.0.0.1:8330/ui_state | head -c 200; echo
curl -sf -X POST http://127.0.0.1:8330/chat -H 'Content-Type: application/json' \
  -d '{"message":"say exactly: all good"}' | head -c 300; echo
curl -sf -X POST http://127.0.0.1:8330/fire -H 'Content-Type: application/json' \
  -d '{"cmd":"lights","slot":"on","conf":1.0}'; echo

echo "== 8) commit + push on the Pi =="
git add -A
git commit -q -m "ship 2026-11-05: me2_tools rename, Tavily web tools, boots_ui.py server, prompt contract"
git push origin HEAD
echo "== DONE =="
