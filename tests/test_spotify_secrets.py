"""Spotify tokens never reach argv, logs or the LLM (2026-11-18).

The refresh token leaked into bootsui/ui.log: curl got it on the command
line, a timeout's TimeoutExpired repr carried the command, and the warning
logged it. The access token had the same path back to the LLM via _api's
{"error": str(e)}.

Offline: subprocess.run is faked, no network.

    cd ~/me3 && .venv/bin/python tests/test_spotify_secrets.py
"""
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import spotify_tools as sp   # noqa: E402
from loguru import logger    # noqa: E402

RESULTS = []
FAKE_REFRESH = "REFRESHtoken_" + "x" * 40
FAKE_ACCESS = "ACCESStoken_" + "y" * 40


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


seen = []


def fake_run(cmd, **kw):
    seen.append((list(cmd), kw.get("input") or ""))
    raise subprocess.TimeoutExpired(cmd, 15)


if __name__ == "__main__":
    log = io.StringIO()
    logger.add(log, level="DEBUG")
    real = subprocess.run
    subprocess.run = fake_run
    os.environ["SPOTIFY_CLIENT_ID"] = "client123"
    try:
        sp._refresh_token({"refresh_token": FAKE_REFRESH})
        sp._load_token = lambda: {"access_token": FAKE_ACCESS}
        out = sp._api("GET", "/me/player")
    finally:
        subprocess.run = real
    argv = " ".join(" ".join(c) for c, _ in seen)
    check("refresh token not on the command line", FAKE_REFRESH not in argv)
    check("refresh token sent on stdin", FAKE_REFRESH in seen[0][1])
    check("access token not on the command line", FAKE_ACCESS not in argv)
    check("access token sent on stdin", FAKE_ACCESS in seen[1][1])
    check("timeout log has no token", FAKE_REFRESH not in log.getvalue() and FAKE_ACCESS not in log.getvalue()
          and "TimeoutExpired" in log.getvalue())
    check("error returned to the LLM has no token", FAKE_ACCESS not in str(out) and "15 seconds" in str(out), str(out))
    ok = all(RESULTS)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.stdout.flush()
    os._exit(0 if ok else 1)
