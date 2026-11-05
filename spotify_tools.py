#

# ME3 — Spotify tools (direct OpenAI function-calling layer).

# Calls the Spotify Web API directly using the token saved by `spotify-mcp auth`

# at ~/.spotify-mcp/tokens.json. Same pattern as me2_tools.TOOL_SPECS.

#

import json

import os

import time

import subprocess

import tempfile

import urllib.parse

from loguru import logger

def _load_token():

    path = os.path.join(os.path.expanduser("~"), ".spotify-mcp", "tokens.json")

    try:

        with open(path) as f:

            t = json.load(f)

        if t.get("expires_at") and t["expires_at"] < time.time() * 1000:

            t = _refresh_token(t)

        return t

    except Exception as e:

        logger.warning(f"could not load spotify token: {e!r}")

        return {}

def _refresh_token(tok):

    client_id = os.environ.get("SPOTIFY_CLIENT_ID", "")

    if not client_id:

        # Linux path first (Pi), then the Windows-native location next to the token.

        for env_path in (

            os.path.join(os.path.expanduser("~"), ".config", "spotify-mcp", "env"),

            os.path.join(os.path.expanduser("~"), ".spotify-mcp", "env"),

        ):

            try:

                with open(env_path) as f:

                    for line in f:

                        if line.startswith("SPOTIFY_CLIENT_ID="):

                            client_id = line.strip().split("=", 1)[1]

                            break

                if client_id:

                    break

            except Exception:

                pass

    if not client_id or not tok.get("refresh_token"):

        return tok

    body = urllib.parse.urlencode({

        "client_id": client_id,

        "grant_type": "refresh_token",

        "refresh_token": tok["refresh_token"],

    })

    cmd = ["curl", "-s", "-X", "POST", "https://accounts.spotify.com/api/token",

           "-H", "Content-Type: application/x-www-form-urlencoded", "-d", body]

    try:

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

        if r.returncode != 0:

            return tok

        new = json.loads(r.stdout)

        tok["access_token"] = new["access_token"]

        tok["expires_at"] = time.time() * 1000 + new.get("expires_in", 3600) * 1000

        if new.get("refresh_token"):

            tok["refresh_token"] = new["refresh_token"]

        path = os.path.join(os.path.expanduser("~"), ".spotify-mcp", "tokens.json")

        with open(path, "w") as f:

            json.dump(tok, f, indent=2)

        os.chmod(path, 0o600)

        logger.info("spotify token refreshed")

    except Exception as e:

        logger.warning(f"spotify token refresh failed: {e!r}")

    return tok

def _api(method, path, body=None):

    import subprocess, tempfile

    tok = _load_token()

    url = f"https://api.spotify.com/v1{path}"

    auth_header = f"Authorization: Bearer {tok.get('access_token', '')}"

    cmd = ["curl", "-s", "-w", "\n__HTTP_STATUS__:%{http_code}", "-X", method, url, "-H", auth_header]

    if body:

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:

            json.dump(body, f)

            body_file = f.name

        cmd += ["-H", "Content-Type: application/json", "-d", f"@{body_file}"]

    else:

        cmd += ["-H", "Content-Type: application/json", "-H", "Content-Length: 0"]

    try:

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

        if body:

            os.unlink(body_file)

        if r.returncode != 0:

            return {"error": f"curl failed: {r.stderr}"}

        out = r.stdout

        status = None

        if "__HTTP_STATUS__:" in out:

            out, status_s = out.rsplit("__HTTP_STATUS__:", 1)

            try:

                status = int(status_s.strip())

            except ValueError:

                status = None

        if not out.strip():

            if status is None or status < 400:

                return {"status": "ok"}

            return {"error": {"status": status, "message": "HTTP " + str(status)}}

        try:

            return json.loads(out)

        except json.JSONDecodeError:

            return {"error": "non-JSON response (HTTP " + str(status) + "): " + out[:200]}

    except Exception as e:

        return {"error": str(e)}

AUTH_MSG = ("Spotify auth is expired or missing — re-run `spotify-mcp auth` and "

            "try again. (Nothing was changed on Spotify.)")

def _auth_error(res) -> bool:

    """True when the API rejected the call on auth (401/403). Must never be

    masked as 'no results' / 'paused' / 'nothing playing'."""

    err = res.get("error") if isinstance(res, dict) else None

    return isinstance(err, dict) and err.get("status") in (401, 403)

# ---- tool specs (name, description, param dict) ----

TOOL_SPECS = [

    (

        "spotify_play_search",

        "Search Spotify and play the first matching track. Use when the user says "

        "something like 'play <song> by <artist>' or 'play some <genre>'.",

        {"query": "string"},

    ),

    ("spotify_pause", "Pause the current Spotify playback.", {}),

    ("spotify_resume", "Resume (unpause) the current Spotify playback.", {}),

    ("spotify_next", "Skip to the next track on Spotify.", {}),

    ("spotify_previous", "Go back to the previous track on Spotify.", {}),

    (

        "spotify_now_playing",

        "Get the track currently playing on Spotify. Use when the user asks "

        "'what's playing' or 'what song is this'.",

        {},

    ),

    (

        "spotify_search",

        "Search Spotify for tracks/artists/albums and return a short list. "

        "Use when the user asks to find something without playing it.",

        {"query": "string"},

    ),

]

def dispatch(tool: str, **kw) -> str:

    """Execute a Spotify tool and return a short human-readable string."""

    if tool == "spotify_play_search":

        q = kw.get("query", "")

        res = _api("GET", f"/search?q={urllib.parse.quote(q)}&type=track&limit=1")

        if _auth_error(res):

            return AUTH_MSG

        tracks = res.get("tracks", {}).get("items", [])

        if not tracks:

            return f"Sorry, I couldn't find anything for '{q}' on Spotify."

        uri = tracks[0]["uri"]

        name = tracks[0]["name"]

        artist = tracks[0]["artists"][0]["name"] if tracks[0].get("artists") else "unknown"

        # Prefer a specific active device if one is connected.

        play_body = {"uri": uri, "position_ms": 0}

        devs = _api("GET", "/me/player/devices").get("devices", [])

        active = [d for d in devs if d.get("is_active")]

        if active:

            play_body["device_id"] = active[0]["id"]

        play = _api("PUT", "/me/player/play", play_body)

        if _auth_error(play):

            return AUTH_MSG

        if "error" in play:

            reason = play.get("error", {}).get("reason", "")

            if reason == "NO_ACTIVE_DEVICE":

                return (f"Found '{name}' by {artist}, but there's no active Spotify device. "

                        f"Open Spotify on your laptop or phone and start anything once, "

                        f"then ask me to play it again.")

            return f"I found '{name}' by {artist} but couldn't start playback: {play.get('error', {}).get('message', reason)}."

        return f"Playing {name} by {artist}."

    elif tool == "spotify_pause":

        res = _api("PUT", "/me/player/pause")

        if _auth_error(res):

            return AUTH_MSG

        return "Paused."

    elif tool == "spotify_resume":

        res = _api("PUT", "/me/player/play")

        if _auth_error(res):

            return AUTH_MSG

        return "Resuming."

    elif tool == "spotify_next":

        res = _api("POST", "/me/player/next")

        if _auth_error(res):

            return AUTH_MSG

        return "Skipped to the next track."

    elif tool == "spotify_previous":

        res = _api("POST", "/me/player/previous")

        if _auth_error(res):

            return AUTH_MSG

        return "Went back to the previous track."

    elif tool == "spotify_now_playing":

        res = _api("GET", "/me/player")

        if _auth_error(res):

            return AUTH_MSG

        if "error" in res:

            return "Nothing is playing right now."

        item = res.get("item", {})

        if not item:

            return "Nothing is playing right now."

        name = item.get("name", "unknown")

        artist = item["artists"][0]["name"] if item.get("artists") else ""

        return f"Currently playing {name} by {artist}."

    elif tool == "spotify_search":

        q = kw.get("query", "")

        res = _api("GET", f"/search?q={urllib.parse.quote(q)}&type=track,artist,album&limit=5")

        if _auth_error(res):

            return AUTH_MSG

        parts = []

        for t in res.get("tracks", {}).get("items", [])[:3]:

            a = t["artists"][0]["name"] if t.get("artists") else ""

            parts.append(f"'{t['name']}' by {a}")

        for a in res.get("artists", {}).get("items", [])[:2]:

            parts.append(f"artist: {a['name']}")

        if not parts:

            return f"No results for '{q}'."

        return "Found: " + ", ".join(parts)

    else:

        return f"Unknown tool: {tool}"

if __name__ == "__main__":

    import sys

    t = sys.argv[1] if len(sys.argv) > 1 else "spotify_now_playing"

    print(dispatch(t, query="Opalite Taylor Swift"))
