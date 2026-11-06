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

        "Search Spotify and play it. The speech recognizer often mishears song and "
        "artist names: BEFORE calling, correct the query to the real title and add the "
        "artist if you know it (e.g. heard 'bohemian rap city' -> 'Bohemian Rhapsody "
        "Queen'). kind: 'track' (a specific song, default), 'artist' (play some <artist>), "
        "'playlist' (a genre, mood or vibe like 'chill jazz'). If the result says it is "
        "NOT sure, read the options to the user and wait; do not call again until they pick.",

        {"query": "string", "kind": "string"},

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

# Spotify Connect device to use when nothing is active: the Pi's own librespot
# ("Boots", 2026-11-07). Overridable with SPOTIFY_DEVICE_NAME.
SPOTIFY_DEVICE_NAME = os.environ.get("SPOTIFY_DEVICE_NAME", "Boots")


def _pick_device():
    """Active device if any, else the one named SPOTIFY_DEVICE_NAME, else None."""
    try:
        devs = _api("GET", "/me/player/devices").get("devices", [])
    except Exception:
        return None
    for d in devs:
        if d.get("is_active"):
            return d.get("id")
    for d in devs:
        if (d.get("name") or "").lower() == SPOTIFY_DEVICE_NAME.lower():
            return d.get("id")
    return None


OPTIONAL_PARAMS = {"spotify_play_search": {"kind"}}

# ---- match scoring (2026-11-07) -------------------------------------------
import re as _re
from difflib import SequenceMatcher as _SM

PLAY_SURE = float(os.environ.get("ME3_SPOTIFY_SURE", "0.72"))   # play without asking
PLAY_MIN = float(os.environ.get("ME3_SPOTIFY_MIN", "0.35"))     # below -> "not found"
_FILLER = {"the", "a", "an", "by", "song", "songs", "track", "music", "play", "some",
           "of", "feat", "ft", "featuring", "please"}


def _norm(s):
    s = (s or "").lower()
    s = _re.sub(r"\s*[\(\[].*?[\)\]]", " ", s)          # (Remastered 2011), [Live]
    s = _re.sub(r"\s+-\s+.*$", " ", s)                  # " - Radio Edit"
    s = _re.sub(r"[^\w\s]", " ", s)
    return [w for w in s.split() if w and w not in _FILLER]


def _tok_hit(w, pool):
    return max((_SM(None, w, p).ratio() for p in pool), default=0.0) >= 0.8


def _cover(a, b):
    """Fraction of tokens in a that (fuzzily) appear in b."""
    return sum(_tok_hit(w, b) for w in a) / len(a) if a else 0.0


def _score_track(q, t):
    title = _norm(t.get("name"))
    artists = [w for a in t.get("artists", []) for w in _norm(a.get("name"))]
    qcov = _cover(q, title + artists)          # did we find everything that was said?
    tcov = _cover(title, q)                    # is the title actually what was said?
    whole = _SM(None, " ".join(q), " ".join(title + artists)).ratio()
    return 0.5 * qcov + 0.35 * tcov + 0.15 * whole


def _choose(query, kind, res):
    """Pick what to play, or say we must ask. Returns a dict with action
    'play' | 'ask' | 'none'."""
    q = _norm(query) or query.lower().split()
    tracks = [t for t in (res.get("tracks") or {}).get("items") or [] if t]
    artists = [a for a in (res.get("artists") or {}).get("items") or [] if a]
    lists = [p for p in (res.get("playlists") or {}).get("items") or [] if p]

    def art(t):
        return t["artists"][0]["name"] if t.get("artists") else ""

    if kind == "playlist" and lists:
        p = lists[0]
        return {"action": "play", "type": "playlist", "uri": p["uri"], "name": p["name"], "artist": ""}

    best_artist = None
    if artists:
        a = artists[0]
        s = _SM(None, " ".join(q), " ".join(_norm(a["name"]))).ratio()
        best_artist = (s, a)
        if kind == "artist" and s >= 0.6:
            return {"action": "play", "type": "artist", "uri": a["uri"], "name": a["name"], "artist": ""}

    # score tracks; Spotify's order (popularity) breaks ties, dupes collapsed
    scored, seen = [], set()
    for i, t in enumerate(tracks):
        key = (" ".join(_norm(t.get("name"))), art(t).lower())
        if key in seen:
            continue
        seen.add(key)
        scored.append((_score_track(q, t) - 0.01 * i, t))
    scored.sort(key=lambda x: -x[0])
    logger.info("spotify match '{}': {}", query,
                [(round(s, 2), t.get("name"), art(t)) for s, t in scored[:4]])

    # "play Taylor Swift" -> the query IS an artist name
    if best_artist and best_artist[0] >= 0.85 and (not scored or scored[0][0] < 0.9):
        a = best_artist[1]
        return {"action": "play", "type": "artist", "uri": a["uri"], "name": a["name"], "artist": ""}

    if scored and scored[0][0] >= PLAY_SURE:
        s, t = scored[0]
        return {"action": "play", "type": "track", "uri": t["uri"], "name": t["name"], "artist": art(t)}
    if kind == "playlist" or (not scored and lists):
        if lists:
            p = lists[0]
            return {"action": "play", "type": "playlist", "uri": p["uri"], "name": p["name"], "artist": ""}
    cands = [f"{t['name']} by {art(t)}" for s, t in scored[:2] if s >= PLAY_MIN]
    if cands:
        return {"action": "ask", "options": cands}
    return {"action": "none"}


def dispatch(tool: str, **kw) -> str:

    """Execute a Spotify tool and return a short human-readable string."""

    if tool == "spotify_play_search":

        q = (kw.get("query") or "").strip()
        kind = (kw.get("kind") or "track").strip().lower()
        if not q:
            return "What should I play?"

        # (2026-11-07) No more blind "play result #1": misheard names used to
        # start a random song. Pull several candidates, score them against what
        # was asked, and ASK when nothing is a clear match.
        res = _api("GET", f"/search?q={urllib.parse.quote(q)}&type=track,artist,playlist&limit=8")

        if _auth_error(res):

            return AUTH_MSG

        pick = _choose(q, kind, res)
        if pick["action"] == "none":
            return f"Sorry, I couldn't find anything for '{q}' on Spotify."
        if pick["action"] == "ask":
            opts = " or ".join(pick["options"])
            return (f"NOT sure which one you meant (nothing matched '{q}' closely). "
                    f"Ask the user: did you mean {opts}?")

        name, artist = pick["name"], pick["artist"]
        # (2026-11-07) Spotify wants {"uris": [...]} (track) or {"context_uri": ...}
        # (artist/playlist) in the body and device_id as a QUERY parameter. The old
        # {"uri": ...} body was ignored -> Spotify just RESUMED whatever was playing.
        dev = _pick_device()
        play_body = ({"uris": [pick["uri"]], "position_ms": 0} if pick["type"] == "track"
                     else {"context_uri": pick["uri"]})
        play = _api("PUT", "/me/player/play" + (f"?device_id={dev}" if dev else ""), play_body)
        if pick["type"] == "artist":
            name, artist = f"songs by {name}", ""
        elif pick["type"] == "playlist":
            name, artist = f"the playlist {name}", ""

        if _auth_error(play):

            return AUTH_MSG

        if "error" in play:

            reason = play.get("error", {}).get("reason", "")

            if reason == "NO_ACTIVE_DEVICE":

                return (f"Found '{name}' by {artist}, but there's no active Spotify device. "

                        f"Open Spotify on your laptop or phone and start anything once, "

                        f"then ask me to play it again.")

            return f"I found '{name}' by {artist} but couldn't start playback: {play.get('error', {}).get('message', reason)}."

        return f"Playing {name}" + (f" by {artist}." if artist else ".")

    elif tool == "spotify_pause":

        res = _api("PUT", "/me/player/pause")

        if _auth_error(res):

            return AUTH_MSG

        return "Paused."

    elif tool == "spotify_resume":

        dev = _pick_device()
        res = _api("PUT", "/me/player/play" + (f"?device_id={dev}" if dev else ""))

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
