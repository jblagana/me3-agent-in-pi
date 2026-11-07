#!/usr/bin/env python3
# Boots  -- Web tools (TOOL_SPECS + dispatch; MCP facade lives in mcp_server.py).
#
# Tools the voice agent can use:
#   1. web_search   -- "search the web for X" -> Wikipedia opensearch + REST
#                      summary extract (reliable, no API key, no bot-gate).
#   2. fetch_url    -- "look at this webpage / read X" -> fetch a URL and return
#                      its readable text (for a specific page the user names).
#   3. check_weather / write_website -- weather lookup + simple site builder.
#
# ARCHITECTURE (mirrors me2_tools.py, see its D11 note):
#   * This file is a plain module: TOOL_SPECS + dispatch(). The agent imports
#     it for its native FunctionSchema layer; mcp_server.py imports it for the
#     unified MCP facade (one process, all tools, stdio transport).
#   * agent.py cannot host pipecat 1.12's MCPService on this stack (D11), so
#     the RUNNING voice agent calls dispatch() directly via FunctionSchema
#     handlers. The classwork "build & run your MCP servers (on RPi)"
#     requirement is satisfied by mcp_server.py, runnable standalone with
#     `python3 mcp_server.py` or any MCP client / `mcp dev`.
#
# Why Wikipedia for "web search" (no key, no bot-gate):
#   DuckDuckGo's HTML endpoint is bot-gated from the Pi's IP (CAPTCHA wall).
#   Wikipedia's MediaWiki API + REST summary endpoint need no key, are not
#   bot-gated, and return clean JSON text -- a dependable knowledge source for
#   a voice agent. A broader web index (SearXNG, Brave, Tavily) can be swapped
#   in later by adding an API key / instance; the tool signature stays the same.
#
import json
import os
import sys
import re
import html as _html
import urllib.parse
import urllib.request
import urllib.error

USER_AGENT = "BootsVoiceAgent/1.0 (Raspberry Pi; contact: jan)"
TIMEOUT = float(os.environ.get("WEB_TOOL_TIMEOUT", "8.0"))

def _get(url: str, headers=None, timeout=None, retries: int = 2) -> str:
    """GET a URL, return the body as text. Fail-soft (returns '' on error).

    Retries on transient failures and sets a real User-Agent so content hosts
    (Wikipedia) don't 403 the default urllib agent. Returns '' on any error so
    callers can degrade gracefully instead of hanging the pipeline."""
    h = {"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*"}
    if headers:
        h.update(headers)
    to = timeout or TIMEOUT
    last = ""
    for _ in range(max(1, retries)):
        req = urllib.request.Request(url, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=to) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            # 4xx = permanent (bot-blocked / not found) -> stop retrying.
            if 400 <= e.code < 500:
                last = ""
                break
            last = ""
        except Exception:
            last = ""
        import time as _t
        _t.sleep(0.4)
    return last

# ---------------------------------------------------------------------------
# Tavily -- real web search (primary when a key is available)
# ---------------------------------------------------------------------------
def _tavily_key() -> str:
    """Tavily key: TAVILY_API_KEY env first, else Windows Credential Manager
    (same store muji uses: target 'muji', key 'tavily_api_key')."""
    k = os.environ.get("TAVILY_API_KEY", "").strip()
    if k:
        return k
    try:
        import keyring
        k = keyring.get_password("muji", "tavily_api_key") or ""
    except Exception:
        k = ""
    return k.strip()

def _tavily_search(query: str, max_results: int = 4) -> str:
    """One bounded Tavily search -> spoken summary. Returns '' on any failure
    (the caller then falls back to Wikipedia)."""
    key = _tavily_key()
    if not key:
        return ""
    try:
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=json.dumps({
                "api_key": key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": True,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        answer = (data.get("answer") or "").strip()
        results = data.get("results") or []
        bits = []
        for res in results[:max_results]:
            title = (res.get("title") or "").strip()
            url = (res.get("url") or "").strip()
            content = (res.get("content") or "").strip()[:300]
            if title or content:
                bits.append(f"{title}. {content} ({url})")
        if answer:
            # Answer + the sources it came from (2026-11-07): the answer-only
            # result left the model no URLs to fetch_url or cite in
            # write_website's `sources`.
            srcs = "; ".join(f"{(r.get('title') or '').strip()[:70]} ({(r.get('url') or '').strip()})"
                             for r in results[:3] if r.get("url"))
            return (f"Here's what I found about {query}: {answer[:500]}"
                    + (f" Sources: {srcs}." if srcs else ""))
        if bits:
            return f"Here's what I found about {query}: " + " ".join(bits)[:800]
        return ""
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# web_search -- Tavily (primary) -> Wikipedia opensearch fallback
# ---------------------------------------------------------------------------
def _web_search(query: str) -> str:
    """Search the web (Tavily first, Wikipedia fallback) for `query`.

    Fail-soft and BOUNDED: every network call has a timeout + bounded retries,
    and if both sources are unreachable/blocked we return a short spoken
    apology (never a hang), so the voice pipeline is never left wedged on
    'Running tool: web_search'."""
    q = (query or "").strip()
    if not q:
        return "I need a search term. What should I look up?"

    # 0) Tavily (real web index) when a key is available (env or keyring).
    tav = _tavily_search(q)
    if tav:
        return tav

    blocked = False

    # 1) Candidate articles via MediaWiki FULL-TEXT search (2026-11-07).
    #    opensearch is a title-PREFIX match: "Mayon Volcano height" found
    #    nothing, so every research query fell through to "couldn't find".
    api = ("https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch="
           + urllib.parse.quote(q) + "&srlimit=5&format=json")
    body = _get(api)
    candidates = []  # list of (title, url, snippet)
    if not body:
        blocked = True
    else:
        try:
            for hit in (json.loads(body).get("query") or {}).get("search") or []:
                t = hit.get("title") or ""
                snip = _strip_html(hit.get("snippet") or "")
                candidates.append((t, "https://en.wikipedia.org/wiki/"
                                   + urllib.parse.quote(t.replace(" ", "_")), snip))
        except Exception:
            blocked = True

    # 2) Top hit's REST summary (clean plain text) + the other hits' titles
    #    and URLs, so a research step can fetch_url the best ones next.
    if candidates:
        title, url, _ = candidates[0]
        api_title = urllib.parse.quote(title.replace(" ", "_"))
        summ = _get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{api_title}")
        extract = ""
        if summ:
            try:
                extract = (json.loads(summ).get("extract") or "").strip()[:700]
            except Exception:
                pass
        others = "; ".join(f"{t} ({u})" for t, u, _s in candidates[1:4])
        if extract:
            return (f"Here's what I found about {title} ({url}): {extract}"
                    + (f" Related pages: {others}." if others else ""))
        if others or candidates:
            return ("Top results: " + "; ".join(
                f"{t} ({u}) — {s[:120]}" for t, u, s in candidates[:4]))

    # 3) Fallback: try a REST summary directly on the raw query.
    direct = _get("https://en.wikipedia.org/api/rest_v1/page/summary/"
                  + urllib.parse.quote(q.replace(" ", "_")))
    if direct:
        try:
            sd = json.loads(direct)
            extract = (sd.get("extract") or "").strip()
            if extract:
                return f"Here's what I found: {extract[:600]}"
        except Exception:
            pass

    # Bounded, speakable failure â€” the pipeline keeps moving.
    if blocked:
        return ("I couldn't reach Wikipedia from here right now "
                "(the connection may be blocked). Try again in a moment, "
                "or give me a specific web address to open.")
    return (f"I couldn't find a clear result for '{q}' right now. "
            "Try a more specific topic, or give me a web address to open.")

# ---------------------------------------------------------------------------
# fetch_url -- fetch a specific web page and return its readable text
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", re.S | re.I)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

def _strip_html(raw: str) -> str:
    """Remove scripts/styles/tags, unescape entities, collapse whitespace."""
    t = _TAG_RE.sub(" ", raw)
    t = _TAG_STRIP_RE.sub(" ", t)
    t = _html.unescape(t)
    t = _WS_RE.sub(" ", t).strip()
    return t

def _fetch_url(url: str) -> str:
    """Fetch a webpage and return its readable text (truncated for speech)."""
    u = (url or "").strip()
    if not u:
        return "I need a web address to open. Which page?"
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    try:
        host = urllib.parse.urlparse(u).netloc.lower()
    except Exception:
        host = ""
    # Safety: only allow public hosts (never localhost / private ranges).
    if host in ("localhost", "127.0.0.1", "0.0.0.0") or host.startswith(("192.168.", "10.")):
        return "I can only open public web pages, not private or local ones."
    # Wikipedia: the plain-text extract API (2026-11-07) — the raw HTML's
    # first 700 chars were pure navigation ("Jump to content Main menu…"),
    # useless as a research source for write_website.
    m = re.match(r"https?://([a-z\-]+)\.(?:m\.)?wikipedia\.org/wiki/([^?#]+)", u)
    if m:
        api = (f"https://{m.group(1)}.wikipedia.org/w/api.php?action=query&prop=extracts"
               f"&explaintext=1&redirects=1&format=json&titles={m.group(2)}")
        body = _get(api)
        try:
            pages = (json.loads(body).get("query") or {}).get("pages") or {}
            text = " ".join((p.get("extract") or "") for p in pages.values())
            text = _WS_RE.sub(" ", text).strip()
            if text:
                return f"From {host}: {text[:_FETCH_CHARS]}"
        except Exception:
            pass
    body = _get(u, headers={"Accept": "text/html, text/plain, */*"}, timeout=TIMEOUT)
    if not body:
        return f"I couldn't load that page: {u}. It may be offline or blocking bots."
    text = _readable_text(body)
    if not text:
        return f"I loaded {u} but couldn't find readable text on it."
    return f"From {host}: {text[:_FETCH_CHARS]}"


# Enough text to research from (the LLM summarizes it); was 700.
_FETCH_CHARS = int(os.environ.get("WEB_FETCH_CHARS", "2500"))


def _readable_text(raw: str) -> str:
    """Main content, not chrome: <article>/<main> if present, else the
    page's <p> paragraphs, else the whole stripped page."""
    for tag in ("article", "main"):
        mm = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", raw, re.S | re.I)
        if mm:
            t = _strip_html(mm.group(1))
            if len(t) > 200:
                return t
    paras = [_strip_html(p) for p in re.findall(r"<p\b[^>]*>(.*?)</p>", raw, re.S | re.I)]
    paras = [p for p in paras if len(p) > 40]
    if paras:
        return " ".join(paras)
    return _strip_html(raw)

# ---------------------------------------------------------------------------
# check_weather -- current weather via Open-Meteo (no key, no bot-gate).
#   Geocoding + forecast are both free, keyless, and JSON. Location order:
#   (1) an explicit place name -> geocoded; (2) stored GPS from the dashboard
#   (~/.me3_location.json, written by the UI's /location endpoint, or "gps");
#   (3) a default home location (U.P. Diliman) so it always works.
# ---------------------------------------------------------------------------
_GPS_FILE = os.path.join(os.path.expanduser("~"), ".me3_location.json")
# Default home = U.P. Diliman (matches the ME2 tile). lat, lon, label.
_HOME_LOCATION = (14.6534, 121.0673, "U.P. Diliman")

# WMO weather-code -> short spoken condition. (Open-Meteo "weathercode" group.)
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "light freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "heavy rain showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with light hail", 99: "thunderstorm with heavy hail",
}

def _geocode(name: str):
    """Resolve a place name to (lat, lon, label) via Open-Meteo geocoding.
    Returns (None, None, None) on failure."""
    try:
        url = ("https://geocoding-api.open-meteo.com/v1/search?name="
               + urllib.parse.quote(name) + "&count=1&language=en&format=json")
        body = _get(url, timeout=TIMEOUT)
        if not body:
            return None, None, None
        data = json.loads(body)
        results = data.get("results") or []
        if not results:
            return None, None, None
        r = results[0]
        label = r.get("name") or name
        # skip a region equal to the place itself ("Tokyo, Tokyo")
        adm = [x for x in (r.get("admin2"), r.get("admin1")) if x and x != label]
        if adm:
            label = label + ", " + ", ".join(adm[:1])
        return r.get("latitude"), r.get("longitude"), label
    except Exception:
        return None, None, None

def _resolve_location(location: str):
    """Decide which (lat, lon, label) to fetch weather for. See the block
    comment above for the resolution order. Always returns a 3-tuple."""
    loc = (location or "").strip()
    # (2) explicit GPS request or a stored GPS file.
    if loc.lower() in ("gps", "here", "my location", "current location"):
        try:
            if os.path.exists(_GPS_FILE):
                with open(_GPS_FILE, encoding="utf-8") as f:
                    g = json.load(f)
                lat, lon = float(g["lat"]), float(g["lon"])
                return lat, lon, (g.get("name") or "your saved location")
        except Exception:
            pass
        # No saved GPS -> default home. Do NOT fall through to _geocode:
        # the geocoder resolves the literal word "here" to Hères, France.
        return _HOME_LOCATION
    # (1) an explicit place name.
    if loc:
        lat, lon, label = _geocode(loc)
        if lat is not None and lon is not None:
            return lat, lon, label
        # Unresolvable name: try to geocode as a plain city, else fail-soft home.
    # (3) default home.
    return _HOME_LOCATION

def _post_weather_card(place, temp, condition, code) -> None:
    """Show this lookup on the Boots UI WEATHER card (POST /weather).
    Fire-and-forget on a daemon thread: never slows or breaks the tool."""
    if temp is None:
        return
    import threading
    base = os.environ.get("ME3_BOOTSUI", "http://127.0.0.1:8330")
    body = json.dumps({"place": place, "temp": temp,
                       "condition": condition, "code": code}).encode()

    def _go():
        try:
            req = urllib.request.Request(base + "/weather", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1.5) as r:
                r.read()
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


def _check_weather(location: str = "") -> str:
    """Current weather for a place. Returns a speakable summary string."""
    lat, lon, name = _resolve_location(location)
    if lat is None or lon is None:
        return "I couldn't figure out that location. Try a city name, or say 'here' to use your saved location."
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               "&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
               "weather_code,wind_speed_10m,is_day&daily=temperature_2m_max,"
               "temperature_2m_min,precipitation_probability_max&timezone=auto&forecast_days=1")
        body = _get(url, timeout=TIMEOUT)
        if not body:
            return f"I couldn't reach the weather service for {name}. Please try again."
        data = json.loads(body)
        cur = data.get("current") or {}
        daily = data.get("daily") or {}
        t_now = cur.get("temperature_2m")
        feels = cur.get("apparent_temperature")
        code = cur.get("weather_code")
        cond = _WMO.get(code, "unremarkable")
        wind = cur.get("wind_speed_10m")
        parts = [f"It's {t_now} degrees in {name} right now."]
        if cond and cond != "unremarkable":
            parts.append(cond.capitalize() + ".")
        if feels is not None:
            parts.append(f"It feels like {feels} degrees.")
        if wind is not None:
            parts.append(f"Wind is about {wind} kilometres per hour.")
        tmax = (daily.get("temperature_2m_max") or [None])[0]
        tmin = (daily.get("temperature_2m_min") or [None])[0]
        pmax = (daily.get("precipitation_probability_max") or [None])[0]
        extras = []
        if tmax is not None and tmin is not None:
            extras.append(f"high {tmax} and low {tmin}")
        if pmax is not None:
            extras.append(f"{pmax} percent chance of rain")
        if extras:
            parts.append("Today: " + ", ".join(extras) + ".")
        _post_weather_card(name, t_now, cond if cond != "unremarkable" else "", code)
        return " ".join(parts)
    except Exception as e:
        return f"I hit a problem checking the weather ({e!r}). Please try again."

# ---------------------------------------------------------------------------
# Canonical tool specs (same shape as me2_tools.TOOL_SPECS)
# (name, description, params) -- params is the OpenAI JSON-schema "properties".
# ---------------------------------------------------------------------------
# Params the model may OMIT (not in the schema's `required`). check_weather
# without a place = U.P. Diliman; a required `location` made the model
# ask "where?" instead of just calling (2026-11-07).
OPTIONAL_PARAMS = {"check_weather": {"location"}, "write_website": {"hero_image"}}

TOOL_SPECS = [
    ("check_weather",
     "Get the CURRENT weather (temperature, conditions, wind). Use when the "
     "user asks about the weather, the temperature, or whether it will rain. "
     "If the user does NOT name a place, call this IMMEDIATELY with no "
     "location — it defaults to U.P. Diliman (the default home). Never ask where. "
     "Pass a location only when the user names a place (a city name).",
     {"location": {"type": "string"}}),
    ("web_search",
     "Search the web for information about a topic or question. Use this when "
     "the user asks a factual question, asks you to look something up, or wants "
     "to know about a person, place, thing, or event. Returns a short summary.",
     {"query": {"type": "string"}}),
    ("fetch_url",
     "Open a specific web page (a URL the user gives you) and read its text. "
     "Use when the user says 'open', 'read', or 'look at' a particular webpage.",
     {"url": {"type": "string"}}),
    ("write_website",
     "Build a modern website about a topic and serve it in a browser. Use when "
     "the user asks you to 'make', 'create', 'build', or 'write' a website/"
     "webpage/web page about something. RESEARCH FIRST: call web_search 2-3 "
     "times and fetch_url on 2+ of the best results, then write the content "
     "from what you actually read — every number, date, and name must come "
     "from a source you fetched, and those pages go in `sources` (never "
     "invent stats). YOU write the content: description = short lede (1-2 "
     "sentences); sections = 4-6 sections, each {heading, body (2-4 sentences "
     "of real prose), points (2-4 key points)}; a section may set "
     "kind='timeline' (points become 'Year — event' entries on a vertical "
     "spine) when the topic has a clear history. Set stats to 3-4 headline "
     "numbers {value, label} (real figures from your sources) for a stat band "
     "under the hero. PHOTOS: set hero_image and, for 2-3 sections, image to a "
     "short Wikipedia-style search phrase for a real photo (e.g. 'Taal Volcano "
     "2020 eruption', 'Tagaytay ridge'); the tool finds free-licence photos and "
     "credits them. A section may also set quote (one short real quote or fact "
     "line from your sources). The tool renders a themed page, verifies it renders, "
     "and returns the URL + verification status. Pick a hue (0-360) that fits "
     "the topic's mood (e.g. 195 ocean, 35 bakery, 265 space, 120 forest, 0 "
     "red-alert); leave empty to auto-pick. Set publish=true ONLY when the "
     "user explicitly asks to publish/host it publicly on GitHub.",
     {"topic": {"type": "string"},
      "title": {"type": "string"},
      "description": {"type": "string"},
      "sections": {"type": "array",
                   "items": {"type": "object",
                             "properties": {
                                 "heading": {"type": "string"},
                                 "body": {"type": "string"},
                                 "kind": {"type": "string",
                                          "enum": ["prose", "timeline"]},
                                 "points": {"type": "array",
                                            "items": {"type": "string"}},
                                 "image": {"type": "string"},
                                 "quote": {"type": "string"}},
                             "required": ["heading", "body"]}},
      "stats": {"type": "array",
                "items": {"type": "object",
                          "properties": {
                              "value": {"type": "string"},
                              "label": {"type": "string"}},
                          "required": ["value", "label"]}},
      "sources": {"type": "array",
                  "items": {"type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"}}}},
      "hero_image": {"type": "string"},
      "hue": {"type": "integer"},
      "publish": {"type": "boolean"}}),
]

def dispatch(tool: str, **kw) -> str:
    """Execute a web tool. Returns a string the LLM can speak back to the user."""
    try:
        if tool == "check_weather":
            return _check_weather(kw.get("location", ""))
        if tool == "web_search":
            return _web_search(kw.get("query", ""))
        if tool == "fetch_url":
            return _fetch_url(kw.get("url", ""))
        if tool == "write_website":
            return _write_website(kw.get("topic", ""), kw.get("title", ""),
                                  kw.get("description", ""),
                                  kw.get("hue"), kw.get("publish", False),
                                  sections=kw.get("sections"),
                                  sources=kw.get("sources"),
                                  stats=kw.get("stats"),
                                  hero_image=kw.get("hero_image"))
        return f"I don't know how to do '{tool}'."
    except Exception as e:
        # Fail-soft: never crash the agent on a web error.
        return f"Something went wrong while using the web ({e!r}). Please try again."

# ---------------------------------------------------------------------------
# write_website -- adaptive-theme site builder + optional GitHub Pages publish
# ---------------------------------------------------------------------------
import time as _time

SITES_DIR = os.environ.get("WEBSITES_DIR",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "bootsui", "sites"))
# Base URL the user can open the generated site from (Pi's LAN IP, UI port).
_SITE_PORT = os.environ.get("WEBSITE_PORT", "8330")


def _site_host() -> str:
    """WEBSITE_HOST if set, else the Pi's LAN address, detected per call
    (2026-11-07). The old fixed default 192.168.1.3 was stale (the Pi is on
    192.168.137.x now), and the default-route source is NOT the answer here:
    the route goes through the VPN (tun0 192.168.100.5), which a LAN browser
    can't reach. So: first IPv4 on a non-loopback, non-tunnel interface."""
    h = os.environ.get("WEBSITE_HOST", "").strip()
    if h:
        return h
    try:
        import subprocess
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True,
                             text=True, timeout=3).stdout
        for line in out.splitlines():
            parts = line.split()
            iface, addr = parts[1], parts[3].split("/")[0]
            if iface == "lo" or iface.startswith(("tun", "tap", "wg", "docker", "veth", "br-")):
                continue
            return addr
    except Exception:
        pass
    try:
        import socket
        return socket.gethostname() + ".local"   # mDNS (avahi runs on the Pi)
    except Exception:
        return "localhost"

# GitHub Pages publish (opt-in via publish=true).
_GH_REPO = os.environ.get("WEBSITE_GH_REPO", "jblagana/boots-sites")
_GH_PAGES = os.environ.get("WEBSITE_PAGES_URL",
                           "https://jblagana.github.io/boots-sites")
_GH_CACHE = os.environ.get("WEBSITE_GH_CACHE",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        ".gh_boots_sites"))

def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s or "site"

def _hue_for(topic: str, hue=None) -> int:
    """Pick the theme hue: explicit (validated) > deterministic hash of topic."""
    try:
        h = int(hue)
        if 0 <= h <= 360:
            return h % 360
    except (TypeError, ValueError):
        pass
    return sum(ord(c) for c in (topic or "site").lower()) % 360

def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ---- photos for write_website (2026-11-07) ---------------------------------
# The LLM gives a search phrase per photo (hero_image, section.image). We look it
# up on Wikipedia (PageImages, free-licence images only, no API key), download a
# standard-width thumbnail and EMBED it as a data: URI, so every site stays one
# self-contained file (the /site/ route serves only .html, and the GitHub
# publish pushes one file). No match / timeout -> that block renders without a
# photo. All lookups run in parallel under one overall deadline.
SITE_IMG_DEADLINE_S = float(os.environ.get("ME3_SITE_IMG_DEADLINE", "15"))
_WIKI_API = "https://en.wikipedia.org/w/api.php"


def _wiki_image(query: str, width: int = 960):
    """Search phrase -> {'data','page','file','w'} or None."""
    import base64
    q = (query or "").strip()
    if not q:
        return None
    params = {"action": "query", "format": "json", "formatversion": "2",
              "generator": "search", "gsrsearch": q, "gsrlimit": "4",
              "gsrnamespace": "0", "prop": "pageimages", "piprop": "thumbnail|name",
              "pithumbsize": str(width), "pilicense": "free"}
    try:
        req = urllib.request.Request(_WIKI_API + "?" + urllib.parse.urlencode(params),
                                     headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=6) as r:
            pages = (json.loads(r.read().decode("utf-8")).get("query") or {}).get("pages") or []
        pages.sort(key=lambda p: p.get("index", 99))
        for p in pages:
            th = p.get("thumbnail") or {}
            src = th.get("source", "")
            if not src or src.lower().endswith((".svg", ".svg.png")) or ".svg/" in src.lower():
                continue                       # logos/maps/diagrams make poor photos
            if th.get("width", 0) < 320:
                continue
            ireq = urllib.request.Request(src, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(ireq, timeout=8) as ir:
                ctype = ir.headers.get("Content-Type", "image/jpeg").split(";")[0]
                blob = ir.read(1_500_000)
            if not ctype.startswith("image/") or len(blob) < 2000:
                continue
            return {"data": f"data:{ctype};base64," + base64.b64encode(blob).decode(),
                    "page": p.get("title", ""), "file": p.get("pageimage", ""),
                    "w": th.get("width", width)}
    except Exception as e:
        print(f"[site-img] {q!r}: {e!r}", file=sys.stderr)
    return None


def _fetch_site_images(hero_q, section_qs):
    """Fetch the hero + section photos in parallel; never the same photo twice.
    Returns (hero, [img|None per section])."""
    from concurrent.futures import ThreadPoolExecutor, wait
    jobs = [("hero", hero_q, 1280)] + [(i, q, 960) for i, q in enumerate(section_qs)]
    jobs = [j for j in jobs if (j[1] or "").strip()]
    out = {}
    if jobs:
        ex = ThreadPoolExecutor(max_workers=min(6, len(jobs)))
        futs = {ex.submit(_wiki_image, q, w): k for (k, q, w) in jobs}
        done, _ = wait(futs, timeout=SITE_IMG_DEADLINE_S)
        for f in done:
            try:
                out[futs[f]] = f.result()
            except Exception:
                pass
        ex.shutdown(wait=False, cancel_futures=True)
    seen = set()

    def take(k):
        im = out.get(k)
        if not im or im["file"] in seen:
            return None
        seen.add(im["file"])
        return im
    hero = take("hero")
    return hero, [take(i) for i in range(len(section_qs))]


def _credit_html(im):
    f = im.get("file", "")
    return ('<span class="credit">Wikimedia Commons</span>' if f else "")


def _render_site(title: str, topic: str, description: str, hue: int,
                 sections=None, sources=None, stats=None, hero_image=None) -> str:
    """Modern single-file page (2026-11-07 template): floating pill nav, rounded
    photo hero, stat tiles, split sections with rounded photos, feature cards
    for points, a dark timeline panel with year cards, pull quotes, and source
    + image-credit cards. One hue drives the palette, light and dark.

    Content is LLM-authored: sections = [{heading, body, points[], kind?,
    image?, quote?}]; kind='timeline' renders points ('Year - event') as year
    cards; image/hero_image are search phrases resolved to Wikipedia photos.
    Empty sections fall back to a lede-only page (never blank)."""
    e = _esc
    desc = (description or "").strip() or (
        "A page about " + (topic or "this topic") +
        ", generated by the Boots voice agent on a Raspberry Pi.")

    secs = []
    for i, s in enumerate(sections or []):
        if not isinstance(s, dict):
            continue
        h = (s.get("heading") or "").strip()
        b = (s.get("body") or "").strip()
        pts = [p.strip() for p in (s.get("points") or [])
               if isinstance(p, str) and p.strip()][:8]
        kind = (s.get("kind") or "prose").strip().lower()
        if kind not in ("prose", "timeline"):
            kind = "prose"
        if not h and not b:
            continue
        secs.append({"id": "s" + str(len(secs) + 1), "heading": h or ("Section " + str(len(secs) + 1)),
                     "body": b, "points": pts, "kind": kind,
                     "image": (s.get("image") or "").strip() if isinstance(s.get("image"), str) else "",
                     "quote": (s.get("quote") or "").strip() if isinstance(s.get("quote"), str) else ""})
    if not secs:
        secs = [{"id": "s1", "heading": "About", "body": desc, "points": [], "kind": "prose",
                 "image": "", "quote": ""}]

    hero_q = (hero_image or "").strip() if isinstance(hero_image, str) else ""
    if not hero_q:
        hero_q = topic or title
    hero, imgs = _fetch_site_images(hero_q, [s["image"] for s in secs])
    for s, im in zip(secs, imgs):
        s["img"] = im

    stat_items = []
    for st in (stats or []):
        if isinstance(st, dict) and (st.get("value") or "").strip():
            stat_items.append({"value": st["value"].strip(), "label": (st.get("label") or "").strip()})
    stat_items = stat_items[:4]

    # ---- blocks ------------------------------------------------------------
    def pic(im, cap):
        if not im:
            return ""
        return (f'<figure class="pic"><img src="{im["data"]}" alt="{e(cap)}" loading="lazy">'
                f'<figcaption><span class="chip">{e(cap)}</span>{_credit_html(im)}</figcaption></figure>')

    def quote(s):
        return (f'<div class="quote"><q>{e(s["quote"])}</q></div>' if s["quote"] else "")

    def year_split(p):
        m = re.match(r"^\s*([0-9][0-9 ,.]{1,12}(?:\s?(?:BCE|BC|CE|AD|ka|Ma))?|c\.\s?\d+|\d+s)\s*[-–—:]\s*(.+)$", p)
        return (m.group(1).strip(), m.group(2).strip()) if m else ("", p)

    blocks = []
    n_split = 0
    for idx, s in enumerate(secs):
        head = (f'<span class="eyebrow">{e(topic)}</span>'
                if idx == 0 and topic and topic.strip().lower() != title.strip().lower() else "")
        body = f'<p class="lede">{e(s["body"])}</p>' if s["body"] else ""
        if s["kind"] == "timeline" and s["points"]:
            cards = []
            for j, p in enumerate(s["points"]):
                y, ev = year_split(p)
                cls = ' class="now"' if j == len(s["points"]) - 1 else ""
                cards.append(f'<li{cls}><b>{e(y or str(j + 1))}</b><span>{e(ev)}</span></li>')
            top_pic = pic(s["img"], s["heading"])
            blocks.append(
                f'<section class="sec" id="{s["id"]}"><div class="panel">'
                f'<div class="panel-top{" solo" if not top_pic else ""}"><div class="text">'
                f'<span class="eyebrow">Timeline</span><h2>{e(s["heading"])}</h2>{body}</div>{top_pic}</div>'
                f'<ol class="years">{"".join(cards)}</ol></div>{quote(s)}</section>')
        elif s["img"] and not s["points"] and idx == len(secs) - 1 and len(secs) > 2:
            im = s["img"]
            blocks.append(
                f'<section class="sec" id="{s["id"]}"><div class="wide">'
                f'<img src="{im["data"]}" alt="{e(s["heading"])}" loading="lazy">{_credit_html(im)}'
                f'<div class="over"><h2>{e(s["heading"])}</h2>'
                f'{"<p>" + e(s["body"]) + "</p>" if s["body"] else ""}</div></div>{quote(s)}</section>')
        elif s["img"]:
            flip = " flip" if n_split % 2 else ""
            n_split += 1
            pts = ("<ul class='ticks'>" + "".join(f"<li>{e(p)}</li>" for p in s["points"]) + "</ul>"
                   if s["points"] else "")
            blocks.append(
                f'<section class="sec" id="{s["id"]}"><div class="split{flip}"><div class="text">{head}'
                f'<h2>{e(s["heading"])}</h2>{body}{pts}</div>{pic(s["img"], s["heading"])}</div>'
                f'{quote(s)}</section>')
        else:
            feats = ""
            if s["points"]:
                feats = '<div class="features">' + "".join(
                    f'<div class="feat"><span class="n">{j + 1:02d}</span><p>{e(p)}</p></div>'
                    for j, p in enumerate(s["points"])) + "</div>"
            blocks.append(
                f'<section class="sec" id="{s["id"]}"><div class="head">{head}<h2>{e(s["heading"])}</h2>'
                f'{body}</div>{feats}{quote(s)}</section>')

    tiles = ""
    if stat_items:
        tiles = '<section class="tiles" aria-label="Key figures">' + "".join(
            f'<div class="tile{" accent" if j == 0 else ""}"><span class="v">{e(st["value"])}</span>'
            f'<span class="l">{e(st["label"])}</span></div>' for j, st in enumerate(stat_items)) + "</section>"

    srcs = [s for s in (sources or []) if isinstance(s, dict) and (s.get("url") or s.get("title"))][:10]

    def host(u):
        try:
            return urllib.parse.urlparse(u).netloc.replace("www.", "")
        except Exception:
            return ""
    src_cards = "".join(
        (f'<a class="src" href="{e(s["url"])}">' if s.get("url") else '<div class="src">')
        + f'<span class="ico">{e((host(s.get("url", "")) or "?")[:1].upper())}</span><span><span class="t">'
        f'{e(s.get("title") or s.get("url"))}</span><span class="d">{e(host(s.get("url", "")))}</span></span>'
        + ("</a>" if s.get("url") else "</div>") for s in srcs)
    credits = [im for im in [hero] + [s["img"] for s in secs] if im]
    cred_cards = "".join(
        f'<a class="src" href="https://commons.wikimedia.org/wiki/File:{urllib.parse.quote(im["file"])}">'
        f'<span class="ico">{j + 1}</span><span><span class="t">{e(im["file"].rsplit(".", 1)[0].replace("_", " "))}</span>'
        f'<span class="d">from the Wikipedia article “{e(im["page"])}”</span></span></a>'
        for j, im in enumerate(credits) if im.get("file"))
    refs = ""
    if src_cards or cred_cards:
        refs = ('<section class="sec" id="sources"><div class="head"><h2>Sources</h2></div>'
                + (f'<div class="cards">{src_cards}</div>' if src_cards else "")
                + (f'<h3 class="subhead">Image credits</h3><div class="cards">{cred_cards}</div>' if cred_cards else "")
                + "</section>")

    links = "".join(f'<a href="#{s["id"]}">{e(s["heading"][:28])}</a>' for s in secs[:5])
    if refs:
        links += '<a class="cta" href="#sources">Sources</a>'
    hero_img = (f'<img src="{hero["data"]}" alt="{e(title)}">' if hero else "")
    hero_credit = ('<span class="credit">Photo · Wikimedia Commons</span>' if hero else "")
    n_src = len(srcs)
    hero_eyebrow = ("" if (topic or "").strip().lower() == title.strip().lower()
                    else f'<span class="eyebrow">{e(topic or "Boots site")}</span>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@400..800&family=Geist+Mono:wght@400;500&display=swap">
<style>
:root {{
  --h: {hue};
  --bg: hsl(var(--h) 20% 98%); --card: hsl(0 0% 100%); --tint: hsl(var(--h) 60% 95%);
  --ink: hsl(var(--h) 18% 9%); --muted: hsl(var(--h) 8% 40%); --line: hsl(var(--h) 14% 89%);
  --accent: hsl(var(--h) 80% 46%); --accent-ink: hsl(var(--h) 80% 34%); --glow: hsl(var(--h) 90% 60% / .18);
  --panel: hsl(var(--h) 16% 9%); --panel-card: hsl(var(--h) 14% 14%); --panel-ink: hsl(var(--h) 12% 93%);
  --panel-muted: hsl(var(--h) 8% 66%); --panel-line: hsl(var(--h) 12% 22%); --panel-accent: hsl(var(--h) 90% 66%);
  --glass: hsl(0 0% 100% / .74);
  --shadow: 0 1px 2px hsl(var(--h) 30% 20% / .06), 0 8px 28px hsl(var(--h) 30% 20% / .07);
  --sans: "Geist", ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
  --mono: "Geist Mono", ui-monospace, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root {{
  --bg: hsl(var(--h) 14% 6%); --card: hsl(var(--h) 12% 10%); --tint: hsl(var(--h) 40% 13%);
  --ink: hsl(var(--h) 12% 94%); --muted: hsl(var(--h) 8% 66%); --line: hsl(var(--h) 10% 18%);
  --accent: hsl(var(--h) 85% 52%); --accent-ink: hsl(var(--h) 90% 70%); --glow: hsl(var(--h) 90% 60% / .14);
  --panel: hsl(var(--h) 14% 10%); --panel-card: hsl(var(--h) 12% 15%); --glass: hsl(var(--h) 14% 10% / .74);
  --shadow: 0 1px 2px hsl(0 0% 0% / .3), 0 8px 28px hsl(0 0% 0% / .35); color-scheme: dark; }} }}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 1rem; line-height: 1.6;
  -webkit-font-smoothing: antialiased; background-image: radial-gradient(60rem 30rem at 85% -10%, var(--glow), transparent 70%); background-repeat: no-repeat; }}
.wrap {{ max-width: 1180px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
a {{ color: var(--accent-ink); text-underline-offset: 3px; }}
:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 8px; }}
h1, h2, h3 {{ margin: 0; text-wrap: balance; letter-spacing: -.035em; line-height: 1.05; font-weight: 700; }}
p {{ margin: 0; }}
.eyebrow {{ display: inline-flex; align-items: center; gap: 8px; font-family: var(--mono); font-size: .75rem; letter-spacing: .04em; color: var(--accent-ink); }}
.eyebrow::before {{ content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 4px var(--glow); }}
.chip {{ display: inline-flex; padding: 6px 12px; border-radius: 999px; font-size: .8rem; font-weight: 500; background: var(--glass); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); color: var(--ink); }}
.credit {{ font-family: var(--mono); font-size: .7rem; color: hsl(0 0% 100% / .85); background: hsl(0 0% 0% / .35); padding: 3px 8px; border-radius: 6px; }}
.nav {{ position: sticky; top: 0; z-index: 5; padding-block: 12px; }}
.nav-in {{ display: flex; align-items: center; gap: 16px; padding: 8px 8px 8px 18px; border-radius: 999px; background: var(--glass); backdrop-filter: blur(16px) saturate(1.4); -webkit-backdrop-filter: blur(16px) saturate(1.4); border: 1px solid var(--line); box-shadow: var(--shadow); }}
.brand {{ font-weight: 700; letter-spacing: -.02em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }}
.links {{ display: flex; gap: 4px; margin-left: auto; overflow-x: auto; scrollbar-width: none; min-width: 0; }}
.links a {{ color: var(--muted); text-decoration: none; font-size: .88rem; padding: 7px 12px; border-radius: 999px; white-space: nowrap; }}
.links a:hover {{ background: var(--tint); color: var(--ink); }}
.links a.cta {{ background: var(--ink); color: var(--bg); font-weight: 500; }}
@media (max-width: 640px) {{ .links a:not(.cta) {{ display: none; }} }}
.hero {{ position: relative; border-radius: 28px; overflow: hidden; min-height: 520px; display: flex; align-items: flex-end; isolation: isolate; box-shadow: var(--shadow);
  background: radial-gradient(80% 120% at 80% 0%, hsl(var(--h) 85% 55%), hsl(var(--h) 40% 14%) 70%); }}
.hero img {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; z-index: -2; }}
.hero::after {{ content: ""; position: absolute; inset: 0; z-index: -1; background: linear-gradient(180deg, hsl(var(--h) 30% 5% / .05) 20%, hsl(var(--h) 30% 5% / .8) 100%); }}
.hero-in {{ width: 100%; padding: clamp(24px, 5vw, 56px); display: grid; gap: 16px; color: #fff; }}
.hero h1 {{ font-size: clamp(2.6rem, 8vw, 6.4rem); font-weight: 800; letter-spacing: -.05em; line-height: .94; max-width: 16ch; }}
.hero p {{ max-width: 52ch; font-size: clamp(1.05rem, 1.6vw, 1.22rem); line-height: 1.5; color: hsl(0 0% 100% / .88); }}
.hero .eyebrow {{ color: hsl(0 0% 100% / .9); }}
.hero .credit {{ justify-self: end; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 150px), 1fr)); gap: 14px; padding-block: 14px 0; }}
.tile {{ background: var(--card); border: 1px solid var(--line); border-radius: 20px; padding: 20px; display: grid; gap: 12px; align-content: space-between; min-height: 130px; box-shadow: var(--shadow); min-width: 0; }}
.tile .v {{ font-size: clamp(1.9rem, 3.6vw, 2.8rem); font-weight: 700; letter-spacing: -.05em; line-height: 1; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }}
.tile .l {{ font-size: .9rem; color: var(--muted); line-height: 1.4; }}
.tile.accent {{ background: var(--accent); border-color: transparent; color: #fff; }}
.tile.accent .l {{ color: hsl(0 0% 100% / .88); }}
.sec {{ padding-block: clamp(52px, 8vw, 92px) 0; scroll-margin-top: 90px; }}
.sec h2 {{ font-size: clamp(1.9rem, 4.2vw, 3rem); }}
.lede {{ font-size: 1.1rem; color: var(--muted); max-width: 60ch; }}
.text, .head {{ display: grid; gap: 16px; min-width: 0; }}
.head {{ max-width: 780px; }}
.split {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.1fr); gap: clamp(28px, 5vw, 64px); align-items: center; }}
.split.flip > .text {{ order: 2; }}
.pic {{ margin: 0; position: relative; border-radius: 28px; overflow: hidden; box-shadow: var(--shadow); background: var(--tint); min-width: 0; }}
.pic img {{ display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: cover; transition: transform .8s cubic-bezier(.2,.7,.2,1); }}
.pic:hover img {{ transform: scale(1.03); }}
.pic figcaption {{ position: absolute; left: 14px; right: 14px; bottom: 14px; display: flex; justify-content: space-between; align-items: end; gap: 10px; flex-wrap: wrap; }}
.ticks {{ list-style: none; margin: 4px 0 0; padding: 0; display: grid; gap: 8px; }}
.ticks li {{ padding-left: 26px; position: relative; color: var(--ink); }}
.ticks li::before {{ content: ""; position: absolute; left: 0; top: .45em; width: 14px; height: 14px; border-radius: 50%; background: var(--tint); box-shadow: inset 0 0 0 4px var(--accent); }}
@media (max-width: 820px) {{ .split {{ grid-template-columns: minmax(0, 1fr); }} .split.flip > .text {{ order: 0; }} }}
.features {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr)); gap: 14px; margin-top: 28px; }}
.feat {{ background: var(--card); border: 1px solid var(--line); border-radius: 20px; padding: 22px; display: grid; gap: 12px; align-content: start; box-shadow: var(--shadow); min-width: 0; }}
.feat .n {{ width: 36px; height: 36px; border-radius: 10px; display: grid; place-items: center; background: var(--tint); color: var(--accent-ink); font-family: var(--mono); font-size: .8rem; font-weight: 500; }}
.feat p {{ font-size: 1rem; }}
.quote {{ margin-top: 18px; border-radius: 20px; padding: clamp(22px, 4vw, 38px); background: var(--tint); }}
.quote q {{ font-size: clamp(1.3rem, 2.8vw, 2rem); font-weight: 600; letter-spacing: -.03em; line-height: 1.22; }}
.panel {{ background: var(--panel); color: var(--panel-ink); border-radius: 28px; padding: clamp(24px, 5vw, 56px); display: grid; gap: 30px; overflow: hidden; position: relative; isolation: isolate; }}
.panel::before {{ content: ""; position: absolute; inset: auto -20% -60% 30%; height: 120%; z-index: -1; background: radial-gradient(closest-side, hsl(var(--h) 90% 55% / .22), transparent); }}
.panel .eyebrow {{ color: var(--panel-accent); }}
.panel .lede {{ color: var(--panel-muted); }}
.panel-top {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, .9fr); gap: clamp(24px, 4vw, 56px); align-items: center; }}
.panel-top.solo {{ grid-template-columns: minmax(0, 1fr); }}
.panel .pic {{ box-shadow: none; background: var(--panel-card); }}
.years {{ list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 170px), 1fr)); gap: 12px; }}
.years li {{ background: var(--panel-card); border: 1px solid var(--panel-line); border-radius: 12px; padding: 16px; display: grid; gap: 6px; align-content: start; min-width: 0; }}
.years b {{ font-size: 1.5rem; letter-spacing: -.04em; font-variant-numeric: tabular-nums; color: var(--panel-accent); line-height: 1.05; overflow-wrap: anywhere; }}
.years span {{ font-size: .9rem; color: var(--panel-muted); line-height: 1.4; }}
.years li.now {{ background: hsl(var(--h) 80% 46%); border-color: transparent; }}
.years li.now b, .years li.now span {{ color: #fff; }}
@media (max-width: 900px) {{ .panel-top {{ grid-template-columns: minmax(0, 1fr); }} }}
.wide {{ position: relative; border-radius: 28px; overflow: hidden; min-height: 440px; display: flex; align-items: flex-end; box-shadow: var(--shadow); background: var(--tint); isolation: isolate; }}
.wide img {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; z-index: -1; }}
.wide .credit {{ position: absolute; right: 14px; top: 14px; }}
.over {{ margin: clamp(14px, 3vw, 28px); max-width: 540px; padding: 26px; border-radius: 20px; background: var(--glass); backdrop-filter: blur(18px) saturate(1.3); -webkit-backdrop-filter: blur(18px) saturate(1.3); display: grid; gap: 12px; }}
.over h2 {{ font-size: clamp(1.7rem, 3.4vw, 2.5rem); }}
.over p {{ color: var(--muted); }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 260px), 1fr)); gap: 12px; margin-top: 22px; }}
.src {{ display: grid; grid-template-columns: 40px minmax(0, 1fr); gap: 12px; align-items: center; padding: 14px; border-radius: 12px; background: var(--card); border: 1px solid var(--line); text-decoration: none; color: var(--ink); }}
a.src:hover {{ border-color: var(--accent); }}
.src .ico {{ width: 40px; height: 40px; border-radius: 10px; background: var(--tint); color: var(--accent-ink); display: grid; place-items: center; font-weight: 700; font-size: .9rem; }}
.src .t {{ display: block; font-weight: 600; font-size: .92rem; line-height: 1.3; overflow-wrap: anywhere; }}
.src .d {{ display: block; font-family: var(--mono); font-size: .72rem; color: var(--muted); overflow-wrap: anywhere; }}
.subhead {{ font-size: 1.1rem; letter-spacing: -.02em; margin-top: 34px; }}
footer {{ margin-top: clamp(52px, 8vw, 92px); border-top: 1px solid var(--line); }}
footer .wrap {{ padding-block: 22px 40px; display: flex; justify-content: space-between; gap: 8px 24px; flex-wrap: wrap; font-size: .82rem; color: var(--muted); }}
.dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: hsl(145 60% 45%); margin-right: 8px; vertical-align: middle; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} html {{ scroll-behavior: auto; }} }}
@media (prefers-reduced-motion: no-preference) {{ .hero img {{ animation: settle 1.8s cubic-bezier(.2,.7,.2,1) both; }} @keyframes settle {{ from {{ transform: scale(1.06); }} to {{ transform: scale(1); }} }} }}
</style>
</head>
<body>
<div class="wrap nav"><nav class="nav-in" aria-label="Sections"><span class="brand">{e(title)}</span><div class="links">{links}</div></nav></div>
<main class="wrap">
<header class="hero">{hero_img}<div class="hero-in">{hero_eyebrow}<h1>{e(title)}</h1><p>{e(desc)}</p>{hero_credit}</div></header>
{tiles}
{"".join(blocks)}
{refs}
</main>
<footer><div class="wrap"><span><span class="dot"></span>Built by Boots on a Raspberry Pi{" · researched from " + str(n_src) + " source" + ("s" if n_src != 1 else "") if n_src else ""}</span><span>{e(_time.strftime('%Y-%m-%d %H:%M'))}</span></div></footer>
</body>
</html>"""

def _gh_publish(slug: str, html: str) -> str:
    """Opt-in: commit the page to the GitHub Pages repo and push.

    Returns a spoken status line. Fail-soft: any git/network error becomes a
    spoken message, never an exception. Uses the machine's git credential
    manager for auth (no token in code)."""
    import subprocess, shutil
    if not shutil.which("git"):
        return "I couldn't publish it publicly (git isn't available on this machine)."

    def _git(*args, cwd=None, timeout=90):
        return subprocess.run(["git", *args], cwd=cwd, timeout=timeout,
                              capture_output=True, text=True)

    try:
        if not os.path.isdir(os.path.join(_GH_CACHE, ".git")):
            r = _git("clone", f"https://github.com/{_GH_REPO}.git", _GH_CACHE)
            if r.returncode != 0:
                return ("I couldn't reach the GitHub publish repo "
                        f"({_GH_REPO}) to publish the page.")
        r = _git("-C", _GH_CACHE, "pull", "--rebase", "origin", "HEAD")
        # pull may fail on a brand-new empty repo; that's fine, keep going.
        with open(os.path.join(_GH_CACHE, slug + ".html"), "w", encoding="utf-8") as f:
            f.write(html)
        r = _git("-C", _GH_CACHE, "add", slug + ".html")
        r = _git("-C", _GH_CACHE, "commit", "-m", f"Boots: {slug}")
        if r.returncode != 0 and "nothing to commit" not in (r.stdout + r.stderr):
            return f"I hit a problem publishing the page ({(r.stderr or r.stdout).strip()[:120]})."
        r = _git("-C", _GH_CACHE, "push", "origin", "HEAD")
        if r.returncode != 0:
            return (f"I couldn't push the page to GitHub ({(r.stderr or r.stdout).strip()[:120]}). "
                    "It's still live on the LAN.")
        return (f"It's also been published to GitHub Pages at "
                f"{_GH_PAGES}/{slug}.html - it goes live in a couple of minutes.")
    except subprocess.TimeoutExpired:
        return "Publishing to GitHub took too long, so I stopped. It's still live on the LAN."
    except Exception as e:
        return f"Publishing to GitHub failed ({e!r}). It's still live on the LAN."

def _verify_site(url: str, html: str) -> str:
    """Tiered render verification. Returns an HONEST status string — the
    label always says exactly which tier ran, never more.

    T1  (always)  : HTTP 200 + byte size + <title> present (urllib).
    T1.5 (always) : static sanity — every local href/src in the page resolves
                    to a file on disk, and <section> tags are balanced.
    T2  (auto)    : headless Chromium (playwright) if importable — renders the
                    page, checks for JS errors, asserts the title is in the
                    DOM. If playwright is absent the result says so, it does
                    NOT pretend a render happened."""
    import urllib.request
    label = "http-only"
    problems = []
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            served = r.read()
            code = r.status
        if code != 200:
            return f"verification FAILED: server returned HTTP {code} for {url}"
        title_ok = "<title>" in served.decode("utf-8", "replace")
        label = f"http-200 ({len(served)} bytes, title {'ok' if title_ok else 'MISSING'})"
        if not title_ok:
            problems.append("no <title> in served page")
    except Exception as e:
        return f"verification FAILED: could not fetch {url} ({e!r})"

    # T1.5 static sanity (runs on the generated HTML itself)
    import re as _re
    secs_open = len(_re.findall(r"<section", html))
    secs_close = len(_re.findall(r"</section>", html))
    if secs_open != secs_close:
        problems.append(f"unbalanced <section> tags ({secs_open} open / {secs_close} close)")
    for ref in _re.findall(r'(?:src|href)="([^"]+)"', html):
        if ref.startswith(("http", "#", "mailto:", "data:")):
            continue
        if not os.path.exists(os.path.join(SITES_DIR, ref.lstrip("/"))):
            problems.append(f"missing local asset: {ref}")
    label += " + static-sanity"

    # T2 headless render (auto-detected; honest when absent)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page()
            js_errors = []
            pg.on("pageerror", lambda e: js_errors.append(str(e)))
            pg.goto(url, wait_until="networkidle", timeout=20000)
            dom_title = pg.title()
            b.close()
        label += f" + headless-render (dom title: {dom_title[:40]!r})"
        if js_errors:
            problems.append("JS errors: " + "; ".join(js_errors[:3]))
    except ImportError:
        label += " [playwright absent — no render check]"
    except Exception as e:
        problems.append(f"headless render check errored ({e!r})")

    if problems:
        return f"verification WARNINGS: {label} — " + "; ".join(problems)
    return f"verified: {label}"

def _write_website(topic: str, title: str, description: str,
                   hue=None, publish=False, sections=None, sources=None,
                   stats=None, hero_image=None) -> str:
    """Generate an adaptive-theme page, save to SITES_DIR, VERIFY the render,
    optionally publish. Content is LLM-authored (sections/stats/sources); the
    tool renders, serves, and checks. Returns a spoken message with the LAN
    URL, the honest verification status, and (when publish is true) the
    GitHub Pages URL."""
    topic = (topic or "").strip()
    title = (title or "").strip() or topic or "Untitled page"
    description = (description or "").strip()
    if not topic and not description:
        return "Tell me what the website should be about and I'll build it."

    h = _hue_for(topic, hue)
    slug = _slugify(title or topic) + "-" + str(int(_time.time() * 1000) % 100000)
    os.makedirs(SITES_DIR, exist_ok=True)
    path = os.path.join(SITES_DIR, slug + ".html")

    html = _render_site(title, topic, description, h,
                        sections=sections, sources=sources, stats=stats,
                        hero_image=hero_image)
    n_img = html.count('src="data:image/')
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        return f"I couldn't save the website file ({e!r})."

    url = f"http://{_site_host()}:{_SITE_PORT}/site/{slug}.html"
    # Verify through loopback: the server is on this machine, and the LAN
    # address can be unreachable from itself (or wrong, like the old default).
    verify = _verify_site(f"http://127.0.0.1:{_SITE_PORT}/site/{slug}.html", html)
    n_sec = len([s for s in (sections or [])
                 if isinstance(s, dict) and ((s.get("heading") or "").strip()
                                             or (s.get("body") or "").strip())])
    msg = (f"Done! I built a website about {topic or title} with {n_sec} "
           f"{'section' if n_sec == 1 else 'sections'}"
           + (f" and {n_img} photo{'s' if n_img != 1 else ''}" if n_img else "")
           + " and saved it. "
           f"You can open it in your browser at {url} "
           f"({verify}).")
    if publish:
        msg += " " + _gh_publish(slug, html)
    return msg

# NOTE: the standalone FastMCP server that used to live here was removed -
# the canonical MCP server is mcp_server.py (one process, all tools).
# This module is consumed via its TOOL_SPECS + dispatch() by both the agent
# (direct function-calling, D11) and mcp_server.py (MCP facade).
