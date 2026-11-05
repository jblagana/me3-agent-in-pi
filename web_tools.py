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
            return f"Here's what I found about {query}: {answer[:500]}"
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

    # 1) Find candidate article titles + URLs via MediaWiki opensearch.
    api = ("https://en.wikipedia.org/w/api.php?action=opensearch&search="
           + urllib.parse.quote(q)
           + "&limit=5&namespace=0&format=json&redirects=resolve")
    body = _get(api)
    candidates = []  # list of (title, url)
    if not body:
        blocked = True
    else:
        try:
            data = json.loads(body)
            titles = data[1] if len(data) > 1 else []
            urls = data[3] if len(data) > 3 else []
            for i, t in enumerate(titles):
                u = urls[i] if i < len(urls) else ""
                candidates.append((t, u))
        except Exception:
            blocked = True

    # 2) Fetch the top candidate's REST summary (clean plain-text extract).
    if candidates:
        title, _url = candidates[0]
        api_title = urllib.parse.quote(title.replace(" ", "_"))
        summ = _get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{api_title}")
        if summ:
            try:
                sd = json.loads(summ)
                extract = (sd.get("extract") or "").strip()
                if extract:
                    extract = extract[:600]
                    return (f"Here's what I found about {title}: {extract}")
            except Exception:
                pass

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
    body = _get(u, headers={"Accept": "text/html, text/plain, */*"}, timeout=TIMEOUT)
    if not body:
        return f"I couldn't load that page: {u}. It may be offline or blocking bots."
    text = _strip_html(body)
    if not text:
        return f"I loaded {u} but couldn't find readable text on it."
    snippet = text[:700]
    return f"From {host}: {snippet}"

# ---------------------------------------------------------------------------
# check_weather -- current weather via Open-Meteo (no key, no bot-gate).
#   Geocoding + forecast are both free, keyless, and JSON. Location order:
#   (1) an explicit place name -> geocoded; (2) stored GPS from the dashboard
#   (~/.me3_location.json, written by the UI's /location endpoint, or "gps");
#   (3) a default home location (UP Diliman) so it always works.
# ---------------------------------------------------------------------------
_GPS_FILE = os.path.join(os.path.expanduser("~"), ".me3_location.json")
# Default home = UP Diliman (matches the ME2 tile). lat, lon, label.
_HOME_LOCATION = (14.6534, 121.0673, "UP Diliman")

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
        adm = [x for x in (r.get("admin2"), r.get("admin1")) if x]
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
        return " ".join(parts)
    except Exception as e:
        return f"I hit a problem checking the weather ({e!r}). Please try again."

# ---------------------------------------------------------------------------
# Canonical tool specs (same shape as me2_tools.TOOL_SPECS)
# (name, description, params) -- params is the OpenAI JSON-schema "properties".
# ---------------------------------------------------------------------------
TOOL_SPECS = [
    ("check_weather",
     "Get the CURRENT weather (temperature, conditions, wind) for a place. Use "
     "when the user asks about the weather, the temperature, or whether it will "
     "rain. Pass a location only when the user names a place (a city name); "
     "leave it empty otherwise — never guess or pass 'here'.",
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
     "under the hero. The tool renders a themed page, verifies it renders, "
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
                                            "items": {"type": "string"}}},
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
                                  stats=kw.get("stats"))
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
_SITE_HOST = os.environ.get("WEBSITE_HOST", "192.168.1.3")
_SITE_PORT = os.environ.get("WEBSITE_PORT", "8330")

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

def _render_site(title: str, topic: str, description: str, hue: int,
                 sections=None, sources=None, stats=None) -> str:
    """Modern single-file page: one hue drives the whole palette (HSL vars).

    Content is LLM-authored: `sections` is a list of {heading, body,
    points[], kind?} where kind='timeline' renders points as a vertical
    spine of 'Year — event' entries; `stats` is a list of {value, label}
    rendered as a stat band under the hero; `sources` a list of {title,
    url}. When sections are empty the page falls back to a lede-only
    layout (never a blank page)."""
    desc = (description or "").strip() or (
        "A page about " + (topic or "this topic") +
        ", generated by the Boots voice agent on a Raspberry Pi.")

    # --- normalize sections -------------------------------------------------
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
        secs.append({"id": "s" + str(i + 1),
                     "heading": h or ("Section " + str(i + 1)),
                     "body": b, "points": pts, "kind": kind})

    # --- normalize stat band ------------------------------------------------
    stat_items = []
    for st in (stats or []):
        if not isinstance(st, dict):
            continue
        v = (st.get("value") or "").strip()
        l = (st.get("label") or "").strip()
        if v:
            stat_items.append({"value": v, "label": l})
    stat_items = stat_items[:4]
    if stat_items:
        stats_html = (
            '  <section class="statband" id="stats">\n'
            + "\n".join(
                f'    <div class="stat"><span class="stat-v">{_esc(s["value"])}</span>'
                f'<span class="stat-l">{_esc(s["label"])}</span></div>'
                for s in stat_items)
            + "\n  </section>")
    else:
        stats_html = ""

    toc = "\n".join(
        f'      <a href="#{s["id"]}">{_esc(s["heading"])}</a>' for s in secs)
    toc_nav = (f'<nav class="toc"><span class="toc-label">Outline</span>'
               f'<div class="toc-links">{toc}</div></nav>'
               if secs else "")

    if secs:
        cards = []
        for s in secs:
            body = (f'\n        <p>{_esc(s["body"])}</p>' if s["body"] else "")
            if s["kind"] == "timeline" and s["points"]:
                items = "\n".join(
                    f'          <li><span class="tl-dot"></span>{_esc(p)}</li>'
                    for p in s["points"])
                extra = (f'\n        <ol class="timeline">\n{items}\n        </ol>')
            elif s["points"]:
                extra = ("\n        <ul>\n"
                         + "\n".join(f"          <li>{_esc(p)}</li>"
                                     for p in s["points"])
                         + "\n        </ul>")
            else:
                extra = ""
            cards.append(f'    <section class="sec" id="{s["id"]}">'
                         f'\n      <h2>{_esc(s["heading"])}</h2>'
                         f"{body}{extra}\n    </section>")
        body_html = "\n".join(cards)
    else:
        body_html = (f'    <section class="sec" id="s1">'
                     f'\n      <h2>About</h2>\n      <p>{_esc(desc)}</p>'
                     f'\n    </section>')

    srcs = [s for s in (sources or [])
            if isinstance(s, dict) and (s.get("url") or s.get("title"))][:10]
    if srcs:
        def _src_li(s):
            if s.get("url"):
                label = _esc(s.get("title") or s["url"])
                return '      <li><a href="' + _esc(s["url"]) + '">' + label + "</a></li>"
            return "      <li>" + _esc(s.get("title") or "") + "</li>"
        items = "\n".join(_src_li(s) for s in srcs)
        sources_html = (f'  <section class="sec sources" id="sources">'
                        f'\n    <h2>Sources</h2>\n    <ol>\n{items}\n    </ol>\n  </section>')
    else:
        sources_html = ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
:root {{
  --h: {hue};
  --bg: hsl(var(--h) 25% 7%);
  --surface: hsl(var(--h) 20% 11%);
  --line: hsl(var(--h) 30% 24%);
  --ink: hsl(var(--h) 15% 92%);
  --muted: hsl(var(--h) 12% 62%);
  --accent: hsl(var(--h) 85% 62%);
  --accent-soft: hsl(var(--h) 85% 62% / .14);
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  margin: 0; background: var(--bg); color: var(--ink); line-height: 1.65;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}}
.hero {{
  position: relative; overflow: hidden; text-align: center;
  padding: 96px 24px 72px; border-bottom: 1px solid var(--line);
}}
.hero::before {{
  content: ""; position: absolute; inset: -40% -10% auto; height: 160%;
  background: radial-gradient(ellipse at 50% 0%,
              hsl(var(--h) 85% 60% / .22), transparent 60%);
  pointer-events: none;
}}
.eyebrow {{
  display: inline-block; font-size: .8rem; letter-spacing: .18em;
  text-transform: uppercase; color: var(--accent);
  border: 1px solid var(--line); border-radius: 999px;
  padding: 6px 14px; margin-bottom: 22px; background: var(--accent-soft);
}}
.hero h1 {{
  margin: 0 auto 14px; max-width: 18ch; font-size: clamp(2.4rem, 6vw, 4.2rem);
  line-height: 1.08; letter-spacing: -0.025em; font-weight: 750;
}}
.hero p {{
  margin: 0 auto; max-width: 56ch; color: var(--muted); font-size: 1.05rem;
}}
.hero .rule {{
  width: 72px; height: 3px; margin: 30px auto 0; border-radius: 2px;
  background: linear-gradient(90deg, var(--accent), transparent);
}}
main {{ max-width: 880px; margin: 0 auto; padding: 48px 20px 24px; }}
.grid {{
  display: grid; gap: 18px; grid-template-columns: 1fr;
}}
@media (min-width: 720px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
.card {{
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 14px; padding: 26px;
}}
.card h2 {{
  margin: 0 0 10px; font-size: 1.12rem; letter-spacing: -0.01em;
  color: var(--accent);
}}
.card p, .card li {{ color: var(--muted); }}
.card ul {{ margin: 0; padding-left: 18px; }}
.card li {{ margin-bottom: 6px; }}
main {{
  max-width: 720px; margin: 0 auto; padding: 36px 24px 8px;
}}
.toc {{
  position: sticky; top: 0; z-index: 5;
  display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center;
  padding: 12px 0; margin-bottom: 26px;
  background: color-mix(in srgb, var(--bg) 88%, transparent);
  backdrop-filter: blur(6px);
  border-bottom: 1px solid var(--line);
}}
.toc-label {{
  font-size: .72rem; letter-spacing: .16em; text-transform: uppercase;
  color: var(--muted); margin-right: 6px;
}}
.toc-links {{ display: flex; flex-wrap: wrap; gap: 4px 14px; }}
.toc-links a {{
  color: var(--muted); text-decoration: none; font-size: .88rem;
  padding: 2px 0; border-bottom: 1px solid transparent;
}}
.toc-links a:hover {{ color: var(--accent); border-bottom-color: var(--accent); }}
.sec {{
  margin-bottom: 34px; scroll-margin-top: 64px;
}}
.sec h2 {{
  margin: 0 0 10px; font-size: 1.35rem; letter-spacing: -0.015em;
  color: var(--ink);
}}
.sec p {{ color: var(--muted); margin: 0 0 12px; }}
.sec ul {{
  margin: 0; padding: 0; list-style: none;
  display: grid; gap: 8px;
}}
.sec li {{
  color: var(--muted); font-size: .95rem;
  padding: 10px 14px; border: 1px solid var(--line); border-radius: 10px;
  background: var(--surface);
}}
.sec li::before {{
  content: ""; display: inline-block; width: 7px; height: 7px;
  border-radius: 50%; background: var(--accent);
  margin-right: 10px; vertical-align: middle;
}}
.sources ol {{ padding-left: 20px; }}
.sources li {{
  background: none; border: none; padding: 0; margin-bottom: 6px;
  font-size: .88rem;
}}
.sources li::before {{ display: none; }}
.statband {{
  display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  margin: 8px 0 40px;
}}
.stat {{
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 14px; padding: 20px; text-align: center;
}}
.stat-v {{
  display: block; font-size: 1.9rem; font-weight: 750; letter-spacing: -0.02em;
  color: var(--accent);
}}
.stat-l {{
  display: block; margin-top: 4px; font-size: .82rem; color: var(--muted);
}}
.timeline {{
  list-style: none; margin: 14px 0 0; padding: 0 0 0 20px;
  border-left: 2px solid var(--line);
}}
.timeline li {{
  position: relative; padding: 0 0 14px 14px; color: var(--muted);
}}
.timeline li:last-child {{ padding-bottom: 0; }}
.tl-dot {{
  position: absolute; left: -26px; top: 6px; width: 10px; height: 10px;
  border-radius: 50%; background: var(--accent);
  box-shadow: 0 0 0 4px var(--accent-soft);
}}
footer {{
  text-align: center; padding: 40px 20px 56px; color: var(--muted);
  font-size: .85rem; border-top: 1px solid var(--line); margin-top: 32px;
}}
a {{ color: var(--accent); }}
</style>
</head>
<body>
  <header class="hero">
    <span class="eyebrow">{_esc(topic or "Boots site")}</span>
    <h1>{_esc(title)}</h1>
    <p>{_esc(desc)}</p>
    <div class="rule"></div>
  </header>
  <main>
{toc_nav}
{stats_html}
{body_html}
{sources_html}
  </main>
  <footer>Generated by Boots &middot; Raspberry Pi &middot; {_esc(_time.strftime('%Y-%m-%d %H:%M'))}</footer>
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
                   stats=None) -> str:
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
                        sections=sections, sources=sources, stats=stats)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        return f"I couldn't save the website file ({e!r})."

    url = f"http://{_SITE_HOST}:{_SITE_PORT}/site/{slug}.html"
    verify = _verify_site(url, html)
    n_sec = len([s for s in (sections or [])
                 if isinstance(s, dict) and ((s.get("heading") or "").strip()
                                             or (s.get("body") or "").strip())])
    msg = (f"Done! I built a website about {topic or title} with {n_sec} "
           f"{'section' if n_sec == 1 else 'sections'} and saved it. "
           f"You can open it in your browser at {url} "
           f"({verify}).")
    if publish:
        msg += " " + _gh_publish(slug, html)
    return msg

# NOTE: the standalone FastMCP server that used to live here was removed -
# the canonical MCP server is mcp_server.py (one process, all tools).
# This module is consumed via its TOOL_SPECS + dispatch() by both the agent
# (direct function-calling, D11) and mcp_server.py (MCP facade).
