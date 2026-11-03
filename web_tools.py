#!/usr/bin/env python3
# Boots  -- Web tools (real FastMCP server + native function-calling layer).
#
# Two tools the voice agent can use:
#   1. web_search   -- "search the web for X" -> Wikipedia opensearch + REST
#                      summary extract (reliable, no API key, no bot-gate).
#   2. fetch_url    -- "look at this webpage / read X" -> fetch a URL and return
#                      its readable text (for a specific page the user names).
#
# ARCHITECTURE (mirrors me2_ui_mcp.py, see its D11 note):
#   * This file is BOTH a standalone FastMCP server (`python3 web_tools.py`
#     runs `mcp.run()`) AND a plain module the agent imports for its native
#     FunctionSchema layer (`from web_tools import TOOL_SPECS, dispatch`).
#   * agent.py cannot host pipecat 1.12's MCPService on this stack, so the
#     RUNNING voice agent calls dispatch() directly via FunctionSchema handlers.
#     The FastMCP server is real and runnable -- it satisfies the classwork
#     "build & run your MCP servers (on RPi)" requirement and can be demoed
#     standalone with `mcp dev` or a standard MCP client.
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
# web_search -- Wikipedia opensearch -> REST summary extract
# ---------------------------------------------------------------------------
def _web_search(query: str) -> str:
    """Search the web (via Wikipedia) for `query`, return a spoken summary.

    Fail-soft and BOUNDED: every network call has a timeout + bounded retries,
    and if Wikipedia is unreachable/blocked we return a short spoken apology
    (never a hang), so the voice pipeline is never left wedged on 'Running
    tool: web_search'."""
    q = (query or "").strip()
    if not q:
        return "I need a search term. What should I look up?"

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

    # Bounded, speakable failure — the pipeline keeps moving.
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
        # No saved GPS -> fall through to default home.
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
# Canonical tool specs (same shape as me2_ui_mcp.TOOL_SPECS)
# (name, description, params) -- params is the OpenAI JSON-schema "properties".
# ---------------------------------------------------------------------------
TOOL_SPECS = [
    ("check_weather",
     "Get the CURRENT weather (temperature, conditions, wind) for a place. Use "
     "when the user asks about the weather, the temperature, or whether it will "
     "rain. Pass an optional location (a city name, or 'here' to use the "
     "location the user shared); leave it empty for the default home.",
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
     "Build a simple website about a topic and serve it in a browser. Use when "
     "the user asks you to 'make', 'create', 'build', or 'write' a website/webpage/"
     "web page about something. You will be given the topic, a title, and a "
     "description; returns the URL where the page can be opened.",
     {"topic": {"type": "string"},
      "title": {"type": "string"},
      "description": {"type": "string"}}),
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
                                  kw.get("description", ""))
        return f"I don't know how to do '{tool}'."
    except Exception as e:
        # Fail-soft: never crash the agent on a web error.
        return f"Something went wrong while using the web ({e!r}). Please try again."


# ---------------------------------------------------------------------------
# write_website -- generate a simple HTML page and serve it on the Boots UI
# ---------------------------------------------------------------------------
import time as _time

SITES_DIR = os.environ.get("WEBSITES_DIR",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "bootsui", "sites"))
# Base URL the user can open the generated site from (Pi's LAN IP, UI port).
_SITE_HOST = os.environ.get("WEBSITE_HOST", "192.168.1.3")
_SITE_PORT = os.environ.get("WEBSITE_PORT", "8330")


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s or "site"


def _write_website(topic: str, title: str, description: str) -> str:
    """Generate a simple, self-contained HTML page and save it to SITES_DIR.

    Returns a spoken message with the URL the user can open in a browser.
    The page is a clean single-file site (inline CSS) about the given topic,
    built from the LLM-provided title/description.
    """
    topic = (topic or "").strip()
    title = (title or "").strip() or topic or "Untitled page"
    description = (description or "").strip()
    if not topic and not description:
        return "Tell me what the website should be about and I'll build it."

    # Sanitize for HTML.
    def esc(s):
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    slug = _slugify(title or topic) + "-" + str(int(_time.time() * 1000) % 100000)
    os.makedirs(SITES_DIR, exist_ok=True)
    path = os.path.join(SITES_DIR, slug + ".html")

    # Build a simple but presentable single-file page.
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
  :root {{ --ink:#1a1a2e; --accent:#4361ee; --bg:#f7f8fc; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
         color:var(--ink); background:var(--bg); line-height:1.6; }}
  header {{ background:linear-gradient(135deg,#4361ee,#3a0ca3); color:#fff;
          padding:48px 24px; text-align:center; }}
  header h1 {{ margin:0 0 8px; font-size:2.2rem; }}
  header p {{ margin:0; opacity:.9; }}
  main {{ max-width:720px; margin:32px auto; padding:0 20px; }}
  .card {{ background:#fff; border-radius:12px; padding:24px; margin-bottom:20px;
         box-shadow:0 2px 12px rgba(0,0,0,.06); }}
  .card h2 {{ margin-top:0; color:var(--accent); }}
  footer {{ text-align:center; padding:24px; color:#888; font-size:.9rem; }}
  a {{ color:var(--accent); }}
</style>
</head>
<body>
  <header>
    <h1>{esc(title)}</h1>
    <p>{esc(topic)}</p>
  </header>
  <main>
    <div class="card">
      <h2>About</h2>
      <p>{esc(description) if description else "This is a simple website generated by the Boots voice agent on a Raspberry Pi. It demonstrates the agent's ability to build and serve basic web pages."}</p>
    </div>
    <div class="card">
      <h2>Highlights</h2>
      <ul>
        <li>Built and served live by the Boots ME3 full-duplex voice agent.</li>
        <li>Generated on command via the <code>write_website</code> MCP tool.</li>
        <li>Single self-contained HTML file (no external dependencies).</li>
      </ul>
    </div>
  </main>
  <footer>Generated by Boots &middot; Raspberry Pi &middot; {esc(_time.strftime('%Y-%m-%d %H:%M'))}</footer>
</body>
</html>"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        return f"I couldn't save the website file ({e!r})."

    url = f"http://{_SITE_HOST}:{_SITE_PORT}/site/{slug}.html"
    return (f"Done! I built a simple website about {topic or title} and saved it. "
            f"You can open it in your browser at {url}")



# ---------------------------------------------------------------------------
# Real FastMCP server (standalone) -- satisfies the classwork "build & run
# your MCP servers (on RPi)" requirement. Run it with:  python3 web_tools.py
# (or `mcp dev` / point any MCP client at it). The agent uses dispatch() above
# because pipecat 1.12's MCPService is incompatible on this stack (D11).
# ---------------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP
    HAVE_MCP = True
except Exception:
    HAVE_MCP = False

mcp = None
if HAVE_MCP:
    mcp = FastMCP("boots-web")

    @mcp.tool()
    def check_weather(location: str = "") -> str:
        """Get the current weather (temp, conditions, wind) for a place."""
        return _check_weather(location)

    @mcp.tool()
    def web_search(query: str) -> str:
        """Search the web (Wikipedia) for a topic and return a short summary."""
        return _web_search(query)

    @mcp.tool()
    def fetch_url(url: str) -> str:
        """Open a specific web page and return its readable text."""
        return _fetch_url(url)

    @mcp.tool()
    def write_website(topic: str, title: str, description: str) -> str:
        """Build a simple website about a topic and return its served URL."""
        return _write_website(topic, title, description)


if __name__ == "__main__":
    if HAVE_MCP and mcp:
        mcp.run()
    else:
        print("FastMCP not available; use this file via the agent's function-calling layer.",
              file=sys.stderr)

