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

USER_AGENT = "BootsVoiceAgent/1.0 (Raspberry Pi; contact: jan)"
TIMEOUT = float(os.environ.get("WEB_TOOL_TIMEOUT", "8.0"))


def _get(url: str, headers=None, timeout=None) -> str:
    """GET a URL, return the body as text. Fail-soft (returns '' on error)."""
    h = {"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout or TIMEOUT) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# web_search -- Wikipedia opensearch -> REST summary extract
# ---------------------------------------------------------------------------
def _web_search(query: str) -> str:
    """Search the web (via Wikipedia) for `query`, return a spoken summary."""
    q = (query or "").strip()
    if not q:
        return "I need a search term. What should I look up?"

    # 1) Find candidate article titles + URLs via MediaWiki opensearch.
    api = ("https://en.wikipedia.org/w/api.php?action=opensearch&search="
           + urllib.parse.quote(q)
           + "&limit=5&namespace=0&format=json&redirects=resolve")
    body = _get(api)
    candidates = []  # list of (title, url)
    if body:
        try:
            data = json.loads(body)
            titles = data[1] if len(data) > 1 else []
            urls = data[3] if len(data) > 3 else []
            for i, t in enumerate(titles):
                u = urls[i] if i < len(urls) else ""
                candidates.append((t, u))
        except Exception:
            pass

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
# Canonical tool specs (same shape as me2_ui_mcp.TOOL_SPECS)
# (name, description, params) -- params is the OpenAI JSON-schema "properties".
# ---------------------------------------------------------------------------
TOOL_SPECS = [
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

