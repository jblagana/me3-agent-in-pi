#!/usr/bin/env python3
# Boots dashboard -- login token for other devices (2026-11-18, HANDOVER 5.1).
#
# The dashboard listens on 0.0.0.0:8330 and can run tools, change the LLM
# endpoint and read the conversation, with no login. Now:
#
#   * Requests from the Pi itself (127.0.0.1 / ::1) pass: the voice agent's
#     pushes, the tools' /fire calls and a browser on the Pi.
#   * Any other device needs the token once: /login (or /login#<token>, which
#     fills the form from the URL fragment, which never reaches the server or logs).
#     The server then sets an HttpOnly, SameSite=Strict cookie holding an
#     HMAC of the token (not the token itself). A new token logs every
#     device out.
#   * POSTs whose Origin is another site are refused (a web page can't drive
#     Boots from someone's browser), local or not.
#   * Generated websites (/site/...) stay open: they're meant to be shared.
#   * No token configured -> other devices are refused (fails closed).
#
# The token lives in /etc/me3/ui.env as ME3_UI_TOKEN (root-only; systemd
# passes it in through EnvironmentFile=).
import hashlib
import hmac
import os
import time
from http.cookies import SimpleCookie
from urllib.parse import urlsplit

COOKIE = "boots_auth"
MAX_AGE = 365 * 24 * 3600
LOCAL = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
PUBLIC_PREFIXES = ("/site/",)
# Icons are public too (exact names only): the login page shows them, and
# browsers fetch the manifest and favicon without cookies. Nothing private.
PUBLIC_PATHS = ("/login", "/logout", "/favicon.ico", "/apple-touch-icon.png",
                "/manifest.webmanifest", "/icons/icon.svg", "/icons/icon-192.png",
                "/icons/icon-512.png", "/icons/icon-maskable-512.png")
FAIL_DELAY_S = 1.0


def token():
    return os.environ.get("ME3_UI_TOKEN", "").strip()


def cookie_value(tok=None):
    tok = token() if tok is None else tok
    return hmac.new(tok.encode(), b"boots-ui-cookie-v1", hashlib.sha256).hexdigest() if tok else ""


def is_local(ip):
    return (ip or "") in LOCAL


def has_cookie(cookie_header):
    want = cookie_value()
    if not want or not cookie_header:
        return False
    try:
        c = SimpleCookie()
        c.load(cookie_header)
        got = c[COOKIE].value if COOKIE in c else ""
    except Exception:
        return False
    return hmac.compare_digest(got, want)


def check_token(given):
    tok = token()
    ok = bool(tok) and hmac.compare_digest((given or "").strip().encode(), tok.encode())
    if not ok:
        time.sleep(FAIL_DELAY_S)          # slows guessing; the token is 256-bit anyway
    return ok


def set_cookie_header():
    return (f"{COOKIE}={cookie_value()}; Path=/; Max-Age={MAX_AGE}; HttpOnly; SameSite=Strict")


def clear_cookie_header():
    return f"{COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict"


def origin_ok(origin, host):
    """A POST may come with no Origin (curl, the agent) or from this server's
    own pages. Anything else is another site."""
    if not origin:
        return True
    try:
        return urlsplit(origin).netloc.lower() == (host or "").lower()
    except Exception:
        return False


def decide(method, path, client_ip, headers):
    """'ok' | 'login' (send to the login page) | 'deny' (401) | 'cross' (403)."""
    if method == "POST" and not origin_ok(headers.get("Origin"), headers.get("Host")):
        return "cross"
    p = (path or "").split("?", 1)[0]
    if p.startswith(PUBLIC_PREFIXES) or p in PUBLIC_PATHS:
        return "ok"
    if is_local(client_ip) or has_cookie(headers.get("Cookie")):
        return "ok"
    if method == "GET" and p in ("/", "/index.html", "/muji", "/classic"):
        return "login"
    return "deny"


LOGIN_PAGE = b"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Boots sign in</title>
<link rel="icon" href="/favicon.ico" sizes="48x48"><link rel="icon" href="/icons/icon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/apple-touch-icon.png"><link rel="manifest" href="/manifest.webmanifest">
<meta name="theme-color" content="#111412">
<style>
:root{--bg:#f6f5f1;--fg:#1f2a22;--muted:#66705f;--line:#d9dccf;--acc:#3f6b4a;--err:#a3402e}
@media (prefers-color-scheme:dark){:root{--bg:#141815;--fg:#e6eae2;--muted:#9aa594;--line:#2c332c;--acc:#7fb08a;--err:#e08a76}}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;
background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;padding:16px}
form{width:100%;max-width:360px;border:1px solid var(--line);border-radius:14px;padding:24px}
h1{font-size:20px;margin:0 0 4px}p{margin:0 0 16px;color:var(--muted);font-size:14px}
input{width:100%;padding:10px 12px;border:1px solid var(--line);border-radius:10px;
background:transparent;color:var(--fg);font:inherit}
button{margin-top:12px;width:100%;padding:10px;border:0;border-radius:10px;background:var(--acc);
color:#fff;font:inherit;cursor:pointer}.err{color:var(--err);font-size:14px;margin-top:10px}
</style></head><body>
<form method="post" action="/login" id="f">
<h1>Boots</h1><p>Enter the dashboard token to use Boots from this device.</p>
<input name="token" id="t" type="password" autocomplete="current-password" placeholder="Token" required autofocus>
<button type="submit">Sign in</button>
<!--ERR-->
</form>
<script>
var h=location.hash.slice(1);
if(h){history.replaceState(null,"",location.pathname);document.getElementById("t").value=decodeURIComponent(h);document.getElementById("f").submit();}
</script></body></html>"""


def login_page(error=False):
    if not error:
        return LOGIN_PAGE
    return LOGIN_PAGE.replace(b"<!--ERR-->", b'<div class="err">That token is not right.</div>')
