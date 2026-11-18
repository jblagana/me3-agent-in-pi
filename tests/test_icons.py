"""Dashboard icons (2026-11-19).

1. files   every icon route points at a file in bootsui/icons/; PNG sizes are right.
2. auth    icons and the manifest load without signing in; the dashboard still doesn't.
3. pages   the dashboard and the login page link the favicon, touch icon and manifest.

Offline. Runs on the laptop too (needs Pillow only for the size check).

    cd ~/me3 && .venv/bin/python tests/test_icons.py
"""
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UI = os.path.join(ROOT, "bootsui")
sys.path.insert(0, UI)

import ui_auth  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def icon_routes():
    src = open(os.path.join(UI, "boots_ui.py"), encoding="utf-8").read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_ICON_ROUTES" for t in node.targets):
            return ast.literal_eval(node.value)
    return {}


def files_case():
    routes = icon_routes()
    check("files: routes found", len(routes) >= 7, len(routes))
    missing = [n for n, _ in routes.values() if not os.path.isfile(os.path.join(UI, "icons", n))]
    check("files: every route has its file", not missing, missing)
    man = json.load(open(os.path.join(UI, "icons", "manifest.webmanifest"), encoding="utf-8"))
    paths = {p for p in routes}
    check("files: manifest icons are all served", all(i["src"] in paths for i in man["icons"]))
    try:
        from PIL import Image
    except Exception:
        print("SKIP sizes (no Pillow)")
        return
    for name, size in (("icon-192.png", 192), ("icon-512.png", 512),
                       ("icon-maskable-512.png", 512), ("apple-touch-icon.png", 180)):
        im = Image.open(os.path.join(UI, "icons", name))
        check(f"files: {name} is {size}px", im.size == (size, size), im.size)
    ico = Image.open(os.path.join(UI, "icons", "favicon.ico"))
    check("files: favicon.ico has 16 and 32", {(16, 16), (32, 32)} <= set(ico.info.get("sizes", ())))


def auth_case():
    os.environ["ME3_UI_TOKEN"] = "test-token-for-icons"
    remote, h = "192.168.1.50", {"Host": "boots:8330"}
    for p in icon_routes():
        check(f"auth: {p} is public", ui_auth.decide("GET", p, remote, h) == "ok")
    check("auth: dashboard still needs login", ui_auth.decide("GET", "/", remote, h) == "login")
    check("auth: API still refused", ui_auth.decide("GET", "/ui_state", remote, h) == "deny")
    check("auth: other /icons/ paths still refused",
          ui_auth.decide("GET", "/icons/../ui_state", remote, h) == "deny")


def pages_case():
    html = open(os.path.join(UI, "boots_ui.html"), encoding="utf-8").read()
    login = ui_auth.LOGIN_PAGE.decode()
    for page, text in (("dashboard", html), ("login", login)):
        check(f"pages: {page} links icons",
              all(s in text for s in ('href="/favicon.ico"', 'href="/apple-touch-icon.png"',
                                      'href="/manifest.webmanifest"')))
    check("pages: header shows the icon", 'class="brand"><img src="/icons/icon.svg"' in html)


if __name__ == "__main__":
    files_case()
    auth_case()
    pages_case()
    ok = all(RESULTS)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)
