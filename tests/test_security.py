"""Dashboard login and the boots-files folder (2026-11-18, HANDOVER 5.1 and 5.3).

1. auth     the Pi itself passes; another device needs the cookie; the page
            sends it to /login; a wrong cookie fails; cross-site POSTs are
            refused; /site/ stays open; no token configured -> refused.
2. files    the file tools work only inside BOOT_FILES_ROOT; ../ and a
            symlink pointing outside are refused.
3. sandbox  run_command and run_python can't see the rest of the home folder
            or /etc/me3; the system is read-only; run_python has no network.
            (Skipped where bubblewrap isn't installed.)

    cd ~/me3 && .venv/bin/python tests/test_security.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bootsui"))
TMP = tempfile.mkdtemp(prefix="boots-sec-test-", dir=os.path.expanduser("~"))
os.environ["BOOT_FILES_ROOT"] = os.path.join(TMP, "boots-files")
os.environ["ME3_UI_TOKEN"] = "test-token-" + "z" * 30

import ui_auth as auth   # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def auth_case():
    auth.FAIL_DELAY_S = 0
    H = lambda **kw: {"Host": "me2pi.local:8330", **kw}
    good = f"{auth.COOKIE}={auth.cookie_value()}"
    check("auth: the Pi itself passes", auth.decide("POST", "/chat", "127.0.0.1", H()) == "ok")
    check("auth: another device without a cookie is refused", auth.decide("POST", "/chat", "192.168.137.5", H()) == "deny")
    check("auth: ... and its page goes to /login", auth.decide("GET", "/", "192.168.137.5", H()) == "login")
    check("auth: with the cookie it passes", auth.decide("GET", "/ui_state", "192.168.137.5", H(Cookie=good)) == "ok")
    check("auth: a wrong cookie fails",
          auth.decide("GET", "/ui_state", "192.168.137.5", H(Cookie=f"{auth.COOKIE}=abc")) == "deny")
    check("auth: cookie is not the token itself", os.environ["ME3_UI_TOKEN"] not in good)
    check("auth: cross-site POST refused, even from the Pi",
          auth.decide("POST", "/fire", "127.0.0.1", H(Origin="http://evil.example")) == "cross")
    check("auth: same-site POST allowed",
          auth.decide("POST", "/chat", "192.168.137.5", H(Origin="http://me2pi.local:8330", Cookie=good)) == "ok")
    check("auth: generated sites stay open", auth.decide("GET", "/site/demo.html", "10.0.0.9", H()) == "ok")
    check("auth: token check", auth.check_token(os.environ["ME3_UI_TOKEN"]) and not auth.check_token("nope")
          and not auth.check_token(""))
    old = os.environ.pop("ME3_UI_TOKEN")
    check("auth: no token configured -> other devices refused",
          auth.decide("GET", "/ui_state", "192.168.137.5", H(Cookie=f"{auth.COOKIE}=")) == "deny"
          and not auth.check_token(""))
    os.environ["ME3_UI_TOKEN"] = old


def files_case():
    import files_tools as ft
    root = ft.BOOT_FILES_ROOT
    check("files: root is the dedicated folder", root == os.path.realpath(os.environ["BOOT_FILES_ROOT"]), root)
    out = ft.dispatch("write_file", path="notes.txt", content="hello")
    check("files: write inside works", os.path.exists(os.path.join(root, "notes.txt")), out)
    out = ft.dispatch("read_file", path="../../.bashrc")
    check("files: ../ escape refused", "only use files" in out, out)
    out = ft.dispatch("read_file", path=os.path.expanduser("~/.bashrc"))
    check("files: absolute path outside refused", "only use files" in out, out)
    secret = os.path.join(TMP, "secret.txt")
    open(secret, "w").write("SECRET-VALUE")
    try:
        os.symlink(secret, os.path.join(root, "link.txt"))
        out = ft.dispatch("read_file", path="link.txt")
        check("files: symlink pointing outside refused", "SECRET-VALUE" not in out and "only use files" in out, out)
        out = ft.dispatch("search_files", pattern="SECRET", path=".")
        check("files: search skips the symlink", "SECRET-VALUE" not in out, out)
    except (OSError, NotImplementedError) as e:
        print(f"SKIP symlink checks ({e!r})")


def sandbox_case():
    import sandbox
    import shell_tools as sh
    if not sandbox.available():
        print("SKIP sandbox (bubblewrap not installed here)")
        return
    out = sh.dispatch("run_command", command="ls " + os.path.expanduser("~"))
    check("sandbox: home folder shows only boots-files",
          os.path.basename(TMP) in out and "me3" not in out.split(), out)
    out = sh.dispatch("run_command", command="cat " + os.path.join(TMP, "secret.txt"))
    check("sandbox: file outside the folder is invisible", "SECRET-VALUE" not in out, out)
    out = sh.dispatch("run_command", command="ls /etc/me3")
    check("sandbox: /etc/me3 hidden", "groq.env" not in out, out)
    out = sh.dispatch("run_command", command="cat notes.txt")
    check("sandbox: the folder itself works", "hello" in out, out)
    out = sh.dispatch("run_command", command="touch /home/x")
    check("sandbox: can't write outside the folder", "Read-only" in out, out)
    out = sh.dispatch("run_command", command="touch /var/x")    # not on the blocklist: the box stops it
    check("sandbox: system is read-only", "Read-only" in out or "Permission" in out, out)
    out = sh.dispatch("run_python", code="print(open('notes.txt').read())")
    check("sandbox: python reads the folder", "hello" in out, out)
    out = sh.dispatch("run_python", code=f"print(open({os.path.join(TMP, 'secret.txt')!r}).read())")
    check("sandbox: python can't read outside", "SECRET-VALUE" not in out, out)
    out = sh.dispatch("run_python", code="import os\nprint(sorted(os.listdir('/home')))")
    check("sandbox: python sees no other home folders", "'jan'" in out or "[" in out and "me3" not in out, out)
    # spelled so the older word filter lets it through: this checks the box itself
    code = "m=__import__('_so'+'cket')\ns=getattr(m,'so'+'cket')()\ns.settimeout(2)\ntry:\n s.connect(('8.8.8.8',53)); print('NET')\nexcept OSError: print('nonet')"
    out = sh.dispatch("run_python", code=code)
    check("sandbox: python has no network", "nonet" in out and "NET\n" not in out, out)


if __name__ == "__main__":
    import shutil
    try:
        auth_case()
        files_case()
        sandbox_case()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    ok = all(RESULTS)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.stdout.flush()
    os._exit(0 if ok else 1)
