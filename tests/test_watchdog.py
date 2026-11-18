"""systemd watchdog pings (2026-11-18, sd_watchdog.py).

Fakes systemd's notify socket and checks, in separate processes:
1. waiting   before the event loop beats, pings flow (EMEET wait = alive)
2. stale     once beats stop, pings stop (hung loop -> systemd restarts)
3. http      the dashboard check pings only while the server answers
4. off       no WATCHDOG_USEC -> start() does nothing

Linux only (unix datagram socket). No network.

    cd ~/me3 && .venv/bin/python tests/test_watchdog.py
"""
import os
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


CHILD = r"""
import sys, time, asyncio, threading
sys.path.insert(0, %r)
import sd_watchdog as w
mode = sys.argv[1]
if mode == "loop":
    w.start()
    time.sleep(1.0)                       # "waiting for the EMEET": no beats yet
    async def main():
        t = asyncio.get_running_loop().create_task(w.loop_beat(0.1))
        await asyncio.sleep(1.0)          # healthy loop
        t.cancel()
        time.sleep(2.0)                   # loop blocked: no beats
    asyncio.run(main())
elif mode == "http":
    up = {"v": True}
    w.start(check=lambda: up["v"])
    time.sleep(1.0); up["v"] = False; time.sleep(1.5)
elif mode == "off":
    print("started" if w.start() else "off")
"""


def run(mode, usec, sock_path):
    env = dict(os.environ, NOTIFY_SOCKET=sock_path)
    if usec:
        env["WATCHDOG_USEC"] = str(usec)
    else:
        env.pop("WATCHDOG_USEC", None)
    env.pop("WATCHDOG_PID", None)
    return subprocess.Popen([sys.executable, "-c", CHILD % ROOT, mode], env=env,
                            stdout=subprocess.PIPE, text=True)


def collect(proc, srv, secs):
    got, t0 = [], time.monotonic()
    srv.settimeout(0.05)
    while time.monotonic() - t0 < secs:
        try:
            got.append((time.monotonic() - t0, srv.recv(64).decode()))
        except socket.timeout:
            pass
    proc.wait(5)
    return got


if __name__ == "__main__":
    d = tempfile.mkdtemp(prefix="wd-test-")
    path = os.path.join(d, "notify")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(path)
    # WatchdogSec 0.3 s -> a ping every 0.1 s; stale after 0.22 s
    pings = collect(run("loop", 300000, path), srv, 4.3)
    t = [p[0] for p in pings if p[1] == "WATCHDOG=1"]
    waiting = [x for x in t if x < 0.9]
    healthy = [x for x in t if 1.2 < x < 1.9]
    hung = [x for x in t if x > 2.5]
    check("waiting: pings before the loop starts", len(waiting) >= 4, str(len(waiting)))
    check("healthy loop: pings continue", len(healthy) >= 4, str(len(healthy)))
    check("hung loop: pings stop", not hung, str(hung))
    pings = collect(run("http", 300000, path), srv, 2.8)
    t = [p[0] for p in pings]
    check("http: pings while the server answers, none after",
          sum(1 for x in t if x < 0.9) >= 4 and not [x for x in t if x > 1.4], str(t))
    p = run("off", 0, path)
    out = p.communicate(timeout=5)[0].strip()
    check("off: no WATCHDOG_USEC -> nothing started", out == "off", out)
    ok = all(RESULTS)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.stdout.flush()
    os._exit(0 if ok else 1)
