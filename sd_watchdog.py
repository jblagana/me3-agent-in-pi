#!/usr/bin/env python3
# Boots -- systemd watchdog pings (2026-11-18).
#
# Restart=always only catches a process that EXITS. A hung one (deadlocked
# event loop, a stuck HTTP server) kept the service "active" forever. With
# WatchdogSec= set on the unit, systemd kills and restarts the service when
# these pings stop. A ping is sent only while the process shows it is alive:
#
#   agent    the asyncio loop beats every few seconds (loop_beat task); a
#            thread pings systemd only while the last beat is fresh. Before
#            the loop starts (waiting for the EMEET can take hours) the
#            thread pings unconditionally: waiting is not hanging.
#   dashboard  a thread fetches /ui_state from the server itself; a ping
#            only follows a successful answer.
#
# No dependency: the protocol is one datagram to $NOTIFY_SOCKET. Outside
# systemd (tests, manual runs) everything here is a no-op.
import asyncio
import os
import socket
import threading
import time

_state = {"beat": None, "started": False}


def _interval():
    """Seconds between pings: a third of WatchdogSec, or None if no watchdog."""
    try:
        usec = int(os.environ.get("WATCHDOG_USEC", "0"))
        pid = int(os.environ.get("WATCHDOG_PID", str(os.getpid())))
    except ValueError:
        return None
    if usec <= 0 or pid != os.getpid():
        return None
    return max(0.05, usec / 1e6 / 3)


def notify(msg):
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(msg.encode())
        return True
    except OSError:
        return False


def start(check=None, stale_after=None):
    """Start the ping thread. check() -> bool decides each ping; without it the
    loop-beat rule (see top) is used. Returns False when there's no watchdog."""
    every = _interval()
    if every is None or _state["started"]:
        return False
    _state["started"] = True
    stale = stale_after or every * 2.2

    def _alive():
        if check is not None:
            try:
                return bool(check())
            except Exception:
                return False
        b = _state["beat"]
        return b is None or time.monotonic() - b < stale

    def _run():
        while True:
            if _alive():
                notify("WATCHDOG=1")
            time.sleep(every)

    threading.Thread(target=_run, daemon=True, name="sd-watchdog").start()
    return True


def beat():
    _state["beat"] = time.monotonic()


async def loop_beat(every=5.0):
    """asyncio task: proves the event loop still runs."""
    while True:
        beat()
        await asyncio.sleep(every)


def http_check(url, timeout=10):
    def _check():
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    return _check
