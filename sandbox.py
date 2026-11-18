#!/usr/bin/env python3
# Boots -- confine run_command / run_python to the Boots files folder (2026-11-18).
#
# The file tools only accept paths inside BOOT_FILES_ROOT, but a shell command
# ("cat ~/.ssh/...") or a Python program (open(...)) could read anything the
# user can. bubblewrap (apt: bubblewrap) runs them in a mount namespace where:
#
#   /home        is empty except BOOT_FILES_ROOT (read-write)
#   /etc/me3, /root, /tmp   are empty, private tmpfs
#   everything else          read-only (system tools still work: df, ping, ls)
#   network      kept for run_command (ping), cut for run_python
#
# No root needed (unprivileged user namespaces). If bwrap is missing (a laptop,
# a test box) the callers fall back to their own checks and say so in the log.
import os
import shutil

FILES_ROOT = os.path.realpath(os.environ.get(
    "BOOT_FILES_ROOT", os.path.join(os.path.expanduser("~"), "boots-files")))
BWRAP = shutil.which("bwrap")
SYSTEM_PYTHON = "/usr/bin/python3"     # the venv lives under /home: hidden in the box


def available():
    return bool(BWRAP) and os.environ.get("ME3_SANDBOX", "1") != "0"


def ensure_root():
    os.makedirs(FILES_ROOT, exist_ok=True)
    return FILES_ROOT


def wrap(argv, cwd=None, net=True):
    """argv -> bwrap argv running it confined, with cwd inside FILES_ROOT."""
    ensure_root()
    cwd = os.path.realpath(cwd or FILES_ROOT)
    if cwd != FILES_ROOT and not cwd.startswith(FILES_ROOT + os.sep):
        cwd = FILES_ROOT
    box = [BWRAP, "--ro-bind", "/", "/",
           "--tmpfs", "/home", "--bind", FILES_ROOT, FILES_ROOT, "--remount-ro", "/home",
           "--tmpfs", "/tmp", "--tmpfs", "/root", "--proc", "/proc", "--dev", "/dev",
           "--unshare-pid", "--unshare-ipc", "--unshare-uts",
           "--die-with-parent", "--new-session",
           "--setenv", "HOME", FILES_ROOT, "--chdir", cwd]
    if os.path.isdir("/etc/me3"):
        box[box.index("--proc"):box.index("--proc")] = ["--tmpfs", "/etc/me3"]
    if not net:
        box.append("--unshare-net")
    return box + list(argv)
