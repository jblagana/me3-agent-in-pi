#!/usr/bin/env python3
# Boots -- Shell tool (TOOL_SPECS + dispatch; mirrors files_tools.py).
#
# One tool:
#   run_command -- "check disk space", "what git branch am I on", "ping the DGX"
#
# CONTRACT (same as the other tool modules): TOOL_SPECS + dispatch(name, **args)
# returning a SHORT string the LLM can speak (1-2 sentences + a few lines max).
#
# SAFETY MODEL (voice-driven, so it must be conservative):
#   * Destructive verbs are BLOCKED: delete / remove / move / rename / copy,
#     permissions, disk writes, package installs, service control, shutdown,
#     network config, and raw shell escapes (sudo, su, python -c, ...).
#   * Shell metacharacters are BLOCKED: pipes, redirects, chaining (;, &&, ||),
#     subshells, command substitution -- a single flat command only.
#   * Writes via redirection are impossible (redirects blocked) and writes to
#     system paths are blocked (/etc, /boot, /usr, /lib, /opt, /sys, /proc,
#     /dev, /root).
#   * Output is capped (1200 chars) and the command times out (20 s) so a
#     hung command can never wedge the voice pipeline.
#
# The blocklist is a guardrail, not a sandbox: if a command slips through,
# it runs as the agent user (jan) with jan's privileges. That is the
# agreed trade for voice shell access (boss order, 2026-11-06).
#
import os
import re
import subprocess

BOOT_SHELL_ROOT = os.environ.get("BOOT_SHELL_ROOT", os.path.expanduser("~"))

TIMEOUT_S = 20           # hard kill for hung commands (voice can't wait)
MAX_OUT_CHARS = 1200     # output cap (voice context budget)

# Destructive / privileged / escape verbs. Matched as whole words anywhere in
# the command line (covers `bash -c "rm ..."`, flags, and chained words).
BLOCKED_VERBS = {
    # delete / move / rename / copy (boss's off-limits list)
    "rm", "rmdir", "unlink", "shred", "mv", "cp", "rsync",
    "rename", "trash-put", "trash-restore",
    # permissions / ownership
    "chmod", "chown", "chgrp", "setfacl",
    # disk / block device writes
    "dd", "mkfs", "fdisk", "parted", "wipefs", "losetup", "mount", "umount",
    # package / system installs
    "apt", "apt-get", "dpkg", "pip", "pip3", "npm", "yarn", "snap",
    "update-alternatives",
    # service / boot control
    "systemctl", "service", "init", "shutdown", "reboot", "halt", "poweroff",
    "kill", "killall", "pkill",
    # privilege escalation / raw interpreters / escapes
    "sudo", "su", "doas", "pkexec", "eval", "exec", "source", "loadkeys",
    # network / system config
    "ip", "ifconfig", "route", "iptables", "nft", "ufw", "systemd-resolve",
    "resolvectl", "crontab", "at",
    # firmware / kernel
    "modprobe", "rmmod", "insmod", "rpi-clone",
}

# Metacharacters: pipes, redirects, chaining, subshells, command substitution,
# glob-ish process substitution. A flat single command only.
BLOCKED_METACHAR = re.compile(r"[|&;<>()`$]|\$\{|\n")

# Writing to system paths (redirects are already blocked; this catches
# commands that take a path argument and write).
SYSTEM_PATH = re.compile(
    r"(^|\s)(/etc|/boot|/usr|/lib|/lib64|/opt|/sys|/proc|/dev|/root|/bin|/sbin)\b")

# `python -c`, `perl -e`, `node -e` style inline code (arbitrary code run).
INLINE_CODE = re.compile(r"\b(python3?|perl|ruby|node|php)\s+(-[a-zA-Z]*c|-e)\b")


def _check(command: str):
    """Return None if the command is allowed, else a short refusal string."""
    if not command or not command.strip():
        return "No command given."
    if BLOCKED_METACHAR.search(command):
        return ("I can't run that — it uses shell tricks (pipes, redirects, "
                "chaining). Give me one plain command.")
    if INLINE_CODE.search(command):
        return "I can't run inline code like that."
    words = re.findall(r"[A-Za-z0-9_./-]+", command)
    for w in words:
        base = os.path.basename(w)
        if base in BLOCKED_VERBS:
            return f"I can't run {base} — that's a destructive or system command, off limits for me."
    if SYSTEM_PATH.search(command):
        return "That touches system folders — off limits. I only work in your home folder."
    return None


def _run(command: str, cwd: str) -> str:
    """Run the (already validated) command; return a speakable result."""
    work = os.path.expanduser(cwd) if cwd else BOOT_SHELL_ROOT
    if not os.path.isdir(work):
        work = BOOT_SHELL_ROOT
    kwargs = dict(shell=True, cwd=work, stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT, text=True, errors="replace")
    if os.name == "posix":
        kwargs["start_new_session"] = True   # own process group -> clean kill
    else:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        proc = subprocess.Popen(command, **kwargs)
    except Exception as e:
        return f"The command failed to start: {type(e).__name__}."
    try:
        out, _ = proc.communicate(timeout=TIMEOUT_S)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        # Kill the WHOLE group: with shell=True the shell's children
        # (ping, sleep, ...) survive a bare proc.kill().
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), 9)
            else:
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        return f"That took longer than {TIMEOUT_S} seconds, so I stopped it."
    out = (out or "").strip()
    total = len(out)
    if total > MAX_OUT_CHARS:
        out = out[:MAX_OUT_CHARS] + f"\n…(truncated, {total} chars total)"
    if rc != 0:
        head = out.splitlines()[:3]
        snippet = " ".join(head)[:300] if head else "no output"
        return f"The command exited {rc}. It said: {snippet}"
    if not out:
        return "The command finished cleanly with no output."
    return out


TOOL_SPECS = [
    ("run_command",
     "Run a single shell command on this machine and report its output. "
     "Use for checks: disk space, git status, process list, pings, file info. "
     "Destructive commands (delete, move, rename, copy), permission changes, "
     "package installs, service control, and shell pipes/redirects are blocked. "
     "One plain command at a time.",
     {"command": "string", "cwd": "string"}),
    ("run_python",
     "Run a short Python 3 program on this machine and report what it prints. "
     "Use for calculations, unit conversions, text or data processing and quick "
     "scripts. Write the WHOLE program yourself and print() the answer. No "
     "input(), no network, standard library only; it must finish within 10 "
     "seconds. If it reports an error, fix the code and call again once.",
     {"code": "string"}),
]

# run_python limits (2026-11-07). Not a full sandbox: the program runs as the
# Boots user, so it is contained by limits, a scratch working directory, an
# empty environment (no API keys) and a refusal for secret paths, other
# programs, file deletion and the network.
PY_TIMEOUT_S = 10
PY_MAX_OUT = 3000
_PY_FORBIDDEN = re.compile(r"/etc/me3|groq\.env|\.spotify-mcp|\.cache/librespot|\.ssh|"
                           r"subprocess|os\.system|os\.popen|shutil\.rmtree|os\.remove|os\.unlink|"
                           r"socket|urllib|requests|http\.client|ctypes", re.I)


def _run_python(code: str) -> str:
    import sys
    import tempfile
    code = (code or "").strip()
    if not code:
        return "There's no code to run."
    if len(code) > 20000:
        return "That program is too long for a quick run (20,000 characters max)."
    if _PY_FORBIDDEN.search(code):
        return ("I won't run that: it touches the network, other programs, file deletion "
                "or private files. Rewrite it with plain Python computation only.")

    def _limits():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (PY_TIMEOUT_S, PY_TIMEOUT_S))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20, 512 * 2**20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 2**20, 10 * 2**20))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))

    with tempfile.TemporaryDirectory(prefix="boots_py_") as d:
        path = os.path.join(d, "program.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True,
                               text=True, timeout=PY_TIMEOUT_S + 2, preexec_fn=_limits,
                               env={"PATH": "/usr/bin:/bin", "HOME": d, "LANG": "C.UTF-8"},
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return f"The program didn't finish within {PY_TIMEOUT_S} seconds, so I stopped it."
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    if r.returncode in (-9, -24):     # SIGKILL / SIGXCPU from the CPU-time limit
        return f"The program didn't finish within {PY_TIMEOUT_S} seconds of CPU time, so it was stopped."
    if r.returncode != 0:
        last = err.splitlines()[-1] if err else f"exit code {r.returncode}"
        return (f"The program failed: {last[:300]}"
                + (f" (output before the error: {out[:500]})" if out else ""))
    if not out:
        return "The program ran without errors but printed nothing. Use print() for the answer."
    if len(out) > PY_MAX_OUT:
        out = out[:PY_MAX_OUT] + f"... [{len(out) - PY_MAX_OUT} more characters]"
    return "Program output:\n" + out



def dispatch(name: str, **args) -> str:
    if name == "run_python":
        return _run_python(str(args.get("code") or ""))
    if name != "run_command":
        return f"I don't know how to do that ({name})."
    command = str(args.get("command") or "").strip()
    cwd = str(args.get("cwd") or "").strip()
    refusal = _check(command)
    if refusal:
        return refusal
    return _run(command, cwd)
