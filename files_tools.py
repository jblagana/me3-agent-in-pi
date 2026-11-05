#!/usr/bin/env python3
# Boots -- File tools (TOOL_SPECS + dispatch; mirrors me2_tools.py / web_tools.py).
#
# Tools the voice agent can use on the LOCAL filesystem (the Pi's):
#   1. list_dir    -- "what's in my notes folder"
#   2. read_file   -- "read my notes file"        (capped output for voice)
#   3. write_file  -- "write this to a file"
#   4. edit_file   -- "change X to Y in that file"
#   5. search_files-- "which file mentions X"      (regex, recursive, capped)
#
# ARCHITECTURE (same contract as me2_tools.py, see its D11 note):
#   * Plain module: TOOL_SPECS + dispatch(). agent.py imports it for its
#     native FunctionSchema layer; mcp_server.py can import it for the
#     unified MCP facade.
#
# VOICE SAFETY:
#   * Every result is a SHORT string the LLM can speak (1-2 sentences + at
#     most a handful of items). read_file/search_files are length-capped so
#     a big file can never dump thousands of tokens into the LLM context.
#   * Paths are resolved relative to BOOT_FILES_ROOT (default: the Pi home)
#     and are ROOT-PRISONED: a path that escapes the root (e.g. ../../../etc)
#     is rejected, so a voice turn can never touch system files.
#
import os
import re

# Where "my files" live. Overridable via env (the Pi agent sets it to the
# user's home so relative paths land somewhere sane).
BOOT_FILES_ROOT = os.environ.get("BOOT_FILES_ROOT", os.path.expanduser("~"))

MAX_READ_CHARS = 1500      # read_file output cap (voice context budget)
MAX_LIST_ITEMS = 25        # list_dir item cap
MAX_SEARCH_HITS = 15       # search_files hit cap
MAX_SEARCH_LINE = 120      # per-hit line truncation
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
             ".cache", ".local", ".config", "sites"}


def _resolve(path: str) -> str:
    """Resolve `path` against the root and prison it inside the root.

    Returns the absolute path, or raises ValueError if it escapes the root.
    """
    p = (path or "").strip()
    if not p:
        raise ValueError("no path given")
    if not os.path.isabs(p):
        p = os.path.join(BOOT_FILES_ROOT, p)
    p = os.path.normpath(os.path.expanduser(p))
    root = os.path.normpath(BOOT_FILES_ROOT)
    if p != root and not p.startswith(root + os.sep):
        raise ValueError(f"path '{path}' is outside the allowed folder ({os.path.basename(root)})")
    return p


def _tool_list_dir(path: str) -> str:
    p = _resolve(path)
    if not os.path.isdir(p):
        return f"That's not a folder: {path}."
    try:
        entries = sorted(os.listdir(p))
    except Exception:
        return f"I couldn't open that folder: {path}."
    if not entries:
        return f"{os.path.basename(p)} is empty."
    dirs = [e for e in entries if os.path.isdir(os.path.join(p, e))]
    files = [e for e in entries if not os.path.isdir(os.path.join(p, e))]
    shown = entries[:MAX_LIST_ITEMS]
    more = len(entries) - len(shown)
    parts = []
    if dirs:
        parts.append(f"{len(dirs)} folders")
    if files:
        parts.append(f"{len(files)} files")
    names = ", ".join(shown)
    out = f"{os.path.basename(p)} has {' and '.join(parts)}: {names}."
    if more > 0:
        out += f" And {more} more."
    return out[:MAX_READ_CHARS]


def _tool_read_file(path: str) -> str:
    p = _resolve(path)
    if not os.path.isfile(p):
        return f"I can't find that file: {path}."
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(MAX_READ_CHARS + 1)
    except Exception:
        return f"I couldn't read that file: {path}."
    truncated = len(text) > MAX_READ_CHARS
    text = text[:MAX_READ_CHARS]
    out = f"Here's {os.path.basename(path)}: {text}"
    if truncated:
        out += " (that's the first part — say the line range if you want more.)"
    return out


def _tool_write_file(path: str, content: str) -> str:
    p = _resolve(path)
    if os.path.isdir(p):
        return f"That's a folder, not a file: {path}."
    try:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        existed = os.path.isfile(p)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return f"I couldn't write that file: {e}."
    verb = "updated" if existed else "wrote"
    return f"I {verb} {os.path.basename(path)} with your text."


def _tool_edit_file(path: str, old_text: str, new_text: str) -> str:
    p = _resolve(path)
    if not os.path.isfile(p):
        return f"I can't find that file: {path}."
    if not old_text:
        return "Tell me exactly what text to replace."
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return f"I couldn't read that file: {path}."
    n = text.count(old_text)
    if n == 0:
        return f"I couldn't find that exact text in {os.path.basename(path)} — read it first and give me the exact string."
    if n > 1:
        return f"That text appears {n} times in {os.path.basename(path)} — give me a longer, unique snippet."
    try:
        with open(p, "w", encoding="utf-8") as f:
            f.write(text.replace(old_text, new_text, 1))
    except Exception as e:
        return f"I couldn't save the edit: {e}."
    return f"Done — I replaced it in {os.path.basename(path)}."


def _tool_search_files(pattern: str, path: str = "", glob: str = "") -> str:
    p = _resolve(path or ".")
    if not os.path.isdir(p):
        p = _resolve(os.path.dirname(path) or ".")
        if not os.path.isdir(p):
            return f"That's not a folder: {path}."
    try:
        rx = re.compile(pattern)
    except re.error:
        rx = re.compile(re.escape(pattern))
    hits = []
    scanned = 0
    try:
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if glob and not re.fullmatch(glob, fn):
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(fp) > 1_000_000:
                        continue
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if rx.search(line):
                                hits.append((fp, i, line.strip()[:MAX_SEARCH_LINE]))
                                if len(hits) >= MAX_SEARCH_HITS:
                                    break
                except Exception:
                    continue
                scanned += 1
                if len(hits) >= MAX_SEARCH_HITS:
                    break
            if len(hits) >= MAX_SEARCH_HITS:
                break
    except Exception:
        pass
    if not hits:
        return f"No matches for '{pattern}' under {os.path.basename(p)}."
    rel_root = os.path.dirname(BOOT_FILES_ROOT)
    lines = [f"{os.path.relpath(fp, rel_root)}:{i}: {ln}" for fp, i, ln in hits]
    out = f"Found {len(hits)} match" + ("es" if len(hits) != 1 else "") + f" in {scanned} files:\n" + "\n".join(lines)
    return out[:MAX_READ_CHARS + 400]


TOOL_SPECS = [
    ("list_dir",     "List the folders and files inside a folder.",
                    {"path": "string"}),
    ("read_file",    "Read a text file and report its contents.",
                    {"path": "string"}),
    ("write_file",   "Write text to a file (creates it, or overwrites it).",
                    {"path": "string", "content": "string"}),
    ("edit_file",    "Replace one exact piece of text with new text in a file.",
                    {"path": "string", "old_text": "string", "new_text": "string"}),
    ("search_files", "Search file contents for a pattern and report the matching lines.",
                    {"pattern": "string", "path": "string", "glob": "string"}),
]

_DISPATCH = {
    "list_dir":     lambda **kw: _tool_list_dir(kw.get("path", "")),
    "read_file":    lambda **kw: _tool_read_file(kw.get("path", "")),
    "write_file":   lambda **kw: _tool_write_file(kw.get("path", ""), kw.get("content", "")),
    "edit_file":    lambda **kw: _tool_edit_file(kw.get("path", ""), kw.get("old_text", ""), kw.get("new_text", "")),
    "search_files": lambda **kw: _tool_search_files(kw.get("pattern", ""), kw.get("path", ""), kw.get("glob", "")),
}


def dispatch(tool: str, **kw) -> str:
    """Execute a file tool. Returns a short string the LLM can speak."""
    fn = _DISPATCH.get(tool)
    if fn is None:
        return f"I don't know how to do '{tool}'."
    try:
        return fn(**kw)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Something went wrong with that file action: {e}."
