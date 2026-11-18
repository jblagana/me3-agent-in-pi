#!/usr/bin/env python3
# Boots -- send only the tools a turn is likely to need (2026-11-18).
#
# All 47 tool definitions were sent on every LLM call: about 4,000 of the
# ~6,000 input tokens, i.e. most of the Groq bill. Now each call carries a
# small CORE set (notes, web search, fetch, weather, load_tools) plus the
# groups the user's words point to, found by keywords (no LLM call):
#
#   home     lights, AC, timers, alarms, reminders, calls, volume (me2_tools)
#   music    Spotify + playback controls
#   build    write_website, build_app, publish_website (640+ tokens alone)
#   files    read / write / edit / search files
#   shell    run_command, run_python
#   library  the document library (always on in files-only mode)
#
# Follow-ups keep working: the previous user message is routed too, and any
# group whose tool ran in the last two turns stays loaded. If the model still
# needs a missing tool it calls load_tools(groups) and the next round of the
# same turn has them (one extra round, only on a miss).
#
# ME3_TOOL_ROUTER=0 sends every tool, as before.
import os
import re

ENABLED = os.environ.get("ME3_TOOL_ROUTER", "1") != "0"

_MUSIC_UI = {"boots_play_music", "boots_pause", "boots_resume", "boots_stop", "boots_next",
             "boots_volume_up", "boots_volume_down", "boots_set_volume"}
CORE_NAMES = {"web_search", "fetch_url", "check_weather", "load_tools"}

KEYWORDS = {
    "music": r"\b(play|played|playing|song|songs|music|spotify|album|artist|band|track|tracks|"
             r"pause|resume|skip|next|previous|stop|volume|louder|quieter|softer|mute|listen|"
             r"playlist|lyrics|sing)\b|turn (it |the music )?(up|down)",
    "home": r"\b(light|lights|lamp|bulb|bright|brightness|dim|dimmer|colou?r|aircon|air ?con|ac|"
            r"aircondition\w*|temperature|degrees|thermostat|cool(er)?|warm(er)?|fan|timer|timers|"
            r"alarm|alarms|wake me|remind|reminder|reminders|call|phone|dial|volume|louder|quieter|"
            r"mute|minutes?|seconds?|o'?clock)\b|\b\d{1,2}(:\d{2})?\s*(am|pm)\b|\b\d{1,2}:\d{2}\b",
    "build": r"\b(website|websites|web ?site|web ?page|landing page|site|app|apps|build|make me|"
             r"create|calculator|quiz|game|publish|github|deploy|html)\b",
    "files": r"\b(file|files|folder|folders|directory|notes?|readme|txt|md|py|json|csv|edit|"
             r"overwrite|save (it|this|that)|write (it|this|that|down)|search my|home folder|boots-files)\b",
    "shell": r"\b(command|terminal|disk|storage|space|git|process|processes|cpu|ram|uptime|ping|"
             r"python|script|calculate|compute|math|sum|average|convert|how many days|"
             r"\d+\s*[-+*/x^%]\s*\d+)\b",
    "library": r"\b(library|collection|collections|files-only|the files|my files|document|"
               r"documents|pdf|pdfs|docx|word file|download|downloads|thesis|paper|papers|"
               r"report|stop using|use my|check the web|sources?)\b",
}
_KW = {g: re.compile(p, re.I) for g, p in KEYWORDS.items()}

GROUP_HELP = ("home (lights, AC, timers, alarms, reminders, calls), music (Spotify, playback, "
              "volume), build (websites, apps, publishing), files (read/write/edit/search "
              "files), shell (system checks, run Python), library (the user's document library)")

TOOL_SPEC = ("load_tools",
             "Load more tools when the one you need is not in your tool list. groups: comma-"
             "separated, any of " + GROUP_HELP + ". Call it, then call the tool you need.",
             {"groups": "string"})


def build_groups(modules):
    """{group: set(tool names)} from the tool modules (keyed by module name)."""
    g = {k: set() for k in ("home", "music", "build", "files", "shell", "library", "core")}
    for mod in modules:
        names = {n for n, _d, _p in getattr(mod, "TOOL_SPECS", [])}
        mn = getattr(mod, "__name__", "")
        if mn.startswith(("me2_tools", "boots_ui_mcp", "me2_ui_mcp")):
            g["home"] |= names
            g["music"] |= names & _MUSIC_UI
        elif mn.startswith("spotify_tools"):
            g["music"] |= names
        elif mn.startswith("web_tools"):
            g["build"] |= names - CORE_NAMES
            g["core"] |= names & CORE_NAMES
        elif mn.startswith("files_tools"):
            g["files"] |= names
        elif mn.startswith("shell_tools"):
            g["shell"] |= names
        elif mn.startswith("library_tools"):
            g["library"] |= names
        elif mn.startswith("memory_tools"):
            g["core"] |= names
    g["core"].add("load_tools")
    return g


def parse_groups(text):
    """'music, home' -> {'music', 'home'} (unknown words ignored)."""
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    alias = {"lights": "home", "light": "home", "timer": "home", "alarm": "home", "reminder": "home",
             "spotify": "music", "song": "music", "website": "build", "app": "build",
             "file": "files", "python": "shell", "command": "shell", "documents": "library",
             "all": "all"}
    out = set()
    for w in words:
        w = alias.get(w, w)
        if w == "all":
            out |= set(KEYWORDS)
        elif w in KEYWORDS:
            out.add(w)
    return out


def pick_groups(texts, used_tools=(), groups_map=None, files_mode=False, extra=()):
    """Groups to load for this call.
    texts: the user's latest message and the one before it.
    used_tools: tool names called in the last two turns (they stay loaded).
    extra: groups the model asked for with load_tools this turn."""
    chosen = set(extra)
    for t in texts:
        for g, rx in _KW.items():
            if t and rx.search(t):
                chosen.add(g)
    if groups_map:
        for name in used_tools:
            for g, names in groups_map.items():
                if g != "core" and name in names:
                    chosen.add(g)
    if files_mode:
        chosen.add("library")
    return chosen


def tool_names(chosen, groups_map):
    names = set(groups_map.get("core", ()))
    for g in chosen:
        names |= groups_map.get(g, set())
    return names


def load_tools_result(groups_text):
    got = parse_groups(groups_text)
    if not got:
        return "No such tool group. Groups: " + GROUP_HELP + "."
    return ("Loaded: " + ", ".join(sorted(got)) + ". Those tools are in your list now; "
            "call the one you need.")
