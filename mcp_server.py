#!/usr/bin/env python3
"""
boots — the MERGED MCP server (one process, whole arsenal).

Exposes every tool from the three tool modules as MCP tools:
  * me2_tools.py  -> 16 Boots UI commands (music, lights, volume, timer,
                         alarm, temperature, reminder, call)
  * spotify_tools.py ->  7 Spotify controls
  * web_tools.py     ->  4 web tools (weather, search, fetch, write_website)

Transport: stdio (FastMCP default) -- a client on the same machine spawns
this process and talks MCP over stdin/stdout. This is the exact shape of
the final setup: server on Pi, client on Pi.

Run:
    python mcp_server.py          # stdio server (for a local client)

The tool IMPLEMENTATIONS stay in the modules (single source of truth);
this file is a thin facade over their dispatch() functions. The voice
agent does NOT use this server -- it calls dispatch() directly via
Pipecat FunctionSchemas (D11: Pipecat's MCPService is incompatible on
this stack). This server is for external clients (MCP inspector, Claude,
other agents) and for the classwork "build & run your MCP server on RPi"
requirement.
"""
import asyncio
import functools
import inspect

from mcp.server.fastmcp import FastMCP

import me2_tools as ui_mod
import spotify_tools as spotify_mod
import web_tools as web_mod
import files_tools as files_mod
import shell_tools as shell_mod

mcp = FastMCP("boots")

_MODULES = [ui_mod, spotify_mod, web_mod, files_mod, shell_mod]

def _make_tool(mod, name, desc, params):
    """Build a properly-typed function for FastMCP to introspect.
    Handles 0..N params (the old per-module wrappers only did 0..1)."""
    async def base(**kwargs):
        # dispatch() is sync (HTTP to :8330 / Spotify / web) -- offload so
        # it never blocks the MCP event loop.
        return await asyncio.to_thread(mod.dispatch, name, **kwargs)

    if not params:
        # NO functools.wraps here (2026-11-07): it set __wrapped__ = base, and
        # FastMCP's signature inspection followed it to base(**kwargs) -> a
        # REQUIRED "kwargs" field -> every zero-arg tool (play/pause/resume/
        # next/stop/volume up/down) failed validation on every voice call.
        async def fn():
            return await asyncio.to_thread(mod.dispatch, name)
        fn.__doc__ = desc
        fn.__signature__ = inspect.Signature([])
    else:
        # Build a closure that takes exactly the declared params. Map the
        # JSON-schema type to the Python annotation (str/int/float/bool).
        # array/object MUST map to list/dict (2026-11-07): they fell through
        # to str, so write_website's sections/stats/sources failed pydantic
        # validation ("Input should be a valid string") on EVERY voice call.
        _T = {"string": str, "integer": int, "number": float, "boolean": bool,
              "array": list, "object": dict}

        def _ann(pschema):
            t = pschema.get("type") if isinstance(pschema, dict) else pschema
            return _T.get(t, str)

        sig = inspect.Signature(
            [inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                               default=None, annotation=_ann(params[p]))
             for p in params]
        )

        def make(pnames):
            @functools.wraps(base)
            async def fn2(**kw):
                return await base(**{k: v for k, v in kw.items() if v is not None})
            fn2.__doc__ = desc
            fn2.__signature__ = sig
            return fn2

        fn = make(list(params))
    fn.__name__ = name
    return fn

for _mod in _MODULES:
    for _name, _desc, _params in _mod.TOOL_SPECS:
        mcp.add_tool(_make_tool(_mod, _name, _desc, _params),
                     name=_name, description=_desc)

if __name__ == "__main__":
    mcp.run()  # stdio transport
