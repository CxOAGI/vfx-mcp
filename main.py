#!/usr/bin/env python3
"""VFX MCP Server - Video editing server using FastMCP and ffmpeg-python.

This module is a thin development entry point. The real server setup and the
console entry point both live in :mod:`vfx_mcp.core.server`; this file simply
exposes them for `uv run python main.py` during development, where the
``vfx_mcp`` package is not pip-installed and therefore not importable without a
small path adjustment.

Typical usage example:
    $ uv run python main.py
    # Server starts using the transport selected by MCP_TRANSPORT (stdio by
    # default). See vfx_mcp.core.server.main for the supported env vars.

In an installed environment prefer the ``vfx-mcp`` console script, which calls
:func:`vfx_mcp.core.server.main` directly.
"""

import sys
from pathlib import Path

# In development the package lives under ./src and is not pip-installed, so make
# it importable. When vfx-mcp is installed as a package this insert is harmless.
_SRC_PATH = Path(__file__).parent / "src"
if _SRC_PATH.is_dir():
    sys.path.insert(0, str(_SRC_PATH))

from vfx_mcp import create_mcp_server, main  # noqa: E402

# Initialize the MCP server (exposed for testing).
mcp = create_mcp_server()

# Run the server when called directly, honoring MCP_TRANSPORT/HOST/PORT.
if __name__ == "__main__":
    main()
