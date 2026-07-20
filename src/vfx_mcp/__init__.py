"""VFX MCP Server - Professional video editing tools via Model Context Protocol.

A comprehensive video editing server built on FastMCP framework, providing
professional-grade video processing capabilities through ffmpeg-python bindings.
"""

__version__ = "0.2.0"
__author__ = "CxOAGI"

from .core.server import create_mcp_server, main

__all__ = ["create_mcp_server", "main"]
