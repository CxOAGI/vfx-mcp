"""FastMCP server setup and tool registration.

This module creates and configures the VFX MCP server with comprehensive
video editing capabilities. Registers all available tool modules and
resource endpoints to provide a complete video processing API.

The server provides:
    - Basic video operations (trim, resize, concatenate)
    - Advanced video effects (filters, speed changes)
    - Audio processing (extraction, mixing, enhancement)
    - Format conversion (codecs, containers)
    - Compositing and automation tools
    - MCP resource endpoints for file discovery

Example:
    Create and start the server:

        server = create_mcp_server()
        # Server is ready to handle MCP requests
"""

import os

from fastmcp import FastMCP


def create_mcp_server() -> FastMCP[None]:
    """Create and configure the VFX MCP server with all tools registered.

    Initializes a FastMCP server instance and registers all available video
    editing tools and resource endpoints. The resulting server provides a
    comprehensive API for video processing operations.

    Returns:
        FastMCP[None]: Configured server instance with all tools registered.

    Example:
        server = create_mcp_server()
        # Server ready to handle video editing requests
    """
    mcp: FastMCP[None] = FastMCP("vfx-mcp")

    # Import and register all tool modules
    from ..resources.mcp_endpoints import register_resource_endpoints
    from ..tools.advanced_compositing import (
        register_compositing_tools,
    )
    from ..tools.audio_processing import (
        register_audio_tools,
    )
    from ..tools.basic_video_ops import (
        register_basic_video_tools,
    )
    from ..tools.batch_automation import (
        register_automation_tools,
    )
    from ..tools.format_conversion import (
        register_format_conversion_tools,
    )
    from ..tools.text_animation import (
        register_animation_tools,
    )
    from ..tools.video_analysis import (
        register_analysis_tools,
    )
    from ..tools.video_effects import (
        register_video_effects_tools,
    )
    from ..tools.video_transitions import (
        register_transition_tools,
    )

    # Register all tool categories
    register_basic_video_tools(mcp)
    register_audio_tools(mcp)
    register_video_effects_tools(mcp)
    register_format_conversion_tools(mcp)
    register_compositing_tools(mcp)
    register_transition_tools(mcp)
    register_animation_tools(mcp)
    register_automation_tools(mcp)
    register_analysis_tools(mcp)
    register_resource_endpoints(mcp)

    return mcp


def main() -> None:
    """Console entry point for the VFX MCP server.

    Creates the configured server and runs it using the transport selected via
    environment variables. This is the target of the ``vfx-mcp`` console script
    declared in ``pyproject.toml``.

    Environment Variables:
        MCP_TRANSPORT: Transport protocol to use. One of ``stdio`` (default),
            ``sse``, ``http`` (alias for ``streamable-http``), or
            ``streamable-http``.
        MCP_HOST: Host interface to bind when using an HTTP-based transport.
            Defaults to ``127.0.0.1``. Ignored for ``stdio``.
        MCP_PORT: TCP port to bind when using an HTTP-based transport.
            Defaults to ``8000``. Ignored for ``stdio``.

    Raises:
        ValueError: If ``MCP_TRANSPORT`` is not a recognized transport or if
            ``MCP_PORT`` is not a valid integer.
    """
    server = create_mcp_server()

    transport = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()

    if transport == "stdio":
        server.run(transport="stdio")
        return

    # Map the documented "http" alias onto FastMCP's streamable-http transport.
    if transport == "http":
        transport = "streamable-http"

    if transport not in {"sse", "streamable-http"}:
        raise ValueError(
            f"Unknown MCP_TRANSPORT {transport!r}; expected one of "
            "'stdio', 'sse', 'http', or 'streamable-http'."
        )

    host = os.environ.get("MCP_HOST", "127.0.0.1")
    port_str = os.environ.get("MCP_PORT", "8000")
    try:
        port = int(port_str)
    except ValueError as exc:
        raise ValueError(
            f"Invalid MCP_PORT {port_str!r}; expected an integer."
        ) from exc

    server.run(transport=transport, host=host, port=port)
