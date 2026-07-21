# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `cxoagi-vfx-mcp`, a video editing MCP (Model Context Protocol) server built
with FastMCP and ffmpeg-python. The server enables LLMs to perform professional
video editing operations through a standardized interface, providing AI-powered
video manipulation and processing workflows.

This repository is the **CxOAGI fork** of the project
(https://github.com/CxOAGI/vfx-mcp), maintained independently for use in
automated omni/veo stitching pipelines. Install with `pip install cxoagi-vfx-mcp`.

## Key Commands

### Development and Testing

```bash
# Install from PyPI
pip install cxoagi-vfx-mcp

# Run the MCP server
vfx-mcp

# Or for development:
# Enter Nix development environment (includes all dependencies)
nix develop

# Install Python dependencies (if not using Nix)
uv sync

# Run the MCP server (development entry point)
uv run python main.py
# or via the console script
uv run vfx-mcp

# Run tests
pytest
# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Run with coverage
pytest --cov=src

# Format code
ruff format .

# Run linting
ruff check .

# Type checking (basedpyright strict is the project standard)
basedpyright src/
# mypy is also configured
mypy main.py src/
```

### Nix Environment Commands

When in the Nix development shell, additional commands are available:

```bash
dx       # Edit flake.nix
tests    # Run all tests (wrapper command)
run      # Run with hot reloading using air
```

## Architecture

### Package Layout

The code is organized as an installable package under `src/vfx_mcp/`. `main.py`
at the repo root is a thin development entry point that imports and runs the
server. The real structure is modular:

```
src/vfx_mcp/
├── __init__.py
├── core/                    # Shared infrastructure
│   ├── server.py            # create_mcp_server(): builds the FastMCP instance
│   │                        #   and registers every tool/resource module
│   ├── utilities.py         # handle_ffmpeg_error, log_operation,
│   │                        #   get_video_metadata, create_standard_output,
│   │                        #   parse_color/parse_resolution helpers
│   └── validation.py        # validate_range, validate_file_path,
│                            #   validate_output_path, validate_video_paths, etc.
├── tools/                   # Tool implementations, grouped by domain.
│   │                        #   Each module exposes a register_*_tools(mcp)
│   │                        #   function that decorates its tools with @mcp.tool.
│   ├── basic_video_ops.py   # trim_video, get_video_info, resize_video,
│   │                        #   concatenate_videos, image_to_video
│   ├── audio_processing.py  # extract_audio, add_audio, adjust_audio_volume,
│   │                        #   mix_audio, audio_fade_in, audio_fade_out
│   ├── video_effects.py     # apply_filter, change_speed, generate_thumbnail
│   ├── format_conversion.py # convert_format
│   ├── advanced_compositing.py  # create_green_screen_effect, apply_motion_blur
│   ├── video_transitions.py # register_transition_tools (WIP: stitch_with_transitions)
│   ├── text_animation.py    # register_animation_tools (stub)
│   ├── video_analysis.py    # register_analysis_tools (stub)
│   └── batch_automation.py  # register_automation_tools (stub)
└── resources/
    └── mcp_endpoints.py     # register_resource_endpoints(mcp): resource endpoints
```

**Registration pattern**: Tools are NOT decorated at module scope. Instead each
`tools/*.py` module defines nested `@mcp.tool async def` functions inside a
`register_<category>_tools(mcp)` function. `core/server.create_mcp_server()`
imports and calls each registration function to build the server. When adding a
tool, add it inside the appropriate `register_*` function (or create a new module
with its own registration function and wire it into `create_mcp_server()`).

**Registered tools** (currently implemented):
- **Basic Operations**: `trim_video`, `get_video_info`, `resize_video`,
  `concatenate_videos`, `image_to_video`
- **Audio Processing**: `extract_audio`, `add_audio` (replace or mix modes),
  `adjust_audio_volume`, `mix_audio`, `audio_fade_in`, `audio_fade_out`
- **Effects & Filters**: `apply_filter`, `change_speed`, `generate_thumbnail`
- **Format Conversion**: `convert_format` (codec/bitrate control)
- **Compositing**: `create_green_screen_effect`, `apply_motion_blur`

**Stubs / work in progress**: The `video_transitions`, `text_animation`,
`video_analysis`, and `batch_automation` modules currently register no tools (or
placeholder registration only). A `stitch_with_transitions` xfade-based stitcher
is planned for `video_transitions.py`. Do not document stub tools as available.

**Resource Endpoints**: MCP resources for file discovery and metadata (in
`resources/mcp_endpoints.py`):
- `videos://list` - Lists available video files
- `videos://{filename}/metadata` - Returns detailed video metadata as JSON
- `tools://advanced/{category}` - Describes advanced tool capabilities

### FFmpeg Integration

All video operations use `ffmpeg-python` for robust video processing:
- **Error Handling**: `handle_ffmpeg_error` catches `ffmpeg.Error`, surfaces
  decoded stderr, and logs via the MCP context.
- **Progress / Logging**: `log_operation(ctx, message)` reports status through
  the optional MCP context.
- **Standard output**: `create_standard_output(stream, output_path, **kwargs)`
  centralizes container/codec defaults.
- **Efficiency**: Uses copy mode for operations that don't require re-encoding.

### Testing Infrastructure

**Fixtures** (`tests/conftest.py`):
- `sample_video`: Generates a short H.264/AAC test video with color bars and a tone
- `sample_videos`: Creates multiple test videos for concatenation tests
- `sample_audio`: Generates an MP3 audio file for audio processing tests
- `temp_dir`: Managed temporary directory for test file isolation
- `mcp_server`: A configured server instance for `fastmcp.Client`-based tests

**Test Categories**:
- `@pytest.mark.unit` - Fast unit tests for individual functions
- `@pytest.mark.integration` - Tests involving actual video processing
- `@pytest.mark.slow` - Performance tests that may take longer to execute

### Development Environment

**Nix Flake Setup**: Provides consistent development environment with:
- Python 3.13 with uv package manager
- FFmpeg with full codec support (`ffmpeg-full`)
- Development tools (ruff, basedpyright)
- Optional multimedia tools (imagemagick, sox)

**Environment Variables**:
- `FFMPEG_PATH` and `FFPROBE_PATH` - Automatically set in Nix environment
- `MCP_TRANSPORT` - Server transport mode (`stdio` or `sse`)
- `MCP_HOST` and `MCP_PORT` - SSE transport configuration
- `VFX_WORKSPACE` - Root directory that input/output paths are resolved against
  and contained within (sandbox root for pipeline deployments)
- `VFX_FFMPEG_TIMEOUT` - Per-call timeout (seconds) for ffmpeg invocations
- `VFX_MAX_CONCURRENCY` - Maximum number of concurrent ffmpeg operations

> Note: `MCP_TRANSPORT`/`MCP_HOST`/`MCP_PORT` and the `VFX_*` sandbox/execution
> variables are wired up as part of the pipeline-hardening work; confirm the
> value is actually read by the current code before relying on it.

## Common Patterns

### Adding New Video Tools

1. Add a nested `async def` with the `@mcp.tool` decorator inside the relevant
   `register_<category>_tools(mcp)` function (or create a new module + register
   function and wire it into `create_mcp_server()`).
2. Include comprehensive type hints and a Google-style docstring.
3. Use the optional `ctx: Context | None = None` parameter for progress reporting.
4. Use the shared helpers: `log_operation`, `create_standard_output`, and
   `handle_ffmpeg_error`; validate arguments with `core/validation.py` helpers.
5. Add corresponding tests in the appropriate test file.

Example pattern:
```python
def register_example_tools(mcp: FastMCP[None]) -> None:
    @mcp.tool
    async def new_operation(
        input_path: str,
        output_path: str,
        parameter: float,
        ctx: Context | None = None,
    ) -> str:
        """Operation description."""
        await log_operation(ctx, "Starting operation...")
        try:
            stream = ffmpeg.input(input_path)
            # Apply transformations
            output = create_standard_output(stream, output_path)
            ffmpeg.run(output, overwrite_output=True)
            return f"Operation completed: {output_path}"
        except ffmpeg.Error as e:
            await handle_ffmpeg_error(e, ctx)
            raise
```

### Speed Optimization Considerations

- For trimming operations, use `c="copy"` to avoid re-encoding.
- Chain multiple atempo filters for speeds > 2.0x (ffmpeg limitation).
- Use appropriate presets (`ultrafast` for testing, `medium` for production).
- Consider file size vs quality tradeoffs when setting bitrates.

### MCP Client Integration

Server supports stdio (and, once wired, SSE) transports for different
integration scenarios:
- **Claude Desktop**: Uses stdio transport with `vfx-mcp` (PyPI) or
  `uv run python main.py` (dev).
- **Web Applications**: Can use SSE transport with host/port configuration.
- **Direct API**: FastMCP client library for programmatic access.

**PyPI Installation**: The fork is published as `cxoagi-vfx-mcp`. Install the
fork's fixes with `pip install cxoagi-vfx-mcp` (the upstream `vfx-mcp` package
does **not** contain this fork's changes).
