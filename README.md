# cxoagi-vfx-mcp 🎬

A powerful video editing MCP (Model Context Protocol) server built with FastMCP and ffmpeg-python. This server allows LLMs to perform video editing operations through a standardized interface, enabling AI-powered video manipulation and processing workflows.

> This repository is the **CxOAGI fork** (`cxoagi-vfx-mcp`) of the project,
> maintained independently at https://github.com/CxOAGI/vfx-mcp for use in
> automated omni/veo stitching pipelines. The upstream PyPI package does not
> contain this fork's fixes — install `cxoagi-vfx-mcp` to get them.

Current version: **0.2.0**

## Features

### Core Video Operations
- **Trimming & Cutting**: Extract segments, fast (keyframe) or frame-accurate (`trim_video`)
- **Concatenation**: Join multiple videos end to end (`concatenate_videos`)
- **Still-to-video**: Turn a static image into a clip (`image_to_video`)
- **Format Conversion**: Transcode between containers/codecs (`convert_format`)
- **Resolution & Quality**: Resize by dimensions or scale factor (`resize_video`)
- **Audio Processing**: Extract, replace, mix, fade, adjust, and loudness-normalize audio
- **Effects & Filters**: Named filters, speed changes, thumbnails
- **Compositing**: Green-screen keying and motion blur
- **Analysis**: Get detailed video metadata (`get_video_info`)

### Stitching & Assembly
- **Transitions**: Stitch N clips with xfade/acrossfade transitions (`stitch_with_transitions`)
- **Manifest stitching**: Assemble a full deliverable from one ordered manifest of clips, trims, and transitions in a single call (`stitch_from_manifest`)
- **Loudness normalization**: Level audio to an EBU R128 target across clips (`normalize_loudness`)
- **Batch conversion**: Convert many files to a target container with bounded concurrency (`batch_convert`)

### MCP Capabilities
- **Tools**: Execute video editing operations
- **Resources**: Discover video files and read metadata
- **Context**: Progress reporting for long operations

## Installation

### From PyPI (Recommended)

```bash
# Install the CxOAGI fork
pip install cxoagi-vfx-mcp

# Run the server
vfx-mcp
```

### Using uv (Development)

```bash
# Clone the repository
git clone https://github.com/CxOAGI/vfx-mcp.git
cd vfx-mcp

# Install dependencies with uv
uv sync

# Run the server
uv run python main.py
# or via the console script
uv run vfx-mcp
```

### Using Nix

```bash
# Enter the development shell
nix develop

# Run the server
python main.py
```

### System Requirements

- Python 3.13+
- FFmpeg (installed automatically with Nix, or install manually)
- uv package manager (for non-Nix installation)

## Quick Start

### Basic Usage

```python
# Connect to the VFX MCP server
from fastmcp import Client

async with Client("python main.py") as client:
    # Trim a video
    result = await client.call_tool("trim_video", {
        "input_path": "input.mp4",
        "output_path": "trimmed.mp4",
        "start_time": 10.0,
        "duration": 30.0
    })

    # Get video information
    info = await client.call_tool("get_video_info", {
        "video_path": "input.mp4"
    })
    print(info)
```

### CLI Usage with Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "vfx": {
      "command": "vfx-mcp",
      "args": []
    }
  }
}
```

Or if using the development version:

```json
{
  "mcpServers": {
    "vfx": {
      "command": "uv",
      "args": ["run", "python", "/path/to/vfx-mcp/main.py"],
      "cwd": "/path/to/vfx-mcp"
    }
  }
}
```

## API Reference

### Basic Video Tools

#### `trim_video`
Extract a segment from a video.

**Parameters:**
- `input_path` (str): Path to input video file
- `output_path` (str): Path for output video file
- `start_time` (float): Start time in seconds
- `duration` (float, optional): Duration in seconds (if not specified, trims to end)
- `accurate` (bool, optional): Frame-accurate cut via re-encode (default `false`, which fast-copies and snaps to the nearest keyframe)
- `crf` / `preset` (optional): Quality controls for the accurate re-encode

#### `concatenate_videos`
Join multiple videos together into a single continuous video. Silent clips and
mismatched resolutions/frame rates are handled automatically; homogeneous inputs
use a lossless stream-copy fast path.

**Parameters:**
- `input_paths` (list[str]): List of video file paths to concatenate (minimum 2)
- `output_path` (str): Path for output video file

> Note: `concatenate_videos` does a straight join and does not apply
> transitions. For crossfades/dissolves between clips, use
> `stitch_with_transitions` (see below).

#### `image_to_video`
Create a video from a static image for a specified duration.

**Parameters:**
- `image_path` (str): Path to the source image
- `output_path` (str): Path for output video file
- `duration` (float): Length of the output clip in seconds
- `framerate` (int, optional): Output frame rate (default: 30)

#### `resize_video`
Change video resolution.

**Parameters:**
- `input_path` (str): Path to input video file
- `output_path` (str): Path for output video file
- `width` (int, optional): Target width (maintains aspect ratio if height not specified)
- `height` (int, optional): Target height (maintains aspect ratio if width not specified)
- `scale` (float, optional): Scale factor (e.g., 0.5 for half size)

#### `get_video_info`
Get detailed video metadata.

**Parameters:**
- `video_path` (str): Path to video file

**Returns:**
- Video metadata including duration, resolution, codec, bitrate, fps, etc.

### Format Conversion

#### `convert_format`
Convert video to a different format or codec.

**Parameters:**
- `input_path` (str): Path to input video file
- `output_path` (str): Path for output video file
- `format` (str, optional): Output container (`mp4`, `avi`, `mkv`, `webm`, `mov`). When set, codecs are auto-selected for that container.
- `video_codec` (str, optional): Video codec (default `libx264`; e.g. `libx265`, `libvpx-vp9`)
- `audio_codec` (str, optional): Audio codec (default `aac`; e.g. `mp3`, `libvorbis`)
- `video_bitrate` (str, optional): Video bitrate (e.g. `2.5M`)
- `audio_bitrate` (str, optional): Audio bitrate (default `128k`)

### Audio Tools

#### `extract_audio`
Extract the audio track from a video.

**Parameters:**
- `input_path` (str): Path to input video file
- `output_path` (str): Path for output audio file

#### `add_audio`
Add or replace the audio track in a video.

**Parameters:**
- `input_path` (str): Path to input video file
- `audio_path` (str): Path to audio file
- `output_path` (str): Path for output video file
- `replace` (bool, optional): Replace existing audio (default: `true`) or mix with it (`false`)
- `audio_volume` (float, optional): Volume level for the new audio, 0.0–2.0 (default: 1.0)

#### `adjust_audio_volume`
Change the volume of a clip's audio track.

#### `mix_audio`
Mix an additional audio track into a video/audio file.

#### `audio_fade_in` / `audio_fade_out`
Apply an audio fade at the start or end of a clip.

### Effects & Filters

#### `apply_filter`
Apply a named visual filter to a video.

**Parameters:**
- `input_path` (str): Path to input video file
- `output_path` (str): Path for output video file
- `filter` (str): One of `brightness`, `contrast`, `saturation`, `hflip`, `grayscale`, `sepia`, `blur`, `sharpen`, `vintage`, or `scale=WIDTHxHEIGHT`
- `strength` (float, optional): Filter intensity, 0.1–3.0 (default: 1.0; ignored by `hflip`)

#### `change_speed`
Adjust video playback speed (audio is retimed when present; silent clips are handled).

**Parameters:**
- `input_path` (str): Path to input video file
- `output_path` (str): Path for output video file
- `speed` (float): Speed multiplier, 0.1–10.0 (e.g., 2.0 for double speed, 0.5 for half)

#### `generate_thumbnail`
Extract a frame as an image thumbnail.

**Parameters:**
- `input_path` (str): Path to input video file
- `output_path` (str): Path for output image file
- `timestamp` (float, optional): Time in seconds to grab (default: 2.5)
- `width` / `height` (int, optional): Thumbnail dimensions (aspect ratio preserved when only one is given)

### Compositing

#### `create_green_screen_effect`
Chroma-key a green/blue screen and composite over a new background.

#### `apply_motion_blur`
Apply a motion-blur effect to a video.

### Stitching & Batch

#### `stitch_with_transitions`
Stitch multiple clips together with crossfade transitions. Clips are normalized
to a common resolution/fps/pixel format first, and silent clips are backed with
silence so the audio crossfade always succeeds.

**Parameters:**
- `input_paths` (list[str]): Clips to stitch, in order (minimum 2)
- `output_path` (str): Path for the stitched video
- `transition` (str, optional): One of `fade`, `crossfade`, `dissolve`, `wipe_left`, `wipe_right`, `wipe_up`, `wipe_down`, `slide_left`, `slide_right` (default: `fade`)
- `duration` (float, optional): Transition length in seconds; must be shorter than every clip (default: 1.0)
- `crf` / `preset` (optional): Encoder quality controls
- `faststart` (bool, optional): Add `+faststart` for progressive playback (default: `true`)

Output duration ≈ `sum(clip_durations) - (n - 1) * duration`.

#### `stitch_from_manifest`
Assemble a finished deliverable from one ordered manifest — trims, cuts, and
crossfades — in a single call.

**Parameters:**
- `manifest` (list[dict]): Ordered clip descriptors. Each entry: `clip` (str, required), `start`/`end` (float, optional trim points), `transition` (str, optional — omit/null for a plain cut), `transition_duration` (float, optional, default 1.0)
- `output_path` (str): Path for the stitched video
- `crf` / `preset` / `faststart` (optional): Encoder quality controls

#### `normalize_loudness`
Normalize perceived loudness (EBU R128 `loudnorm`) so clips from different
sources sound level. Video is stream-copied; audio-only inputs are supported.

**Parameters:**
- `input_path` (str) / `output_path` (str)
- `target_i` (float, optional): Integrated loudness in LUFS (default: -14.0)
- `target_tp` (float, optional): Max true peak in dBTP (default: -1.0)
- `target_lra` (float, optional): Loudness range in LU (default: 11.0)

#### `batch_convert`
Convert many videos to a target container concurrently (bounded by
`VFX_MAX_CONCURRENCY`).

**Parameters:**
- `input_paths` (list[str]): Source videos
- `output_dir` (str): Directory for the converted files (created if needed)
- `format` (str, optional): Target container, e.g. `mp4`, `mkv`, `webm` (default: `mp4`)
- `crf` / `preset` (optional): Encoder quality controls

### Resource Endpoints

#### `videos://list`
List available video files.

#### `videos://{filename}/metadata`
Get metadata for a specific video file.

#### `tools://advanced/{category}`
Describe advanced tool capabilities by category.

## Roadmap

The following modules are still stubs and register no tools yet:

- **Text & graphics** (`text_animation.py`) — animated text/overlays.
- **Video analysis** (`video_analysis.py`) — scene detection, dominant colors,
  and statistics extraction.

## Examples

### Create a Video Montage

```python
async with Client("python main.py") as client:
    # 1. Trim clips from source videos
    clips = []
    for i, (video, start, duration) in enumerate([
        ("vacation.mp4", 30, 5),
        ("birthday.mp4", 120, 8),
        ("concert.mp4", 45, 6)
    ]):
        clip_path = f"clip_{i}.mp4"
        await client.call_tool("trim_video", {
            "input_path": video,
            "output_path": clip_path,
            "start_time": start,
            "duration": duration
        })
        clips.append(clip_path)

    # 2. Concatenate clips into one video
    await client.call_tool("concatenate_videos", {
        "input_paths": clips,
        "output_path": "montage.mp4"
    })

    # 3. Add background music
    await client.call_tool("add_audio", {
        "input_path": "montage.mp4",
        "audio_path": "background_music.mp3",
        "output_path": "final_montage.mp4"
    })
```

### Process Video for Web

```python
async with Client("python main.py") as client:
    # Convert to a web-friendly format
    await client.call_tool("convert_format", {
        "input_path": "raw_video.mov",
        "output_path": "web_video.mp4",
        "format": "mp4"
    })

    # Create multiple resolutions
    for width in [1920, 1280, 854]:
        await client.call_tool("resize_video", {
            "input_path": "web_video.mp4",
            "output_path": f"web_video_{width}.mp4",
            "width": width
        })

    # Generate thumbnail
    await client.call_tool("generate_thumbnail", {
        "video_path": "web_video.mp4",
        "output_path": "thumbnail.jpg"
    })
```

## Architecture

### Project Structure

```
vfx-mcp/
├── README.md                    # This file
├── flake.nix                    # Nix development environment
├── pyproject.toml               # Python project configuration
├── uv.lock                      # Locked dependencies
├── Dockerfile                   # Container image (python:3.13-slim + ffmpeg)
├── main.py                      # Development entry point (runs the server)
├── src/
│   └── vfx_mcp/
│       ├── __init__.py
│       ├── core/                # Shared infrastructure
│       │   ├── server.py        # create_mcp_server()/main(): builds + registers everything
│       │   ├── utilities.py     # run_ffmpeg_async, error handling, metadata, output builder
│       │   ├── media.py         # shared stitching/codec helpers (normalize, xfade map, ...)
│       │   └── validation.py    # workspace sandbox + argument/path validation
│       ├── tools/               # Tool implementations grouped by domain
│       │   ├── basic_video_ops.py     # trim, resize, concat, image_to_video, info
│       │   ├── audio_processing.py    # extract/add/mix/fade/volume/normalize_loudness
│       │   ├── video_effects.py       # apply_filter, change_speed, thumbnail
│       │   ├── format_conversion.py   # convert_format
│       │   ├── advanced_compositing.py# green screen, motion blur
│       │   ├── video_transitions.py   # stitch_with_transitions
│       │   ├── batch_automation.py    # stitch_from_manifest, batch_convert
│       │   ├── text_animation.py      # (stub)
│       │   └── video_analysis.py      # (stub)
│       └── resources/
│           └── mcp_endpoints.py # MCP resource endpoints
└── tests/                       # pytest suite + fixtures (conftest.py)
```

### Key Components

1. **FastMCP Server**: `core/server.create_mcp_server()` builds the server and
   calls each module's `register_*_tools(mcp)` function; `main()` runs it under
   the transport chosen by `MCP_TRANSPORT`.
2. **Tool Modules**: Each `tools/*.py` module registers its `@mcp.tool` functions.
3. **FFmpeg Integration**: ffmpeg-python graphs are executed via
   `run_ffmpeg_async` (off the event loop, with a concurrency cap, timeout, and
   partial-output cleanup on failure/cancel).
4. **Shared Helpers**: `log_operation`, `handle_ffmpeg_error`,
   `create_standard_output`, `run_ffmpeg_async` (`core/utilities.py`), plus
   stitching/codec helpers in `core/media.py`.
5. **Sandbox**: `core/validation.py` resolves and contains all input/output
   paths within `VFX_WORKSPACE` (opt-in) and rejects URL/protocol inputs.

### Environment Variables

- `MCP_TRANSPORT` / `MCP_HOST` / `MCP_PORT`: transport selection (`stdio` default; `sse`/`http` bind host/port).
- `VFX_WORKSPACE`: sandbox root; when set, all paths are contained within it.
- `VFX_FFMPEG_TIMEOUT`: per-operation ffmpeg timeout in seconds (default: none).
- `VFX_MAX_CONCURRENCY`: max concurrent ffmpeg processes (default: 3).

## Development

### Setting Up Development Environment

#### With Nix (Recommended for consistent environment)

```bash
# Enter development shell with all dependencies
nix develop

# Run tests
pytest

# Run linting
ruff check .

# Format code
ruff format .
```

#### With uv

```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Run linting
uv run ruff check .

# Format code
uv run ruff format .
```

### Adding New Tools

1. Add a nested `async def` with the `@mcp.tool` decorator inside the appropriate
   `register_<category>_tools(mcp)` function under `src/vfx_mcp/tools/` (or create
   a new module and wire its register function into `create_mcp_server()`).
2. Add proper type hints and a Google-style docstring.
3. Validate/resolve paths with `safe_input_path` / `safe_output_path`, build the
   graph with `create_standard_output`, and execute it with `run_ffmpeg_async`
   (which handles errors, timeouts, concurrency, and partial-output cleanup — do
   not call `ffmpeg.run()` directly).
4. Add corresponding tests.

Example:

```python
def register_example_tools(mcp: FastMCP[None]) -> None:
    @mcp.tool
    async def rotate_video(
        input_path: str,
        output_path: str,
        angle: int,
        ctx: Context | None = None,
    ) -> str:
        """Rotate video by 90, 180, or 270 degrees."""
        if angle not in (90, 180, 270):
            raise ValueError("Angle must be 90, 180, or 270 degrees")

        resolved_input = safe_input_path(input_path)
        resolved_output = safe_output_path(output_path)

        await log_operation(ctx, f"Rotating video by {angle} degrees...")
        stream = ffmpeg.input(str(resolved_input))
        stream = ffmpeg.filter(stream, "rotate", angle=math.radians(angle))
        output = create_standard_output(stream, str(resolved_output))
        await run_ffmpeg_async(output, ctx=ctx, output_path=str(resolved_output))
        return f"Video rotated and saved to {resolved_output}"
```

### Testing

Run the test suite:

```bash
# All tests
pytest

# Specific test file
pytest tests/test_basic_operations.py

# With coverage
pytest --cov=src
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-tool`)
3. Make your changes and add tests
4. Run linting and tests
5. Commit your changes (`git commit -m 'Add amazing tool'`)
6. Push to the branch (`git push origin feature/amazing-tool`)
7. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [FastMCP](https://github.com/jlowin/fastmcp) - The fast, Pythonic MCP framework
- Powered by [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) - Python bindings for FFmpeg
- Uses [Model Context Protocol](https://modelcontextprotocol.io) - Standard for LLM integrations
