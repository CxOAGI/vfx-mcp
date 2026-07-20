"""Basic video editing operations: trim, resize, concatenate, and info.

This module provides fundamental video editing tools for trimming segments,
resizing videos, concatenating multiple files, and retrieving video metadata.
All operations use FFmpeg for processing and include comprehensive error handling.

Example:
    Register tools with MCP server:

        mcp = FastMCP('video-editor')
        register_basic_video_tools(mcp)
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import ffmpeg
from fastmcp import Context, FastMCP

from ..core import (
    create_standard_output,
    even_dimension,
    get_video_metadata,
    log_operation,
    normalize_audio_stream,
    normalize_video_stream,
    run_ffmpeg_async,
    safe_input_path,
    safe_output_path,
    silent_audio_source,
    validate_range,
)
from ..core.utilities import VideoMetadata

# Fallback geometry used only when no input could be probed for dimensions.
_FALLBACK_WIDTH = 1280
_FALLBACK_HEIGHT = 720
_FALLBACK_FPS = 30.0

# Containers for which ``-movflags +faststart`` is meaningful (relocates the
# moov atom for progressive playback). Applying it to other containers errors.
_FASTSTART_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".m4a"})


def _wants_faststart(output_path: Path) -> bool:
    """Return whether ``-movflags +faststart`` applies to this container."""
    return output_path.suffix.lower() in _FASTSTART_SUFFIXES


def _inputs_homogeneous(metas: list[VideoMetadata]) -> bool:
    """Return ``True`` when every input shares codec/geometry/fps/pix_fmt.

    Homogeneous inputs can be joined with the concat *demuxer* and ``-c copy``
    for a lossless, near-instant stitch. Audio must also match: either every
    input is silent, or all carry an audio stream with the same codec, sample
    rate and channel count. A mismatch on any dimension forces the re-encode
    path.
    """
    first = metas[0]
    if "video" not in first:
        return False

    fv = first["video"]
    for meta in metas[1:]:
        if "video" not in meta:
            return False
        v = meta["video"]
        if (
            v["codec"] != fv["codec"]
            or v["width"] != fv["width"]
            or v["height"] != fv["height"]
            or v["pixel_format"] != fv["pixel_format"]
            or round(v["fps"], 2) != round(fv["fps"], 2)
        ):
            return False

    # Audio must be uniformly absent or uniformly matching.
    audio_present = ["audio" in m for m in metas]
    if any(audio_present) != all(audio_present):
        return False
    if all(audio_present):
        fa = first["audio"]
        for meta in metas[1:]:
            a = meta["audio"]
            if (
                a["codec"] != fa["codec"]
                or a["sample_rate"] != fa["sample_rate"]
                or a["channels"] != fa["channels"]
            ):
                return False
    return True


def _target_geometry(metas: list[VideoMetadata]) -> tuple[int, int, float]:
    """Compute a common (even width, even height, fps) for normalization.

    Uses the largest width/height and fastest frame rate across all probed
    inputs so no source is upscaled beyond the biggest clip, falling back to a
    720p/30fps default only when no video stream could be read.
    """
    widths = [m["video"]["width"] for m in metas if "video" in m]
    heights = [m["video"]["height"] for m in metas if "video" in m]
    rates = [m["video"]["fps"] for m in metas if "video" in m and m["video"]["fps"] > 0]

    width = even_dimension(max(widths)) if widths else _FALLBACK_WIDTH
    height = even_dimension(max(heights)) if heights else _FALLBACK_HEIGHT
    fps = max(rates) if rates else _FALLBACK_FPS
    return width, height, fps


def _write_concat_list(inputs: list[Path]) -> str:
    """Write an ffmpeg concat-demuxer list file and return its path.

    Each line is ``file '<absolute path>'`` with single quotes escaped per the
    concat demuxer's quoting rules. The caller is responsible for deleting the
    returned temp file.
    """
    handle, list_path = tempfile.mkstemp(suffix=".txt", prefix="vfx_concat_")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            for path in inputs:
                escaped = str(path).replace("'", "'\\''")
                fh.write(f"file '{escaped}'\n")
    except Exception:
        os.unlink(list_path)
        raise
    return list_path


def register_basic_video_tools(
    mcp: FastMCP[object],
) -> None:
    """Register basic video editing tools with the MCP server.

    Adds fundamental video editing operations including trim, resize,
    concatenate, get_video_info, and image_to_video functions to the
    provided FastMCP server instance.

    Args:
        mcp: The FastMCP server instance to register tools with.

    Returns:
        None
    """

    @mcp.tool
    async def trim_video(
        input_path: str,
        output_path: str,
        start_time: float,
        duration: float | None = None,
        accurate: bool = False,
        crf: int | None = None,
        preset: str | None = None,
        ctx: Context | None = None,
    ) -> str:
        """Extract a segment from a video.

        Extracts a portion of a video file starting at the specified time.
        If duration is not provided (``None``), extracts from ``start_time`` to
        the end of the video.

        Two cutting modes are available:
            - Fast (``accurate=False``, default): stream-copies with ``-c copy``
              for near-instant, lossless trimming. Cuts snap to the nearest
              preceding keyframe, so the clip may begin slightly before
              ``start_time`` (up to one GOP, typically ~2s).
            - Accurate (``accurate=True``): re-encodes so the cut lands on the
              exact requested frame. Slower and lossy, but frame-precise —
              required for tight stitching workflows.

        Args:
            input_path: Path to the input video file.
            output_path: Path where the trimmed video will be saved.
            start_time: Start time in seconds (must be >= 0).
            duration: Duration in seconds to extract (must be > 0 when given).
                If ``None``, extracts to the end of the video.
            accurate: When True, re-encode for a frame-accurate cut instead of
                the fast keyframe-aligned copy.
            crf: Constant Rate Factor for the accurate re-encode (lower = higher
                quality). Ignored in fast copy mode.
            preset: libx264 speed/quality preset for the accurate re-encode.
                Ignored in fast copy mode.
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating the video was trimmed and saved.

        Raises:
            ValueError: If ``start_time`` is negative or ``duration`` <= 0.
            RuntimeError: If ffmpeg encounters an error during processing.
        """
        if start_time < 0:
            raise ValueError("start_time must be >= 0")
        if duration is not None and duration <= 0:
            raise ValueError("duration must be greater than 0")

        resolved_input = safe_input_path(input_path)
        resolved_output = safe_output_path(output_path)

        mode = "accurate (re-encode)" if accurate else "fast (copy)"
        await log_operation(
            ctx,
            f"Trimming video from {start_time}s"
            + (f" for {duration}s" if duration is not None else " to end")
            + f" [{mode}]",
        )

        stream = ffmpeg.input(str(resolved_input), ss=start_time)
        extra: dict[str, float] = {}
        if duration is not None:
            extra["t"] = duration

        if accurate:
            output = create_standard_output(
                stream,
                str(resolved_output),
                crf=crf,
                preset=preset,
                faststart=_wants_faststart(resolved_output),
                **extra,
            )
        else:
            output = ffmpeg.output(
                stream,
                str(resolved_output),
                c="copy",
                **extra,
            )

        await run_ffmpeg_async(output, ctx=ctx, output_path=str(resolved_output))
        return f"Video trimmed successfully and saved to {resolved_output}"

    # Ensure function is registered with MCP
    del trim_video

    @mcp.tool
    async def get_video_info(
        video_path: str,
    ) -> VideoMetadata:
        """Get detailed video metadata.

        Analyzes a video file and extracts comprehensive metadata including
        format information, video stream properties, and audio stream properties.
        Uses ffmpeg.probe() to gather the information.

        Args:
            video_path: Path to the video file to analyze.

        Returns:
            A dictionary containing video metadata with detailed information
            about format, video stream, and audio stream properties, including
            filename, format, duration, size, bitrate, and optional video/audio
            stream metadata.

        Raises:
            RuntimeError: If ffmpeg encounters an error during analysis.
        """
        resolved = safe_input_path(video_path)
        return get_video_metadata(str(resolved))

    # Ensure function is registered with MCP
    del get_video_info

    @mcp.tool
    async def resize_video(
        input_path: str,
        output_path: str,
        width: int | None = None,
        height: int | None = None,
        scale: float | None = None,
        crf: int | None = None,
        preset: str | None = None,
        ctx: Context | None = None,
    ) -> str:
        """Resize a video to specified dimensions or scale factor.

        Resizes a video using one of three methods: specific width (maintaining
        aspect ratio), specific height (maintaining aspect ratio), or uniform
        scaling by a factor. Exactly one parameter must be provided.

        Output dimensions are always forced to even numbers: the auto-computed
        axis uses ``-2`` and the caller-specified axis is rounded down to an
        even value, so libx264 with ``yuv420p`` never rejects an odd dimension.

        Args:
            input_path: Path to the input video file.
            output_path: Path where the resized video will be saved.
            width: Target width in pixels. Height will be calculated automatically.
            height: Target height in pixels. Width will be calculated automatically.
            scale: Scaling factor (0.1 to 10.0). 1.0 = original size.
            crf: Constant Rate Factor (lower = higher quality).
            preset: libx264 speed/quality preset (e.g. "medium", "ultrafast").
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating the video was resized and saved.

        Raises:
            ValueError: If parameter constraints are not met.
            RuntimeError: If ffmpeg encounters an error during processing.
        """
        param_count = sum(x is not None for x in [width, height, scale])
        if param_count != 1:
            raise ValueError("Provide exactly one: width, height, or scale")

        resolved_input = safe_input_path(input_path)
        resolved_output = safe_output_path(output_path)

        stream = ffmpeg.input(str(resolved_input))

        if scale is not None:
            validate_range(
                scale,
                0.1,
                10.0,
                "Scale factor",
            )
            # trunc(...*/2)*2 forces an even result on both axes.
            stream = ffmpeg.filter(
                stream,
                "scale",
                f"trunc(iw*{scale}/2)*2",
                f"trunc(ih*{scale}/2)*2",
            )
            await log_operation(
                ctx,
                f"Resizing video by {scale}x",
            )
        elif width is not None:
            # -2 keeps the auto axis even and aspect-correct.
            stream = ffmpeg.filter(stream, "scale", str(even_dimension(width)), "-2")
            await log_operation(
                ctx,
                f"Resizing video to width {even_dimension(width)}px",
            )
        else:  # height
            assert height is not None  # Type narrowing for type checker
            stream = ffmpeg.filter(stream, "scale", "-2", str(even_dimension(height)))
            await log_operation(
                ctx,
                f"Resizing video to height {even_dimension(height)}px",
            )

        # Only the video is filtered, so map='0:a?' carries the source audio
        # through when present (and is a no-op on silent sources), while
        # copy_audio stream-copies it instead of a pointless AAC re-encode.
        output = create_standard_output(
            stream,
            str(resolved_output),
            crf=crf,
            preset=preset,
            faststart=_wants_faststart(resolved_output),
            copy_audio=True,
            map="0:a?",
        )
        await run_ffmpeg_async(output, ctx=ctx, output_path=str(resolved_output))
        return f"Video resized and saved to {resolved_output}"

    # Ensure function is registered with MCP
    del resize_video

    @mcp.tool
    async def concatenate_videos(
        input_paths: list[str],
        output_path: str,
        re_encode: bool | None = None,
        crf: int | None = None,
        preset: str | None = None,
        ctx: Context | None = None,
    ) -> str:
        """Concatenate multiple videos into a single continuous video.

        Joins two or more video files end to end. The tool probes every input
        and picks the best strategy automatically:

            - Lossless fast path: when all inputs are homogeneous (same codec,
              resolution, frame rate, pixel format and matching/absent audio),
              they are stitched with the concat *demuxer* + ``-c copy`` — near
              instant and with zero generational quality loss. This is the
              common case for clips from the same generation pipeline.
            - Re-encode path: when inputs are heterogeneous, each is normalized
              (scale + pad to a common resolution, fps, SAR, pixel format)
              before the concat *filter* joins them. Silent inputs get a
              generated ``anullsrc`` audio track so audio streams stay aligned;
              when *no* input has audio the result is a video-only concat.

        Args:
            input_paths: Paths to the video files to concatenate (minimum 2).
                Order is preserved in the output.
            output_path: Path where the concatenated video will be saved.
            re_encode: Force a strategy. ``None`` (default) auto-detects: copy
                when homogeneous, re-encode otherwise. ``True`` always
                re-encodes with normalization. ``False`` always uses the
                lossless demuxer copy (only safe for homogeneous inputs).
            crf: Constant Rate Factor for the re-encode path (lower = higher
                quality). Ignored for the copy path.
            preset: libx264 speed/quality preset for the re-encode path.
                Ignored for the copy path.
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating videos were concatenated and saved.

        Raises:
            ValueError: If fewer than 2 videos are provided.
            RuntimeError: If an input cannot be probed or ffmpeg fails.
        """
        if len(input_paths) < 2:
            raise ValueError("At least 2 videos required for concatenation")

        resolved_inputs = [safe_input_path(p) for p in input_paths]
        resolved_output = safe_output_path(output_path)

        # Probe every input. The binary is required at runtime (Docker/CI);
        # a failure here is surfaced as a clean RuntimeError naming the file
        # rather than an opaque traceback.
        metas: list[VideoMetadata] = []
        for path in resolved_inputs:
            try:
                metas.append(get_video_metadata(str(path)))
            except RuntimeError as e:
                raise RuntimeError(
                    f"Could not probe input for concatenation: {path} ({e})"
                ) from e

        homogeneous = _inputs_homogeneous(metas)
        if re_encode is None:
            use_copy = homogeneous
        else:
            use_copy = not re_encode

        await log_operation(
            ctx,
            f"Concatenating {len(resolved_inputs)} videos "
            f"({'lossless copy' if use_copy else 're-encode'})",
        )

        if use_copy:
            list_path = _write_concat_list(resolved_inputs)
            try:
                demux = ffmpeg.input(list_path, format="concat", safe=0)
                settings: dict[str, str] = {"c": "copy"}
                if _wants_faststart(resolved_output):
                    settings["movflags"] = "+faststart"
                output = ffmpeg.output(demux, str(resolved_output), **settings)
                await run_ffmpeg_async(
                    output, ctx=ctx, output_path=str(resolved_output)
                )
            finally:
                if os.path.exists(list_path):
                    os.unlink(list_path)
            return (
                "Videos concatenated (lossless copy) and saved to "
                f"{resolved_output}"
            )

        # Re-encode path with per-input normalization.
        width, height, fps = _target_geometry(metas)
        has_audio = ["audio" in m for m in metas]
        any_audio = any(has_audio)

        streams: list[Any] = []
        for i, path in enumerate(resolved_inputs):
            inp = ffmpeg.input(str(path))
            streams.append(
                normalize_video_stream(inp.video, width=width, height=height, fps=fps)
            )
            if not any_audio:
                continue
            if has_audio[i]:
                streams.append(normalize_audio_stream(inp.audio))
            else:
                # Inject matched-duration silence for this segment so the
                # concat filter's audio and video segment counts align.
                duration = metas[i]["duration"] or 0.0
                silence = silent_audio_source(duration)
                streams.append(normalize_audio_stream(silence.audio))

        n = len(resolved_inputs)
        faststart = _wants_faststart(resolved_output)
        if any_audio:
            joined = ffmpeg.concat(*streams, v=1, a=1, n=n).node
            enc: dict[str, str | int | float] = {
                "vcodec": "libx264",
                "pix_fmt": "yuv420p",
                "acodec": "aac",
            }
            if crf is not None:
                enc["crf"] = crf
            if preset is not None:
                enc["preset"] = preset
            if faststart:
                enc["movflags"] = "+faststart"
            output = ffmpeg.output(joined[0], joined[1], str(resolved_output), **enc)
        else:
            joined = ffmpeg.concat(*streams, v=1, a=0, n=n)
            output = create_standard_output(
                joined,
                str(resolved_output),
                crf=crf,
                preset=preset,
                faststart=faststart,
            )

        await run_ffmpeg_async(output, ctx=ctx, output_path=str(resolved_output))
        return f"Videos concatenated successfully and saved to {resolved_output}"

    # Ensure function is registered with MCP
    del concatenate_videos

    @mcp.tool
    async def image_to_video(
        image_path: str,
        output_path: str,
        duration: float,
        framerate: int = 30,
        crf: int | None = None,
        preset: str | None = None,
        ctx: Context | None = None,
    ) -> str:
        """Create a video from a static image for a specified duration.

        Converts a static image into a video by displaying the image for the
        specified duration. The output will be a video file with the given
        framerate showing the same image throughout. Output dimensions are
        forced even so odd-sized images do not fail libx264/yuv420p encoding.

        Args:
            image_path: Path to the input image file (supports common formats).
            output_path: Path where the video will be saved.
            duration: Duration of the video in seconds.
            framerate: Framerate of the output video (default: 30 fps).
            crf: Constant Rate Factor (lower = higher quality).
            preset: libx264 speed/quality preset (e.g. "medium", "ultrafast").
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating the video was created and saved.

        Raises:
            ValueError: If duration is not positive or framerate is invalid.
            RuntimeError: If ffmpeg encounters an error during processing.
        """
        if duration <= 0:
            raise ValueError("Duration must be positive")

        if framerate <= 0 or framerate > 120:
            raise ValueError("Framerate must be between 1 and 120 fps")

        resolved_input = safe_input_path(image_path)
        resolved_output = safe_output_path(output_path)

        await log_operation(
            ctx,
            f"Creating {duration}s video from image at {framerate} fps",
        )

        stream = ffmpeg.input(
            str(resolved_input),
            loop=1,
            t=duration,
            framerate=framerate,
        )
        # Force even dimensions; odd-sized stills break yuv420p H.264.
        stream = ffmpeg.filter(
            stream,
            "scale",
            "trunc(iw/2)*2",
            "trunc(ih/2)*2",
        )
        output = create_standard_output(
            stream,
            str(resolved_output),
            crf=crf,
            preset=preset,
            faststart=_wants_faststart(resolved_output),
        )
        await run_ffmpeg_async(output, ctx=ctx, output_path=str(resolved_output))
        return f"Video created successfully and saved to {resolved_output}"

    # Ensure function is registered with MCP
    del image_to_video
