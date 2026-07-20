"""Video effects and filters: speed changes, filters, and thumbnails.

This module provides advanced video manipulation tools including visual filters,
speed changes, and thumbnail generation. Supports a wide range of effects from
basic color adjustments to artistic filters like sepia, blur, and sharpening.

Supported filters:
    - brightness: Adjust video brightness (0.0-2.0)
    - contrast: Adjust video contrast (0.0-2.0)
    - saturation: Adjust color saturation (0.0-2.0)
    - hflip: Flip video horizontally
    - grayscale: Convert to grayscale
    - sepia: Apply sepia tone effect
    - blur: Apply gaussian blur
    - sharpen: Apply unsharp mask sharpening
    - vintage: Apply vintage color grading
    - scale: Resize video with specific dimensions

Example:
    Apply a blur effect to a video:

        await apply_filter(
            input_path="input.mp4",
            output_path="blurred.mp4",
            filter="blur",
            strength=1.5
        )
"""

import ffmpeg
from fastmcp import Context, FastMCP

from ..core import (
    create_standard_output,
    get_video_metadata,
    log_operation,
    run_ffmpeg_async,
    safe_input_path,
    safe_output_path,
    validate_filter_name,
    validate_range,
)


def _force_even(value: int) -> int:
    """Round a scale dimension to an even number libx264/yuv420p accepts.

    Positive values are rounded down to the nearest even integer. The ffmpeg
    "auto" sentinel ``-1`` is mapped to ``-2`` so ffmpeg preserves aspect ratio
    while still producing an even dimension.

    Args:
        value: The requested scale dimension.

    Returns:
        An even dimension (or ``-2`` for the auto sentinel).
    """
    if value < 0:
        return -2
    return value - (value % 2)


def _speed_output_settings(
    crf: int | None,
    preset: str | None,
) -> dict[str, str | int | float]:
    """Build encoder settings for a two-stream (video + audio) speed output.

    Mirrors the defaults of :func:`create_standard_output` (libx264 / yuv420p /
    aac) while honouring the optional ``crf`` and ``preset`` quality controls.
    Needed because ``create_standard_output`` maps only a single stream, whereas
    a sped-up clip with audio must map both a video and an audio stream.

    Args:
        crf: Optional Constant Rate Factor for libx264.
        preset: Optional libx264 speed/quality preset.

    Returns:
        Keyword settings suitable for :func:`ffmpeg.output`.
    """
    settings: dict[str, str | int | float] = {
        "vcodec": "libx264",
        "pix_fmt": "yuv420p",
        "acodec": "aac",
    }
    if crf is not None:
        settings["crf"] = crf
    if preset is not None:
        settings["preset"] = preset
    return settings


def _has_audio_stream(video_path: str) -> bool:
    """Return ``True`` when the file at ``video_path`` carries an audio stream.

    Silent sources (e.g. Veo 2 output) have no audio stream, so tools that map
    ``stream["a"]`` must guard against them. Metadata is gathered via
    :func:`get_video_metadata`, whose result only includes an ``"audio"`` key
    when an audio stream is present.

    Args:
        video_path: Path to the video file to probe.

    Returns:
        ``True`` if an audio stream exists, ``False`` otherwise.
    """
    return "audio" in get_video_metadata(video_path)


def register_video_effects_tools(
    mcp: FastMCP[None],
) -> None:
    """Register video effects tools with the MCP server.

    Adds advanced video manipulation capabilities including visual filters,
    speed changes, and thumbnail generation to the provided FastMCP server.

    Args:
        mcp: The FastMCP server instance to register tools with.

    Returns:
        None
    """

    @mcp.tool
    async def apply_filter(
        input_path: str,
        output_path: str,
        filter: str,
        strength: float = 1.0,
        crf: int | None = None,
        preset: str | None = None,
        ctx: Context | None = None,
    ) -> str:
        """Apply visual effects filter to a video.

        Applies various visual filters to enhance or stylize video content.
        Filter strength can be adjusted to control the intensity of the effect.
        The video stream is re-encoded (libx264) while any existing audio track
        is preserved.

        Available filters:
            - brightness: Brightens or darkens the video (0.1-3.0)
            - contrast: Adjusts contrast levels (0.1-3.0)
            - saturation: Controls color saturation (0.1-3.0)
            - hflip: Horizontally flips the video (strength ignored)
            - grayscale: Converts to grayscale based on strength (0.0-1.0)
            - sepia: Applies sepia tone effect (0.1-1.0)
            - blur: Applies gaussian blur (strength controls radius)
            - sharpen: Applies unsharp mask sharpening (0.1-3.0)
            - vintage: Applies vintage color grading
            - scale=WxH: Resizes to specific dimensions (rounded to even values)

        Args:
            input_path: Path to the input video file.
            output_path: Path where the filtered video will be saved.
            filter: Name of the filter to apply. See available filters above.
            strength: Filter intensity (0.1 to 3.0). 1.0 = normal strength.
            crf: Optional libx264 Constant Rate Factor (0-51, lower is higher
                quality). Defaults to the encoder default when omitted.
            preset: Optional libx264 speed/quality preset (e.g. ``"ultrafast"``,
                ``"medium"``). Defaults to the encoder default when omitted.
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating filter was applied and video saved.

        Raises:
            ValueError: If filter name or strength is invalid.
            RuntimeError: If ffmpeg encounters an error during processing.

        Example:
            Apply a moderate blur effect:

                result = await apply_filter(
                    input_path="input.mp4",
                    output_path="blurred.mp4",
                    filter="blur",
                    strength=1.5
                )
        """
        _ = validate_filter_name(filter)
        validate_range(strength, 0.1, 3.0, "Filter strength")
        if crf is not None:
            validate_range(crf, 0, 51, "CRF")

        resolved_input = str(safe_input_path(input_path))
        resolved_output = str(safe_output_path(output_path))

        await log_operation(
            ctx,
            f"Applying {filter} filter with strength {strength}",
        )

        stream: ffmpeg.Stream = ffmpeg.input(resolved_input)

        # Apply different filters based on name
        if filter == "blur":
            # Apply gaussian blur with strength controlling the blur radius
            blur_radius = max(0.5, min(strength * 5, 10))  # Scale strength
            stream = ffmpeg.filter(stream, "gblur", sigma=blur_radius)
        elif filter == "brightness":
            stream = ffmpeg.filter(
                stream,
                "eq",
                brightness=strength - 1,
            )
        elif filter == "contrast":
            stream = ffmpeg.filter(
                stream,
                "eq",
                contrast=strength,
            )
        elif filter == "saturation":
            stream = ffmpeg.filter(
                stream,
                "eq",
                saturation=strength,
            )
        elif filter == "vintage":
            # Apply vintage effect using color correction
            stream = ffmpeg.filter(
                stream,
                "eq",
                brightness=0.1 * strength,
                contrast=1.2 * strength,
                saturation=0.7 * strength,
            )
        elif filter == "sepia":
            sepia_strength = min(strength, 1.0)
            stream = ffmpeg.filter(
                stream,
                "colorchannelmixer",
                rr=0.393 * sepia_strength,
                rg=0.769 * sepia_strength,
                rb=0.189 * sepia_strength,
            )
        elif filter == "grayscale":
            stream = ffmpeg.filter(stream, "hue", s=1 - strength)
        elif filter == "hflip":
            stream = ffmpeg.filter(stream, "hflip")
        elif filter == "sharpen":
            # Apply unsharp mask for sharpening with strength controlling amount
            sharpen_amount = max(0.1, min(strength, 3.0))  # Scale strength
            stream = ffmpeg.filter(
                stream,
                "unsharp",
                luma_msize_x=5,
                luma_msize_y=5,
                luma_amount=sharpen_amount,
            )
        elif filter.startswith("scale="):
            # Handle scale filter with parameters like scale=640:360.
            # Force even dimensions so libx264/yuv420p does not reject odd
            # width/height; an axis of -1 (auto) becomes -2 (auto + even).
            scale_params = filter.split("=")[1]
            width, height = scale_params.split(":")
            stream = ffmpeg.filter(
                stream,
                "scale",
                str(_force_even(int(width))),
                str(_force_even(int(height))),
            )

        output: ffmpeg.Stream = create_standard_output(
            stream,
            resolved_output,
            crf=crf,
            preset=preset,
            map="0:a?",
        )
        await run_ffmpeg_async(output, ctx=ctx, output_path=resolved_output)
        return f"{filter.title()} filter applied and saved to {output_path}"

    @mcp.tool
    async def change_speed(
        input_path: str,
        output_path: str,
        speed: float,
        crf: int | None = None,
        preset: str | None = None,
        ctx: Context | None = None,
    ) -> str:
        """Change the playback speed of a video.

        Adjusts video playback speed while maintaining audio synchronization.
        Values greater than 1.0 speed up the video, values less than 1.0 slow it down.

        The function handles FFmpeg's atempo filter limitations (0.5-2.0 range) by
        automatically chaining multiple atempo filters for extreme speed changes.
        This ensures smooth audio processing at any speed within the supported range.

        Silent inputs (no audio stream, e.g. Veo 2 output) are handled
        gracefully: only the video presentation timestamps are adjusted and the
        output is written video-only, skipping the atempo chain entirely.

        Args:
            input_path: Path to the input video file.
            output_path: Path where the speed-adjusted video will be saved.
            speed: Speed multiplier (0.1 to 10.0). Examples:
                - 0.5 = half speed (slow motion)
                - 1.0 = normal speed (no change)
                - 2.0 = double speed (fast forward)
            crf: Optional libx264 Constant Rate Factor (0-51, lower is higher
                quality). Defaults to the encoder default when omitted.
            preset: Optional libx264 speed/quality preset (e.g. ``"ultrafast"``,
                ``"medium"``). Defaults to the encoder default when omitted.
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating speed was changed and video saved.

        Raises:
            ValueError: If speed factor is out of valid range or zero/negative.
            RuntimeError: If ffmpeg encounters an error during processing.

        Example:
            Create slow motion at half speed:

                result = await change_speed(
                    input_path="normal.mp4",
                    output_path="slow_motion.mp4",
                    speed=0.5
                )
        """
        # Custom validation for speed - must be positive
        if speed <= 0:
            raise ValueError("Speed factor must be greater than 0")

        validate_range(speed, 0.1, 10.0, "Speed factor")
        if crf is not None:
            validate_range(crf, 0, 51, "CRF")

        resolved_input = str(safe_input_path(input_path))
        resolved_output = str(safe_output_path(output_path))

        # Probe up front so we know whether an audio stream exists; mapping
        # stream["a"] on a silent source would otherwise fail with an opaque
        # ffmpeg mapping error (H4).
        has_audio = _has_audio_stream(resolved_input)

        await log_operation(
            ctx,
            f"Changing video speed by {speed}x"
            + ("" if has_audio else " (silent input, video only)"),
        )

        stream: ffmpeg.Stream = ffmpeg.input(resolved_input)

        # Apply speed change to the video by scaling presentation timestamps.
        video_stream: ffmpeg.Stream = ffmpeg.filter(
            stream["v"],
            "setpts",
            f"PTS/{speed}",
        )

        if not has_audio:
            # No audio stream: write video only, skipping atempo entirely.
            output: ffmpeg.Stream = create_standard_output(
                video_stream,
                resolved_output,
                crf=crf,
                preset=preset,
            )
        else:
            # Handle atempo filter limitations (0.5-2.0 range). For speeds
            # outside this range, chain multiple atempo filters.
            audio_stream: ffmpeg.Stream = stream["a"]
            current_speed = speed

            while current_speed > 2.0:
                audio_stream = ffmpeg.filter(audio_stream, "atempo", "2.0")
                current_speed /= 2.0

            while current_speed < 0.5:
                audio_stream = ffmpeg.filter(audio_stream, "atempo", "0.5")
                current_speed /= 0.5

            if current_speed != 1.0:
                audio_stream = ffmpeg.filter(
                    audio_stream, "atempo", str(current_speed)
                )

            # Both video and audio are re-encoded. create_standard_output
            # only maps a single stream, so build the two-stream output
            # directly using the same encoding settings/quality controls.
            output = ffmpeg.output(
                video_stream,
                audio_stream,
                resolved_output,
                **_speed_output_settings(crf, preset),
            )

        await run_ffmpeg_async(output, ctx=ctx, output_path=resolved_output)

        speed_desc = "faster" if speed > 1.0 else "slower"
        return (
            f"Video speed changed {speed_desc} ({speed}x) and saved to "
            f"{output_path}"
        )

    @mcp.tool
    async def generate_thumbnail(
        input_path: str,
        output_path: str,
        timestamp: float = 2.5,
        width: int | None = None,
        height: int | None = None,
        ctx: Context | None = None,
    ) -> str:
        """Generate a thumbnail image from a video.

        Extracts a single frame from the video at the specified timestamp and
        resizes it to create a thumbnail image.

        Args:
            input_path: Path to the input video file.
            output_path: Path where the thumbnail image will be saved.
            timestamp: Time in seconds to extract frame from (0.0 to video duration).
            width: Thumbnail width in pixels (50 to 1920). If None, uses original width.
            height: Thumbnail height in pixels (50 to 1080). If None, uses original.
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating thumbnail was generated and saved.

        Raises:
            ValueError: If dimensions are out of valid ranges.
            RuntimeError: If ffmpeg encounters an error during processing.
        """
        if timestamp < 0:
            raise ValueError("Timestamp must be non-negative")
        if width is not None:
            validate_range(width, 50, 1920, "Width")
        if height is not None:
            validate_range(height, 50, 1080, "Height")

        resolved_input = str(safe_input_path(input_path))
        resolved_output = str(safe_output_path(output_path))

        size_desc = (
            "original size"
            if width is None and height is None
            else f"{width or 'auto'}x{height or 'auto'}"
        )
        await log_operation(
            ctx,
            f"Generating {size_desc} thumbnail at {timestamp}s",
        )

        stream: ffmpeg.Stream = ffmpeg.input(resolved_input, ss=timestamp)

        # Only apply scaling if dimensions are specified
        if width is not None or height is not None:
            # Use -2 for the auto axis so ffmpeg preserves aspect ratio and
            # yields an even dimension when only one axis is specified.
            scale_width = width if width is not None else -2
            scale_height = height if height is not None else -2
            stream = ffmpeg.filter(
                stream, "scale", str(scale_width), str(scale_height)
            )

        output: ffmpeg.Stream = ffmpeg.output(stream, resolved_output, vframes=1)
        await run_ffmpeg_async(output, ctx=ctx, output_path=resolved_output)
        return f"Thumbnail generated and saved to {output_path}"

    # Mark decorated functions as used (they're accessed via the @mcp.tool decorator)
    _ = (apply_filter, change_speed, generate_thumbnail)
