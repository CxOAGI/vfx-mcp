"""Format and codec conversion tools.

This module provides comprehensive video and audio format conversion capabilities
with support for various containers, codecs, and quality settings. Enables
conversion between different video formats while maintaining quality control.

Supported formats:
    - mp4: MPEG-4 container (widely compatible)
    - avi: Audio Video Interleave (legacy support)
    - mkv: Matroska container (open source, feature-rich)
    - webm: WebM container (web optimized)

Common codec combinations:
    - H.264 + AAC: Best compatibility (mp4, mkv)
    - H.265 + AAC: Higher compression (mp4, mkv)
    - VP9 + Vorbis: Open source (webm, mkv)

Example:
    Convert to high-quality H.265 (the explicit ``video_codec`` wins over the
    codec the target ``format`` would otherwise default to):

        await convert_format(
            input_path="input.avi",
            output_path="output.mp4",
            format="mp4",
            video_codec="libx265",
            video_bitrate="2M",
        )
"""

from pathlib import Path

import ffmpeg
from fastmcp import Context, FastMCP

from ..core import (
    create_standard_output,
    handle_ffmpeg_error,
    log_operation,
    run_ffmpeg_async,
    safe_input_path,
    safe_output_path,
)

# Fallback codecs used when neither the caller nor the target ``format`` picks
# a codec. libx264/aac is the most broadly compatible MP4 pairing.
_DEFAULT_VIDEO_CODEC = "libx264"
_DEFAULT_AUDIO_CODEC = "aac"

# Per-container codec defaults. Passing ``format`` only sets these DEFAULTS; an
# explicit ``video_codec``/``audio_codec`` argument always wins (see M4).
_FORMAT_CODECS: dict[str, dict[str, str]] = {
    "mp4": {"video": "libx264", "audio": "aac"},
    "avi": {"video": "libx264", "audio": "mp3"},
    "mkv": {"video": "libx264", "audio": "aac"},
    "webm": {"video": "libvpx-vp9", "audio": "libvorbis"},
    "mov": {"video": "libx264", "audio": "aac"},
}

# Container families that support the MP4/MOV ``moov`` atom and therefore
# benefit from ``-movflags +faststart`` for progressive (streamed) playback.
_FASTSTART_FORMATS: frozenset[str] = frozenset({"mp4", "mov", "m4v", "m4a"})
_FASTSTART_SUFFIXES: frozenset[str] = frozenset({".mp4", ".mov", ".m4v", ".m4a"})


def _wants_faststart(output_path: str, format: str | None) -> bool:
    """Return True when the output is an MP4/MOV-family deliverable.

    ``-movflags +faststart`` is only valid for the MP4/MOV muxers, so it is
    enabled based on the explicit ``format`` (when given) or the output file's
    extension, and never for containers such as webm/avi/mkv.
    """
    if format is not None:
        return format.lower() in _FASTSTART_FORMATS
    return Path(output_path).suffix.lower() in _FASTSTART_SUFFIXES


def register_format_conversion_tools(
    mcp: FastMCP[None],
) -> None:
    """Register format conversion tools with the MCP server.

    Adds video and audio format conversion capabilities with support for
    various containers, codecs, and quality settings to the FastMCP server.

    Args:
        mcp: The FastMCP server instance to register tools with.

    Returns:
        None
    """

    @mcp.tool
    async def convert_format(
        input_path: str,
        output_path: str,
        format: str | None = None,
        video_codec: str | None = None,
        audio_codec: str | None = None,
        video_bitrate: str | None = None,
        audio_bitrate: str | None = "128k",
        crf: int | None = None,
        preset: str | None = None,
        ctx: Context | None = None,
    ) -> str:
        """Convert video format and adjust encoding settings.

        Converts a video file to a different container/codec with customizable
        quality settings. The optional ``format`` argument only selects
        *default* codecs for that container; any explicit ``video_codec`` or
        ``audio_codec`` you pass always takes precedence. MP4/MOV outputs are
        automatically written with ``-movflags +faststart`` for progressive
        playback.

        Args:
            input_path: Path to the input video file (within the workspace).
            output_path: Path where the converted video will be saved.
            format: Target container ("mp4", "avi", "mkv", "webm", "mov").
                When given, it only sets default codecs; explicit codec
                arguments override these defaults.
            video_codec: Video codec ("libx264", "libx265", "libvpx-vp9",
                etc.). When None, falls back to the ``format`` default (or
                libx264).
            audio_codec: Audio codec ("aac", "mp3", "libvorbis", etc.). When
                None, falls back to the ``format`` default (or aac).
            video_bitrate: Target video bitrate (e.g. "1M", "2.5M"). When None,
                the encoder chooses (or ``crf`` governs quality).
            audio_bitrate: Target audio bitrate (e.g. "128k", "192k", "320k").
            crf: Constant Rate Factor for the video encoder (lower = higher
                quality; typical range 18-28 for x264/x265). Ignored by
                encoders that do not support it.
            preset: Encoder speed/quality preset (e.g. "ultrafast", "medium",
                "slow").
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating the format was converted and saved.

        Raises:
            RuntimeError: If ffmpeg encounters an error during processing.
            ValueError: If a path escapes the workspace sandbox.
            FileNotFoundError: If the input file does not exist.
        """
        # Resolve/validate paths against the workspace sandbox before touching
        # ffmpeg so protocol inputs and path traversal are rejected up front.
        resolved_input = safe_input_path(input_path)
        resolved_output = safe_output_path(output_path)

        # ``format`` only supplies DEFAULTS; an explicit codec argument wins
        # (M4). We detect "explicit" by the argument being non-None.
        format_defaults = _FORMAT_CODECS.get(format.lower(), {}) if format else {}
        resolved_video_codec = (
            video_codec
            if video_codec is not None
            else format_defaults.get("video", _DEFAULT_VIDEO_CODEC)
        )
        resolved_audio_codec = (
            audio_codec
            if audio_codec is not None
            else format_defaults.get("audio", _DEFAULT_AUDIO_CODEC)
        )

        faststart = _wants_faststart(output_path, format)

        await log_operation(
            ctx,
            f"Converting format: {resolved_video_codec}/{resolved_audio_codec} "
            f"(vbr: {video_bitrate or 'auto'}, abr: {audio_bitrate or 'auto'}, "
            f"crf: {crf if crf is not None else 'auto'}, "
            f"preset: {preset or 'default'}, faststart: {faststart})",
        )

        try:
            stream = ffmpeg.input(str(resolved_input))

            output = create_standard_output(
                stream,
                str(resolved_output),
                crf=crf,
                preset=preset,
                video_bitrate=video_bitrate,
                audio_bitrate=audio_bitrate,
                faststart=faststart,
                vcodec=resolved_video_codec,
                acodec=resolved_audio_codec,
            )
            await run_ffmpeg_async(output, ctx=ctx)
            return f"Format converted successfully and saved to {output_path}"
        except ffmpeg.Error as e:
            await handle_ffmpeg_error(e, ctx)
            raise  # This line is never reached but satisfies type checker

    # Acknowledge that the function is registered with MCP
    _ = convert_format
