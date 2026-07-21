"""Common utilities and helper functions for VFX operations."""

import asyncio
import os
from fractions import Fraction
from pathlib import Path
from typing import NotRequired, TypedDict

import ffmpeg
from fastmcp import Context

# Maximum number of characters of ffmpeg stdout/stderr surfaced in an error
# message. ffmpeg can emit tens of kilobytes of progress/log noise; only the
# tail is diagnostically useful, so we keep the last ``_MAX_ERROR_CHARS``.
_MAX_ERROR_CHARS = 4000


def _default_max_concurrency() -> int:
    """Read the max concurrent ffmpeg jobs from ``VFX_MAX_CONCURRENCY``."""
    raw = os.environ.get("VFX_MAX_CONCURRENCY")
    if not raw:
        return 3
    try:
        value = int(raw)
    except ValueError:
        return 3
    return value if value >= 1 else 1


def _default_timeout() -> float | None:
    """Read the default ffmpeg timeout (seconds) from ``VFX_FFMPEG_TIMEOUT``.

    Returns ``None`` (no timeout) when the variable is unset or unparseable.
    """
    raw = os.environ.get("VFX_FFMPEG_TIMEOUT")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# Global concurrency limiter shared by every ``run_ffmpeg_async`` call so a
# burst of MCP requests cannot spawn an unbounded number of ffmpeg processes.
# asyncio.Semaphore construction does not require a running event loop.
_ffmpeg_semaphore: asyncio.Semaphore = asyncio.Semaphore(_default_max_concurrency())


def _truncate_tail(text: str, max_chars: int = _MAX_ERROR_CHARS) -> str:
    """Keep only the last ``max_chars`` characters of ``text``.

    The tail of ffmpeg output holds the actual error, so we trim from the
    front and align the cut to a line boundary when possible.
    """
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    newline = tail.find("\n")
    if newline != -1:
        tail = tail[newline + 1 :]
    return f"...[truncated {len(text) - len(tail)} chars]...\n{tail}"


def _decode(raw: bytes | None) -> str:
    """Best-effort UTF-8 decode of ffmpeg stdout/stderr bytes."""
    if not raw:
        return ""
    try:
        return raw.decode("utf-8")
    except Exception:
        return str(raw)


def _unlink_quietly(path: str | None) -> None:
    """Best-effort removal of a (possibly partial) output file.

    Used to clean up the half-written file ffmpeg leaves behind when an encode
    is killed (timeout / client disconnect) or exits non-zero. Does nothing
    when ``path`` is ``None`` and swallows :class:`OSError` (missing file,
    permission issue) so cleanup never masks the original failure or a pending
    cancellation.
    """
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _terminate(process: object) -> None:
    """Kill an ffmpeg subprocess and reap it, ignoring any errors.

    ``process`` is the object returned by :func:`ffmpeg.run_async` (a
    ``subprocess.Popen``). Both ``kill`` and ``wait`` are guarded so cleanup is
    safe to run from a timeout handler or a cancellation handler.
    """
    kill = getattr(process, "kill", None)
    if callable(kill):
        try:
            kill()
        except Exception:
            pass
    wait = getattr(process, "wait", None)
    if callable(wait):
        try:
            wait()
        except Exception:
            pass


async def handle_ffmpeg_error(e: ffmpeg.Error, ctx: Context | None = None) -> None:
    """Standard error handling for ffmpeg operations.

    Decodes the captured stdout/stderr, truncates it to a sane tail so huge
    ffmpeg logs do not flood MCP responses, logs it via the context, and
    re-raises as a :class:`RuntimeError`. This function always raises.
    """
    stderr_msg = _truncate_tail(_decode(e.stderr))
    stdout_msg = _truncate_tail(_decode(e.stdout))

    error_msg = f"FFmpeg error: {stderr_msg or stdout_msg or str(e)}"
    if ctx:
        await ctx.error(error_msg)
    raise RuntimeError(error_msg) from e


async def run_ffmpeg_async(
    output_stream: ffmpeg.Stream,
    *,
    timeout: float | None = None,
    ctx: Context | None = None,
    output_path: str | None = None,
) -> None:
    """Run an ffmpeg output stream without blocking the event loop.

    The blocking ffmpeg subprocess is launched in a worker thread (via
    :func:`asyncio.to_thread`) while a module-level semaphore caps the number
    of concurrent encodes (``VFX_MAX_CONCURRENCY``, default 3). Output is
    always overwritten and stdout/stderr are captured so that failures can be
    translated through :func:`handle_ffmpeg_error`.

    When the encode does not complete cleanly -- it times out, is cancelled
    (e.g. the MCP client disconnects mid-encode), or exits non-zero -- ffmpeg
    typically leaves a truncated, unplayable file behind. If ``output_path`` is
    supplied, that partial file is removed on any of these paths so callers
    never observe a corrupt deliverable.

    Args:
        output_stream: A configured ffmpeg output node (e.g. from
            :func:`create_standard_output`).
        timeout: Maximum seconds to allow the encode to run. When ``None``
            the default from ``VFX_FFMPEG_TIMEOUT`` is used (or no timeout at
            all when that is unset). On timeout the process is killed and a
            :class:`RuntimeError` is raised.
        ctx: Optional MCP context for error reporting.
        output_path: Optional path to the file ffmpeg is writing. When given,
            the partial output is best-effort deleted on timeout, cancellation
            or non-zero exit. Defaults to ``None`` (no cleanup), keeping the
            signature backward compatible for callers that do not pass it.

    Raises:
        RuntimeError: On ffmpeg failure (non-zero exit) or timeout.
        asyncio.CancelledError: Re-raised unchanged when the awaiting task is
            cancelled while the encode is running (after killing the process
            and cleaning up any partial output). Cancellation is never
            swallowed.
    """
    effective_timeout = timeout if timeout is not None else _default_timeout()

    async with _ffmpeg_semaphore:
        process = await asyncio.to_thread(
            ffmpeg.run_async,
            output_stream,
            pipe_stdout=True,
            pipe_stderr=True,
            overwrite_output=True,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.to_thread(process.communicate),
                timeout=effective_timeout,
            )
        except TimeoutError:
            await asyncio.to_thread(_terminate, process)
            _unlink_quietly(output_path)
            error_msg = (
                f"FFmpeg operation timed out after {effective_timeout} seconds "
                "and was killed."
            )
            if ctx:
                await ctx.error(error_msg)
            raise RuntimeError(error_msg) from None
        except asyncio.CancelledError:
            # The awaiting task was cancelled (e.g. client disconnect) while the
            # encode was in flight. Kill and reap the subprocess synchronously
            # -- re-awaiting during cancellation is fragile and the killed
            # process reaps almost immediately -- drop the partial file, then
            # re-raise so cancellation is never swallowed.
            _terminate(process)
            _unlink_quietly(output_path)
            raise

        retcode = process.poll()
        if retcode:
            _unlink_quietly(output_path)
            error = ffmpeg.Error("ffmpeg", stdout, stderr)
            await handle_ffmpeg_error(error, ctx)


async def log_operation(ctx: Context | None, message: str) -> None:
    """Log operation info if context is available."""
    if ctx:
        await ctx.info(message)


class VideoStreamMetadata(TypedDict):
    """Video stream metadata."""

    codec: str
    width: int
    height: int
    fps: float
    aspect_ratio: str
    pixel_format: str


class AudioStreamMetadata(TypedDict):
    """Audio stream metadata."""

    codec: str
    channels: int
    sample_rate: int
    bitrate: int


class VideoMetadata(TypedDict):
    """Video metadata structure."""

    filename: str
    format: str
    duration: float
    size: int
    bitrate: int
    video: NotRequired[VideoStreamMetadata]
    audio: NotRequired[AudioStreamMetadata]


def get_video_metadata(
    video_path: str,
) -> VideoMetadata:
    """Extract comprehensive video metadata using ffmpeg probe."""
    try:
        probe = ffmpeg.probe(video_path)
        format_info = probe.get("format", {})

        # Find video and audio streams
        video_stream = next(
            (s for s in probe["streams"] if s.get("codec_type") == "video"),
            None,
        )
        audio_stream = next(
            (s for s in probe["streams"] if s.get("codec_type") == "audio"),
            None,
        )

        metadata: VideoMetadata = {
            "filename": Path(format_info.get("filename", "")).name,
            "format": format_info.get("format_name", ""),
            "duration": float(format_info.get("duration", 0)),
            "size": int(format_info.get("size", 0)),
            "bitrate": int(format_info.get("bit_rate", 0)),
        }

        if video_stream:
            video_meta: VideoStreamMetadata = {
                "codec": video_stream.get("codec_name", ""),
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "fps": _parse_frame_rate(video_stream.get("r_frame_rate", "0/1")),
                "aspect_ratio": video_stream.get("display_aspect_ratio", ""),
                "pixel_format": video_stream.get("pix_fmt", ""),
            }
            metadata["video"] = video_meta

        if audio_stream:
            audio_meta: AudioStreamMetadata = {
                "codec": audio_stream.get("codec_name", ""),
                "channels": int(audio_stream.get("channels", 0)),
                "sample_rate": int(audio_stream.get("sample_rate", 0)),
                "bitrate": int(audio_stream.get("bit_rate", 0)),
            }
            metadata["audio"] = audio_meta

        return metadata

    except ffmpeg.Error as e:
        raise RuntimeError(f"Error analyzing video: {e}") from e


def create_standard_output(
    stream: ffmpeg.Stream,
    output_path: str,
    *,
    crf: int | None = None,
    preset: str | None = None,
    video_bitrate: str | None = None,
    audio_bitrate: str | None = None,
    copy_video: bool = False,
    copy_audio: bool = False,
    faststart: bool = False,
    **kwargs: str | int | float,
) -> ffmpeg.Stream:
    """Create an ffmpeg output node with standard encoding settings.

    Defaults produce an H.264 (``libx264``) / AAC / ``yuv420p`` MP4-friendly
    stream, matching the historical behaviour. Quality can be tuned per-call
    and either stream can be stream-copied to avoid re-encoding.

    Args:
        stream: The ffmpeg stream to encode.
        output_path: Destination file path.
        crf: Constant Rate Factor for libx264 (lower = higher quality).
            Ignored when ``copy_video`` is True.
        preset: libx264 speed/quality preset (e.g. ``"ultrafast"``,
            ``"medium"``). Ignored when ``copy_video`` is True.
        video_bitrate: Target video bitrate (e.g. ``"2M"``). Ignored when
            ``copy_video`` is True.
        audio_bitrate: Target audio bitrate (e.g. ``"128k"``). Ignored when
            ``copy_audio`` is True.
        copy_video: Stream-copy the video (``vcodec="copy"``, no ``pix_fmt``)
            for a lossless, near-instant pass-through.
        copy_audio: Stream-copy the audio (``acodec="copy"``).
        faststart: Add ``-movflags +faststart`` (relocates the moov atom for
            progressive playback of MP4/MOV deliverables).
        **kwargs: Extra ffmpeg output options passed through verbatim and
            overriding any of the above (e.g. ``map="0:a?"``).

    Returns:
        The configured ffmpeg output node.
    """
    settings: dict[str, str | int | float] = {}

    if copy_video:
        settings["vcodec"] = "copy"
    else:
        settings["vcodec"] = "libx264"
        settings["pix_fmt"] = "yuv420p"
        if crf is not None:
            settings["crf"] = crf
        if preset is not None:
            settings["preset"] = preset
        if video_bitrate is not None:
            settings["video_bitrate"] = video_bitrate

    if copy_audio:
        settings["acodec"] = "copy"
    else:
        settings["acodec"] = "aac"
        if audio_bitrate is not None:
            settings["audio_bitrate"] = audio_bitrate

    if faststart:
        settings["movflags"] = "+faststart"

    # Caller-provided kwargs win over the computed defaults.
    settings.update(kwargs)
    return ffmpeg.output(stream, output_path, **settings)


COLOR_MAP = {
    "green": "0x00FF00",
    "blue": "0x0000FF",
    "red": "0xFF0000",
    "cyan": "0x00FFFF",
    "magenta": "0xFF00FF",
    "yellow": "0xFFFF00",
    "white": "0xFFFFFF",
    "black": "0x000000",
    "gray": "0x808080",
    "orange": "0xFFA500",
    "purple": "0x800080",
    "pink": "0xFFC0CB",
}


def parse_color(color: str) -> str:
    """Parse color name or hex code to ffmpeg-compatible format."""
    if color.lower() in COLOR_MAP:
        return COLOR_MAP[color.lower()]
    elif color.startswith("#"):
        return "0x" + color[1:]
    elif color.startswith("0x"):
        return color
    else:
        raise ValueError(f"Invalid color format: {color}")


def parse_resolution(
    resolution: str,
) -> tuple[int, int]:
    """Parse resolution string to (width, height) tuple."""
    try:
        width, height = map(int, resolution.split("x"))
        return width, height
    except (ValueError, AttributeError):
        raise ValueError("Resolution must be in format 'WIDTHxHEIGHT'") from None


def parse_size_range(
    size_range: str,
) -> tuple[float, float]:
    """Parse size range string to (min_size, max_size) tuple."""
    try:
        min_size, max_size = map(float, size_range.split(":"))
        if min_size >= max_size:
            raise ValueError("Invalid size range")
        return min_size, max_size
    except (ValueError, AttributeError):
        raise ValueError(
            "Size range must be in format 'min:max' (e.g., '2:8')"
        ) from None


def _parse_frame_rate(frame_rate: str) -> float:
    """Parse frame rate string (e.g., '30/1' or '30000/1001') to float."""
    try:
        if "/" in frame_rate:
            numerator, denominator = frame_rate.split("/")
            return float(Fraction(int(numerator), int(denominator)))
        return float(frame_rate)
    except (ValueError, ZeroDivisionError):
        return 0.0
