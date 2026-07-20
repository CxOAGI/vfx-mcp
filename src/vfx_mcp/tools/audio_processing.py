"""Audio processing tools: extraction, mixing, and enhancement.

This module provides comprehensive audio processing capabilities including
audio extraction from video files, audio mixing, volume adjustment, fade
effects, and EBU R128 loudness normalization. Supports multiple audio formats
with quality control.

All tools validate their input/output paths against the workspace sandbox
(:func:`safe_input_path` / :func:`safe_output_path`) and run their encodes
through :func:`run_ffmpeg_async`, so long operations never block the FastMCP
event loop and respect the global concurrency/timeout controls.

Video preservation:
    The fade/volume/loudness tools accept both audio-only files and videos.
    When the input carries a video stream it is preserved (stream-copied)
    alongside the processed audio; audio-only inputs produce an audio-only
    output as before. Which branch is taken is decided by a runtime probe.

Supported audio formats:
    - mp3: MPEG Audio Layer III (lossy)
    - wav: Waveform Audio File Format (lossless)
    - aac: Advanced Audio Coding (lossy)
    - ogg: Ogg Vorbis (lossy, open source)
    - flac: Free Lossless Audio Codec (lossless)

Example:
    Extract high-quality audio from video:

        await extract_audio(
            input_path="video.mp4",
            output_path="audio.flac",
            format="flac"
        )
"""

from typing import Any

import ffmpeg
from fastmcp import Context, FastMCP

from ..core import (
    handle_ffmpeg_error,
    log_operation,
    run_ffmpeg_async,
    safe_input_path,
    safe_output_path,
    validate_range,
)


def _probe_stream_types(path: str) -> tuple[bool, bool]:
    """Probe ``path`` and report which stream types it contains.

    Args:
        path: Filesystem path to the media file to probe.

    Returns:
        A ``(has_video, has_audio)`` tuple of booleans.

    Raises:
        RuntimeError: If the file cannot be probed by ffmpeg.
    """
    try:
        probe = ffmpeg.probe(path)
    except ffmpeg.Error as e:
        detail = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"Could not probe input '{path}': {detail}") from e

    streams: list[dict[str, Any]] = probe.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return has_video, has_audio


def register_audio_tools(mcp: FastMCP[Any]) -> None:
    """Register audio processing tools with the MCP server.

    Adds comprehensive audio manipulation capabilities including extraction,
    mixing, volume adjustment, fade effects, and loudness normalization to the
    FastMCP server.

    Args:
        mcp: The FastMCP server instance to register tools with.

    Returns:
        None
    """

    @mcp.tool
    async def extract_audio(
        input_path: str,
        output_path: str,
        format: str = "mp3",
        bitrate: str = "192k",
        ctx: Context | None = None,
    ) -> str:
        """Extract audio from a video file.

        Extracts the audio track from a video file and saves it as a separate
        audio file. Supports various output formats with intelligent codec selection
        and quality-based encoding for optimal results.

        Format-specific behavior:
            - wav: Uses PCM encoding (bitrate ignored, lossless)
            - mp3: Uses libmp3lame encoder with specified bitrate
            - aac: Uses AAC encoder with specified bitrate
            - flac: Uses FLAC encoder (bitrate ignored, lossless)
            - ogg: Uses libvorbis with quality-based encoding

        Args:
            input_path: Path to the input video file.
            output_path: Path where the extracted audio will be saved.
            format: Output audio format. Supported: "mp3", "wav", "aac", "flac", "ogg".
            bitrate: Audio bitrate (e.g., "128k", "192k", "320k").
                    Ignored for lossless formats (wav, flac).
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating audio was extracted and saved.

        Raises:
            ValueError: If format is not supported or the input has no audio
                stream to extract.
            RuntimeError: If ffmpeg encounters an error during processing.

        Example:
            Extract high-quality MP3 audio:

                result = await extract_audio(
                    input_path="movie.mp4",
                    output_path="soundtrack.mp3",
                    format="mp3",
                    bitrate="320k"
                )
        """
        supported_formats = [
            "mp3",
            "wav",
            "aac",
            "flac",
            "ogg",
        ]
        if format not in supported_formats:
            raise ValueError(f"Format must be one of: {', '.join(supported_formats)}")

        safe_in = str(safe_input_path(input_path))
        safe_out = str(safe_output_path(output_path))

        # Guard against inputs with no audio stream (e.g. silent Veo 2 clips):
        # mapping ``0:a`` on such a file fails with an opaque ffmpeg error.
        _, has_audio = _probe_stream_types(safe_in)
        if not has_audio:
            raise ValueError(f"Input has no audio stream to extract: {input_path}")

        await log_operation(
            ctx,
            f"Extracting audio as {format} at {bitrate}",
        )

        try:
            stream: Any = ffmpeg.input(safe_in)

            # Extract only the audio stream
            audio_stream: Any = stream["a"]

            if format == "wav":
                output: Any = ffmpeg.output(
                    audio_stream,
                    safe_out,
                    acodec="pcm_s16le",
                )
            else:
                codec_map = {
                    "mp3": "libmp3lame",
                    "aac": "aac",
                    "flac": "flac",
                    "ogg": "libvorbis",
                }

                output_kwargs: dict[str, Any] = {
                    "acodec": codec_map[format],
                }

                # Handle bitrate differently for different formats
                if format == "ogg":
                    # libvorbis uses quality-based encoding (VBR) by default
                    # Use -q:a instead of bitrate for better compatibility
                    if bitrate:
                        # Convert bitrate to approximate quality level
                        bitrate_num = int(bitrate.rstrip("k"))
                        if bitrate_num <= 96:
                            output_kwargs["qscale:a"] = "0"  # ~64kbps
                        elif bitrate_num <= 128:
                            output_kwargs["qscale:a"] = "2"  # ~96kbps
                        elif bitrate_num <= 192:
                            output_kwargs["qscale:a"] = "4"  # ~128kbps
                        elif bitrate_num <= 256:
                            output_kwargs["qscale:a"] = "6"  # ~192kbps
                        else:
                            output_kwargs["qscale:a"] = "8"  # ~256kbps+
                elif format not in ["flac"] and bitrate:
                    output_kwargs["audio_bitrate"] = bitrate

                output = ffmpeg.output(
                    audio_stream,
                    safe_out,
                    **output_kwargs,
                )

            await run_ffmpeg_async(output, ctx=ctx)
            return f"Audio extracted successfully and saved to {output_path}"
        except ffmpeg.Error as e:
            await handle_ffmpeg_error(e, ctx)
            raise  # Re-raise to ensure function returns on all paths

    @mcp.tool
    async def add_audio(
        input_path: str,
        audio_path: str,
        output_path: str,
        replace: bool = True,
        audio_volume: float = 1.0,
        ctx: Context | None = None,
    ) -> str:
        """Add or replace audio in a video file.

        Combines a video file with an audio file. Can either replace the
        existing audio track entirely or mix the new audio with the video's
        original audio.

        Modes:
            - replace (default): the output contains the video from
              ``input_path`` and audio ONLY from ``audio_path`` — the original
              audio track is dropped (``-map 0:v -map 1:a``). Exactly one audio
              stream is produced.
            - mix: the new audio is mixed with the video's original audio via
              the ``amix`` filter with ``normalize=0``, so input levels are
              summed rather than halved. ``audio_volume`` scales the added
              track before mixing; keep it at or below 1.0 (and lower the
              source) to avoid clipping. If the input video has no audio
              stream, mix falls back to using the new audio alone.

        The video stream is stream-copied (``vcodec=copy``) in both modes, so
        no video re-encoding or generational quality loss occurs.

        Args:
            input_path: Path to the input video file.
            audio_path: Path to the audio file to add.
            output_path: Path where the output video will be saved.
            replace: Whether to replace existing audio (True) or mix (False).
            audio_volume: Volume level for the new audio (0.0 to 2.0).
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating audio was added and video saved.

        Raises:
            ValueError: If parameters are out of valid ranges.
            RuntimeError: If ffmpeg encounters an error during processing.
        """
        validate_range(audio_volume, 0.0, 2.0, "Audio volume")

        safe_in = str(safe_input_path(input_path))
        safe_audio = str(safe_input_path(audio_path))
        safe_out = str(safe_output_path(output_path))

        # Determine whether the video actually has an original audio track;
        # mix mode needs it, and its absence turns mix into a plain replace.
        _, input_has_audio = _probe_stream_types(safe_in)
        effective_replace = replace or not input_has_audio

        mode = "replace" if effective_replace else "mix"
        await log_operation(
            ctx,
            f"Adding audio to video (mode: {mode}, volume: {audio_volume})",
        )

        try:
            video_input: Any = ffmpeg.input(safe_in)
            audio_input: Any = ffmpeg.input(safe_audio)

            new_audio: Any = audio_input.audio
            if audio_volume != 1.0:
                new_audio = ffmpeg.filter(new_audio, "volume", audio_volume)

            if effective_replace:
                # Video from input (stream-copied), audio ONLY from the new
                # file: -map 0:v -map 1:a. The original audio is dropped, so
                # the output has a single audio track.
                output: Any = ffmpeg.output(
                    video_input.video,
                    new_audio,
                    safe_out,
                    vcodec="copy",
                    acodec="aac",
                    shortest=None,
                )
            else:  # mix
                # Mix the video's original audio with the new audio. normalize=0
                # sums the inputs at full level instead of halving them.
                mixed_audio: Any = ffmpeg.filter(
                    [video_input.audio, new_audio],
                    "amix",
                    inputs=2,
                    duration="shortest",
                    normalize=0,
                )
                output = ffmpeg.output(
                    video_input.video,
                    mixed_audio,
                    safe_out,
                    vcodec="copy",
                    acodec="aac",
                )

            await run_ffmpeg_async(output, ctx=ctx)
            return f"Audio {mode}d successfully and saved to {output_path}"
        except ffmpeg.Error as e:
            await handle_ffmpeg_error(e, ctx)
            raise  # Re-raise to ensure function returns on all paths

    @mcp.tool
    async def adjust_audio_volume(
        input_path: str,
        output_path: str,
        volume: float,
        ctx: Context | None = None,
    ) -> str:
        """Adjust the volume level of an audio or video file.

        Changes the audio volume by a specified factor. Can be used to make
        audio louder or quieter without changing other properties. When the
        input is a video, its video stream is preserved (stream-copied) and
        only the audio is re-encoded; audio-only inputs produce an audio-only
        output.

        Args:
            input_path: Path to the input audio/video file.
            output_path: Path where the adjusted audio/video will be saved.
            volume: Volume adjustment factor (0.0 to 3.0). 1.0 = no change.
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating volume was adjusted and file saved.

        Raises:
            ValueError: If volume is out of valid range or the input has no
                audio stream.
            RuntimeError: If ffmpeg encounters an error during processing.
        """
        validate_range(volume, 0.0, 3.0, "Volume")

        safe_in = str(safe_input_path(input_path))
        safe_out = str(safe_output_path(output_path))

        has_video, has_audio = _probe_stream_types(safe_in)
        if not has_audio:
            raise ValueError(f"Input has no audio stream to adjust: {input_path}")

        await log_operation(
            ctx,
            f"Adjusting audio volume to {volume}x",
        )

        try:
            stream: Any = ffmpeg.input(safe_in)
            audio: Any = ffmpeg.filter(stream.audio, "volume", volume)

            if has_video:
                # Preserve the video stream (copy) alongside the filtered audio.
                output: Any = ffmpeg.output(
                    stream.video,
                    audio,
                    safe_out,
                    vcodec="copy",
                    acodec="aac",
                )
            else:
                output = ffmpeg.output(audio, safe_out)

            await run_ffmpeg_async(output, ctx=ctx)
            return f"Audio volume adjusted to {volume}x and saved to {output_path}"
        except ffmpeg.Error as e:
            await handle_ffmpeg_error(e, ctx)
            raise  # Re-raise to ensure function returns on all paths

    @mcp.tool
    async def mix_audio(
        audio1_path: str,
        audio2_path: str,
        output_path: str,
        audio1_volume: float = 1.0,
        audio2_volume: float = 1.0,
        ctx: Context | None = None,
    ) -> str:
        """Mix two audio files together.

        Combines two audio tracks into a single output file with adjustable
        volume levels for each input. The ``amix`` filter runs with
        ``normalize=0`` so the inputs are summed at their scaled levels rather
        than halved; keep the per-track volumes low enough (their sum at or
        below 1.0) to avoid clipping.

        Args:
            audio1_path: Path to the first audio file.
            audio2_path: Path to the second audio file.
            output_path: Path where the mixed audio will be saved.
            audio1_volume: Volume level for first audio (0.0 to 2.0).
            audio2_volume: Volume level for second audio (0.0 to 2.0).
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating audio files were mixed and saved.

        Raises:
            ValueError: If volume levels are out of valid range.
            RuntimeError: If ffmpeg encounters an error during processing.
        """
        validate_range(audio1_volume, 0.0, 2.0, "Audio1 volume")
        validate_range(audio2_volume, 0.0, 2.0, "Audio2 volume")

        safe_in1 = str(safe_input_path(audio1_path))
        safe_in2 = str(safe_input_path(audio2_path))
        safe_out = str(safe_output_path(output_path))

        await log_operation(
            ctx,
            f"Mixing audio files (vol1: {audio1_volume}, vol2: {audio2_volume})",
        )

        try:
            audio1: Any = ffmpeg.input(safe_in1)
            audio2: Any = ffmpeg.input(safe_in2)

            # Apply volume adjustments if needed
            if audio1_volume != 1.0:
                audio1 = ffmpeg.filter(audio1, "volume", audio1_volume)
            if audio2_volume != 1.0:
                audio2 = ffmpeg.filter(audio2, "volume", audio2_volume)

            # Mix the audio tracks. normalize=0 preserves the summed level.
            mixed_audio: Any = ffmpeg.filter(
                [audio1, audio2],
                "amix",
                inputs=2,
                duration="longest",
                normalize=0,
            )
            output: Any = ffmpeg.output(mixed_audio, safe_out)
            await run_ffmpeg_async(output, ctx=ctx)
            return f"Audio files mixed successfully and saved to {output_path}"
        except ffmpeg.Error as e:
            await handle_ffmpeg_error(e, ctx)
            raise  # Re-raise to ensure function returns on all paths

    @mcp.tool
    async def audio_fade_in(
        input_path: str,
        output_path: str,
        duration: float,
        ctx: Context | None = None,
    ) -> str:
        """Apply a fade-in effect to audio.

        Gradually increases the audio volume from silence to full volume
        over the specified duration at the beginning of the audio. When the
        input is a video, its video stream is preserved (stream-copied) and
        only the audio is faded; audio-only inputs produce an audio-only
        output.

        Args:
            input_path: Path to the input audio/video file.
            output_path: Path where the fade-in audio/video will be saved.
            duration: Duration of fade-in effect in seconds (0.1 to 10.0).
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating fade-in was applied and file saved.

        Raises:
            ValueError: If duration is out of valid range or the input has no
                audio stream.
            RuntimeError: If ffmpeg encounters an error during processing.
        """
        validate_range(duration, 0.1, 10.0, "Fade duration")

        safe_in = str(safe_input_path(input_path))
        safe_out = str(safe_output_path(output_path))

        has_video, has_audio = _probe_stream_types(safe_in)
        if not has_audio:
            raise ValueError(f"Input has no audio stream to fade: {input_path}")

        await log_operation(
            ctx,
            f"Applying {duration}s fade-in effect",
        )

        try:
            stream: Any = ffmpeg.input(safe_in)
            audio: Any = ffmpeg.filter(
                stream.audio, "afade", type="in", duration=duration
            )

            if has_video:
                output: Any = ffmpeg.output(
                    stream.video,
                    audio,
                    safe_out,
                    vcodec="copy",
                    acodec="aac",
                )
            else:
                output = ffmpeg.output(audio, safe_out)

            await run_ffmpeg_async(output, ctx=ctx)
            return f"Fade-in effect applied ({duration}s) and saved to {output_path}"
        except ffmpeg.Error as e:
            await handle_ffmpeg_error(e, ctx)
            raise  # Re-raise to ensure function returns on all paths

    @mcp.tool
    async def audio_fade_out(
        input_path: str,
        output_path: str,
        duration: float,
        ctx: Context | None = None,
    ) -> str:
        """Apply a fade-out effect to audio.

        Gradually decreases the audio volume from full volume to silence
        over the specified duration at the end of the audio. When the input is
        a video, its video stream is preserved (stream-copied) and only the
        audio is faded; audio-only inputs produce an audio-only output.

        Args:
            input_path: Path to the input audio/video file.
            output_path: Path where the fade-out audio/video will be saved.
            duration: Duration of fade-out effect in seconds (0.1 to 10.0).
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating fade-out was applied and file saved.

        Raises:
            ValueError: If duration is out of valid range or the input has no
                audio stream.
            RuntimeError: If ffmpeg encounters an error during processing.
        """
        validate_range(duration, 0.1, 10.0, "Fade duration")

        safe_in = str(safe_input_path(input_path))
        safe_out = str(safe_output_path(output_path))

        has_video, has_audio = _probe_stream_types(safe_in)
        if not has_audio:
            raise ValueError(f"Input has no audio stream to fade: {input_path}")

        await log_operation(
            ctx,
            f"Applying {duration}s fade-out effect",
        )

        try:
            stream: Any = ffmpeg.input(safe_in)
            audio: Any = ffmpeg.filter(
                stream.audio, "afade", type="out", duration=duration
            )

            if has_video:
                output: Any = ffmpeg.output(
                    stream.video,
                    audio,
                    safe_out,
                    vcodec="copy",
                    acodec="aac",
                )
            else:
                output = ffmpeg.output(audio, safe_out)

            await run_ffmpeg_async(output, ctx=ctx)
            return f"Fade-out effect applied ({duration}s) and saved to {output_path}"
        except ffmpeg.Error as e:
            await handle_ffmpeg_error(e, ctx)
            raise  # Re-raise to ensure function returns on all paths

    @mcp.tool
    async def normalize_loudness(
        input_path: str,
        output_path: str,
        target_i: float = -14.0,
        target_tp: float = -1.0,
        target_lra: float = 11.0,
        ctx: Context | None = None,
    ) -> str:
        """Normalize perceived loudness using the EBU R128 ``loudnorm`` filter.

        Adjusts the integrated loudness of a clip's audio to a consistent
        target so that clips stitched from different sources sound level.
        Defaults target -14 LUFS integrated with -1 dBTP true-peak headroom,
        matching common streaming delivery targets. When the input is a video,
        its video stream is preserved (stream-copied) and only the audio is
        normalized; audio-only inputs produce an audio-only output.

        Note: this is a single-pass (dynamic) normalization. For the most
        accurate results a two-pass measurement is preferred, but single-pass
        is well-suited to per-clip levelling ahead of stitching.

        Args:
            input_path: Path to the input audio/video file.
            output_path: Path where the normalized audio/video will be saved.
            target_i: Target integrated loudness in LUFS (-70.0 to -5.0).
            target_tp: Maximum true peak in dBTP (-9.0 to 0.0).
            target_lra: Target loudness range in LU (1.0 to 50.0).
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating loudness was normalized and file saved.

        Raises:
            ValueError: If a target is out of valid range or the input has no
                audio stream.
            RuntimeError: If ffmpeg encounters an error during processing.
        """
        validate_range(target_i, -70.0, -5.0, "Target integrated loudness")
        validate_range(target_tp, -9.0, 0.0, "Target true peak")
        validate_range(target_lra, 1.0, 50.0, "Target loudness range")

        safe_in = str(safe_input_path(input_path))
        safe_out = str(safe_output_path(output_path))

        has_video, has_audio = _probe_stream_types(safe_in)
        if not has_audio:
            raise ValueError(f"Input has no audio stream to normalize: {input_path}")

        await log_operation(
            ctx,
            f"Normalizing loudness to {target_i} LUFS (TP {target_tp} dBTP, "
            f"LRA {target_lra} LU)",
        )

        try:
            stream: Any = ffmpeg.input(safe_in)
            audio: Any = ffmpeg.filter(
                stream.audio,
                "loudnorm",
                i=target_i,
                tp=target_tp,
                lra=target_lra,
            )

            if has_video:
                output: Any = ffmpeg.output(
                    stream.video,
                    audio,
                    safe_out,
                    vcodec="copy",
                    acodec="aac",
                )
            else:
                output = ffmpeg.output(audio, safe_out)

            await run_ffmpeg_async(output, ctx=ctx)
            return f"Loudness normalized to {target_i} LUFS and saved to {output_path}"
        except ffmpeg.Error as e:
            await handle_ffmpeg_error(e, ctx)
            raise  # Re-raise to ensure function returns on all paths
