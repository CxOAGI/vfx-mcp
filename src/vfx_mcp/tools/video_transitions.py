"""Video transition and sequencing tools.

This module implements N-clip stitching with crossfade-style transitions using
FFmpeg's ``xfade`` (video) and ``acrossfade`` (audio) filters, chained pairwise
across the input clips. Clips are normalized to a common resolution, frame
rate, SAR and pixel format before the transition graph is built so that
heterogeneous sources (e.g. Veo/omni clips of differing geometry) can be
stitched cleanly. Silent clips are backed by an ``anullsrc`` audio source so
that ``acrossfade`` always has two audio inputs to work with.

Example:
    Register tools with an MCP server::

        mcp = FastMCP('video-editor')
        register_transition_tools(mcp)
"""

import ffmpeg
from fastmcp import Context, FastMCP

from ..core import (
    XFADE_TRANSITION_MAP,
    even_dimension,
    get_video_metadata,
    log_operation,
    normalize_audio_stream,
    normalize_video_stream,
    run_ffmpeg_async,
    safe_input_path,
    safe_output_path,
    silent_audio_source,
    validate_transition_type,
)


def _build_stitch_streams(
    input_paths: list[str],
    durations: list[float],
    has_audio: list[bool],
    *,
    target_width: int,
    target_height: int,
    target_fps: int,
    xfade_name: str,
    transition_duration: float,
) -> tuple[ffmpeg.Stream, ffmpeg.Stream]:
    """Build the pairwise xfade/acrossfade filtergraph for ``input_paths``.

    Each successive clip is blended into the running accumulator with an
    ``xfade`` (video) and ``acrossfade`` (audio). The ``xfade`` ``offset`` for
    the i-th blend is the cumulative duration of everything already stitched
    minus one transition duration, i.e. ``sum(d[0..i-1]) - i * T``. The final
    output therefore lasts ``sum(durations) - (n - 1) * T``.

    This function performs no probing (durations/audio flags are supplied by the
    caller), so the resulting graph can be inspected via ``ffmpeg.get_args``.

    Returns:
        A ``(video_stream, audio_stream)`` tuple ready to be muxed.
    """
    video_streams: list[ffmpeg.Stream] = []
    audio_streams: list[ffmpeg.Stream] = []

    for path, duration, clip_has_audio in zip(
        input_paths, durations, has_audio, strict=True
    ):
        source = ffmpeg.input(path)
        video_streams.append(
            normalize_video_stream(
                source.video,
                width=target_width,
                height=target_height,
                fps=target_fps,
            )
        )
        if clip_has_audio:
            audio_streams.append(normalize_audio_stream(source.audio))
        else:
            audio_streams.append(normalize_audio_stream(silent_audio_source(duration)))

    video_acc = video_streams[0]
    audio_acc = audio_streams[0]
    cumulative = durations[0]

    for index in range(1, len(input_paths)):
        offset = cumulative - transition_duration
        video_acc = ffmpeg.filter(
            [video_acc, video_streams[index]],
            "xfade",
            transition=xfade_name,
            duration=transition_duration,
            offset=offset,
        )
        audio_acc = ffmpeg.filter(
            [audio_acc, audio_streams[index]],
            "acrossfade",
            d=transition_duration,
        )
        cumulative += durations[index] - transition_duration

    return video_acc, audio_acc


def register_transition_tools(
    mcp: FastMCP[None],
) -> None:
    """Register video transition tools with the MCP server.

    Args:
        mcp: The FastMCP server instance to register tools with.

    Returns:
        None
    """

    @mcp.tool
    async def stitch_with_transitions(
        input_paths: list[str],
        output_path: str,
        transition: str = "fade",
        duration: float = 1.0,
        crf: int | None = None,
        preset: str | None = None,
        faststart: bool = True,
        ctx: Context | None = None,
    ) -> str:
        """Stitch multiple clips together with crossfade transitions.

        Joins ``input_paths`` in order, blending each successive clip into the
        previous one with an ``xfade`` video transition and an ``acrossfade``
        audio transition of ``duration`` seconds. Clips are first normalized to
        a common resolution (the largest input, rounded to even dimensions),
        frame rate (the fastest input), SAR and pixel format, so heterogeneous
        sources stitch cleanly. Clips with no audio track are backed by silence
        so the audio crossfade always succeeds.

        The output duration is approximately
        ``sum(clip_durations) - (n - 1) * duration``.

        Args:
            input_paths: Paths to the clips to stitch, in order (minimum 2).
            output_path: Destination path for the stitched video.
            transition: Transition style. One of ``fade``, ``crossfade``,
                ``dissolve``, ``wipe_left``, ``wipe_right``, ``wipe_up``,
                ``wipe_down``, ``slide_left`` or ``slide_right``.
            duration: Transition duration in seconds (must be shorter than the
                shortest clip).
            crf: Optional libx264 Constant Rate Factor (lower = higher quality).
            preset: Optional libx264 speed/quality preset (e.g. ``medium``).
            faststart: Relocate the moov atom for progressive playback
                (``-movflags +faststart``). Defaults to ``True``.
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating the stitched video was saved.

        Raises:
            ValueError: If fewer than 2 clips are given, the transition is
                unknown, or ``duration`` is not shorter than every clip.
            RuntimeError: If ffmpeg encounters an error during processing.
        """
        if len(input_paths) < 2:
            raise ValueError("At least 2 clips are required for stitching")

        if duration <= 0:
            raise ValueError("Transition duration must be greater than 0")

        # Validate against the review vocabulary, then map to the xfade name.
        validated = validate_transition_type(transition)
        xfade_name = XFADE_TRANSITION_MAP[validated]

        resolved_inputs = [safe_input_path(path) for path in input_paths]
        resolved_output = safe_output_path(output_path)

        await log_operation(
            ctx,
            f"Stitching {len(resolved_inputs)} clips with "
            f"'{validated}' transitions ({duration}s each)",
        )

        # Probe every clip for duration, geometry, fps and audio presence.
        durations: list[float] = []
        has_audio: list[bool] = []
        target_width = 0
        target_height = 0
        target_fps = 0.0

        for path in resolved_inputs:
            metadata = get_video_metadata(str(path))
            clip_duration = metadata["duration"]
            durations.append(clip_duration)
            has_audio.append("audio" in metadata)

            if clip_duration <= duration:
                raise ValueError(
                    "Transition duration "
                    f"({duration}s) must be shorter than every clip; "
                    f"'{path.name}' is only {clip_duration}s"
                )

            video_meta = metadata.get("video")
            if video_meta is None:
                raise ValueError(f"No video stream found in '{path.name}'")
            target_width = max(target_width, video_meta["width"])
            target_height = max(target_height, video_meta["height"])
            target_fps = max(target_fps, video_meta["fps"])

        target_width = even_dimension(target_width)
        target_height = even_dimension(target_height)
        # Fall back to 30fps if probing could not determine a frame rate.
        fps = round(target_fps) if target_fps > 0 else 30

        video_stream, audio_stream = _build_stitch_streams(
            [str(path) for path in resolved_inputs],
            durations,
            has_audio,
            target_width=target_width,
            target_height=target_height,
            target_fps=fps,
            xfade_name=xfade_name,
            transition_duration=duration,
        )

        # ``create_standard_output`` only supports a single output stream;
        # here we must mux the separate xfade video and acrossfade audio
        # streams, so build the output directly while mirroring its
        # standard libx264/yuv420p/aac settings and quality controls.
        output_kwargs: dict[str, str | int | float] = {
            "vcodec": "libx264",
            "pix_fmt": "yuv420p",
            "acodec": "aac",
        }
        if crf is not None:
            output_kwargs["crf"] = crf
        if preset is not None:
            output_kwargs["preset"] = preset
        if faststart:
            output_kwargs["movflags"] = "+faststart"

        output = ffmpeg.output(
            video_stream,
            audio_stream,
            str(resolved_output),
            **output_kwargs,
        )
        await run_ffmpeg_async(output, ctx=ctx, output_path=str(resolved_output))
        return (
            f"Stitched {len(resolved_inputs)} clips with '{validated}' "
            f"transitions and saved to {output_path}"
        )

    # Ensure the function is registered with MCP.
    del stitch_with_transitions
