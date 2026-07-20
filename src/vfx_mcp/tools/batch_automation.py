"""Batch processing and automation tools.

This module implements the Phase 3 batch/manifest tooling that lets an
omni/veo-style pipeline produce a finished deliverable in a *single* MCP call:

- ``stitch_from_manifest`` takes an ordered list of clip descriptors (each with
  optional in/out trim points and an optional transition) and stitches them
  into one video. Junctions with no transition are plain cuts (via the
  ``concat`` filter); junctions with a transition are crossfades (via ``xfade``
  for video and ``acrossfade`` for audio). Clips are normalized to a common
  resolution, frame rate, SAR and pixel format first, and silent clips are
  backed by ``anullsrc`` so the audio graph always has two inputs to blend.
- ``batch_convert`` converts many inputs to a target container concurrently,
  with the number of simultaneous ffmpeg processes bounded by the shared
  ``run_ffmpeg_async`` semaphore so a large batch cannot overwhelm the host.

Example:
    Register tools with an MCP server::

        mcp = FastMCP('video-editor')
        register_automation_tools(mcp)
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import ffmpeg
from fastmcp import Context, FastMCP

from ..core import (
    create_standard_output,
    get_video_metadata,
    handle_ffmpeg_error,
    log_operation,
    run_ffmpeg_async,
    safe_input_path,
    safe_output_path,
    validate_transition_type,
)

# Map the review's transition vocabulary (as normalized by
# ``validate_transition_type``) onto the names understood by FFmpeg's ``xfade``
# filter. ``crossfade`` is an alias for a plain fade.
_XFADE_TRANSITION_MAP: dict[str, str] = {
    "fade": "fade",
    "crossfade": "fade",
    "dissolve": "dissolve",
    "wipe_left": "wipeleft",
    "wipe_right": "wiperight",
    "wipe_up": "wipeup",
    "wipe_down": "wipedown",
    "slide_left": "slideleft",
    "slide_right": "slideright",
}

# Canonical audio format used for the transition/concat graph so that clips with
# differing sample rates / channel layouts (and injected silence) join cleanly.
_TARGET_SAMPLE_RATE = 44100
_TARGET_CHANNEL_LAYOUT = "stereo"

# Default crossfade length (seconds) when a manifest entry requests a transition
# but does not specify ``transition_duration``.
_DEFAULT_TRANSITION_DURATION = 1.0

# Containers whose muxer supports ``-movflags +faststart`` for progressive
# playback of the delivered file.
_FASTSTART_FORMATS: frozenset[str] = frozenset({"mp4", "mov", "m4v", "m4a"})


@dataclass(frozen=True)
class _ManifestEntry:
    """A validated single manifest entry describing one clip in the sequence."""

    clip: str
    start: float | None
    end: float | None
    transition: str | None
    transition_duration: float | None


def _even(value: int) -> int:
    """Round ``value`` up to the nearest even integer (libx264/yuv420p safe)."""
    value = max(value, 2)
    return value if value % 2 == 0 else value + 1


def _as_optional_number(value: object, field: str, index: int) -> float | None:
    """Coerce a manifest value to ``float | None`` with a clear error.

    ``bool`` is rejected explicitly because it is a subclass of ``int`` and a
    boolean here is almost certainly a mistake.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"manifest[{index}].{field} must be a number or null, "
            f"got {type(value).__name__}"
        )
    return float(value)


def _parse_manifest(manifest: list[dict[str, object]]) -> list[_ManifestEntry]:
    """Validate raw manifest input and return typed entries.

    Args:
        manifest: The caller-supplied manifest (list of dict descriptors).

    Returns:
        A list of validated :class:`_ManifestEntry` objects.

    Raises:
        ValueError: If the manifest is empty, an entry is not a mapping, a
            ``clip`` is missing/blank, or a field has the wrong type.
    """
    if not isinstance(manifest, list):
        raise ValueError("manifest must be a list of clip descriptors")
    if not manifest:
        raise ValueError("manifest must contain at least one entry")

    entries: list[_ManifestEntry] = []
    for index, raw in enumerate(manifest):
        if not isinstance(raw, dict):
            raise ValueError(
                f"manifest[{index}] must be an object, got {type(raw).__name__}"
            )

        clip = raw.get("clip")
        if not isinstance(clip, str) or not clip.strip():
            raise ValueError(
                f"manifest[{index}].clip is required and must be a non-empty string"
            )

        start = _as_optional_number(raw.get("start"), "start", index)
        end = _as_optional_number(raw.get("end"), "end", index)
        if start is not None and start < 0:
            raise ValueError(f"manifest[{index}].start must be >= 0")
        if end is not None and start is not None and end <= start:
            raise ValueError(
                f"manifest[{index}].end ({end}) must be greater than start ({start})"
            )

        transition_raw = raw.get("transition")
        transition: str | None
        if transition_raw is None:
            transition = None
        elif isinstance(transition_raw, str):
            transition = transition_raw
        else:
            raise ValueError(
                f"manifest[{index}].transition must be a string or null, "
                f"got {type(transition_raw).__name__}"
            )

        transition_duration = _as_optional_number(
            raw.get("transition_duration"), "transition_duration", index
        )
        if transition_duration is not None and transition_duration <= 0:
            raise ValueError(
                f"manifest[{index}].transition_duration must be greater than 0"
            )

        entries.append(
            _ManifestEntry(
                clip=clip,
                start=start,
                end=end,
                transition=transition,
                transition_duration=transition_duration,
            )
        )

    return entries


def _clip_duration(entry: _ManifestEntry, full_duration: float) -> float:
    """Return the effective (post-trim) duration for a manifest entry."""
    start = entry.start or 0.0
    end = entry.end if entry.end is not None else full_duration
    return max(end - start, 0.0)


def _normalize_video(
    stream: ffmpeg.Stream,
    entry: _ManifestEntry,
    *,
    width: int,
    height: int,
    fps: int,
) -> ffmpeg.Stream:
    """Trim (if requested) then conform a video stream to a canonical shape.

    ``xfade`` and the ``concat`` filter both require their inputs to share an
    identical resolution, pixel format, SAR and frame rate, so every clip is
    conformed to the same target. Aspect ratio is preserved by scaling to fit
    and letterbox/pillarbox padding the remainder.
    """
    if entry.start is not None or entry.end is not None:
        trim_kwargs: dict[str, float] = {}
        if entry.start is not None:
            trim_kwargs["start"] = entry.start
        if entry.end is not None:
            trim_kwargs["end"] = entry.end
        stream = ffmpeg.filter(stream, "trim", **trim_kwargs)
        stream = ffmpeg.filter(stream, "setpts", "PTS-STARTPTS")

    stream = ffmpeg.filter(
        stream,
        "scale",
        width,
        height,
        force_original_aspect_ratio="decrease",
    )
    stream = ffmpeg.filter(
        stream,
        "pad",
        width,
        height,
        "(ow-iw)/2",
        "(oh-ih)/2",
    )
    stream = ffmpeg.filter(stream, "setsar", "1")
    stream = ffmpeg.filter(stream, "fps", fps)
    return ffmpeg.filter(stream, "format", "yuv420p")


def _normalize_audio(stream: ffmpeg.Stream, entry: _ManifestEntry) -> ffmpeg.Stream:
    """Trim (if requested) then conform audio to the canonical format."""
    if entry.start is not None or entry.end is not None:
        atrim_kwargs: dict[str, float] = {}
        if entry.start is not None:
            atrim_kwargs["start"] = entry.start
        if entry.end is not None:
            atrim_kwargs["end"] = entry.end
        stream = ffmpeg.filter(stream, "atrim", **atrim_kwargs)
        stream = ffmpeg.filter(stream, "asetpts", "PTS-STARTPTS")

    return ffmpeg.filter(
        stream,
        "aformat",
        sample_rates=_TARGET_SAMPLE_RATE,
        channel_layouts=_TARGET_CHANNEL_LAYOUT,
    )


def _silent_audio(duration: float) -> ffmpeg.Stream:
    """Create a bounded silent audio source for a clip with no audio track."""
    return ffmpeg.input(
        f"anullsrc=channel_layout={_TARGET_CHANNEL_LAYOUT}"
        f":sample_rate={_TARGET_SAMPLE_RATE}",
        f="lavfi",
        t=duration,
    )


def _build_manifest_streams(
    entries: list[_ManifestEntry],
    resolved_paths: list[str],
    durations: list[float],
    has_audio: list[bool],
    *,
    target_width: int,
    target_height: int,
    target_fps: int,
) -> tuple[ffmpeg.Stream, ffmpeg.Stream]:
    """Build the video/audio filtergraph for a manifest.

    Each successive clip is joined to the running accumulator either with a
    plain ``concat`` cut (when the entry has no transition) or with an
    ``xfade``/``acrossfade`` crossfade (when a transition is specified). The
    ``xfade`` ``offset`` for a crossfade is the cumulative duration of
    everything already stitched minus the transition length.

    This function performs no probing (durations/audio flags are supplied by the
    caller), so the resulting graph can be inspected via ``ffmpeg.get_args``.

    Returns:
        A ``(video_stream, audio_stream)`` tuple ready to be muxed.
    """
    video_streams: list[ffmpeg.Stream] = []
    audio_streams: list[ffmpeg.Stream] = []

    for entry, path, duration, clip_has_audio in zip(
        entries, resolved_paths, durations, has_audio, strict=True
    ):
        source = ffmpeg.input(path)
        video_streams.append(
            _normalize_video(
                source.video,
                entry,
                width=target_width,
                height=target_height,
                fps=target_fps,
            )
        )
        if clip_has_audio:
            audio_streams.append(_normalize_audio(source.audio, entry))
        else:
            # Silence spanning exactly the (possibly trimmed) clip length; the
            # trim points are already baked into ``duration``.
            audio_streams.append(
                _normalize_audio(
                    _silent_audio(duration),
                    _ManifestEntry(
                        clip=entry.clip,
                        start=None,
                        end=None,
                        transition=entry.transition,
                        transition_duration=entry.transition_duration,
                    ),
                )
            )

    video_acc = video_streams[0]
    audio_acc = audio_streams[0]
    cumulative = durations[0]

    for index in range(1, len(entries)):
        entry = entries[index]
        if entry.transition is None:
            # Plain cut: concatenate video and audio pairwise.
            video_acc = ffmpeg.concat(video_acc, video_streams[index], v=1, a=0)
            audio_acc = ffmpeg.concat(audio_acc, audio_streams[index], v=0, a=1)
            cumulative += durations[index]
        else:
            transition_duration = (
                entry.transition_duration
                if entry.transition_duration is not None
                else _DEFAULT_TRANSITION_DURATION
            )
            xfade_name = _XFADE_TRANSITION_MAP[
                validate_transition_type(entry.transition)
            ]
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


def register_automation_tools(
    mcp: FastMCP[None],
) -> None:
    """Register batch automation tools with the MCP server.

    Adds ``stitch_from_manifest`` (single-call manifest stitching) and
    ``batch_convert`` (bounded-concurrency format conversion) to the provided
    FastMCP server instance.

    Args:
        mcp: The FastMCP server instance to register tools with.

    Returns:
        None
    """

    @mcp.tool
    async def stitch_from_manifest(
        manifest: list[dict[str, object]],
        output_path: str,
        crf: int | None = None,
        preset: str | None = None,
        faststart: bool = True,
        ctx: Context | None = None,
    ) -> str:
        """Stitch an ordered manifest of clips into one deliverable.

        Each manifest entry describes one clip and looks like::

            {
                "clip": "shot_01.mp4",       # required, path within workspace
                "start": 0.0,                 # optional in-point (seconds)
                "end": 3.5,                   # optional out-point (seconds)
                "transition": "fade",         # optional; omit/null for a cut
                "transition_duration": 1.0    # optional crossfade length (s)
            }

        Clips are joined in list order. For each entry after the first, a
        missing/null ``transition`` produces a plain cut, while a named
        transition produces an ``xfade`` (video) + ``acrossfade`` (audio)
        crossfade of ``transition_duration`` seconds (default
        ``1.0``). The ``transition`` on the first entry is ignored (nothing
        precedes it).

        All clips are normalized to a common resolution (the largest input,
        rounded to even dimensions), frame rate (the fastest input), SAR and
        pixel format so heterogeneous sources stitch cleanly, and clips with no
        audio track are backed by silence so audio joins always succeed.

        The output duration is approximately
        ``sum(clip_durations) - sum(transition_durations)``.

        Args:
            manifest: Ordered list of clip descriptors (see above). At least one
                entry is required.
            output_path: Destination path for the stitched video.
            crf: Optional libx264 Constant Rate Factor (lower = higher quality).
            preset: Optional libx264 speed/quality preset (e.g. ``medium``).
            faststart: Relocate the moov atom for progressive playback
                (``-movflags +faststart``). Defaults to ``True``.
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating the stitched video was saved.

        Raises:
            ValueError: If the manifest is malformed, a clip is missing, or a
                transition duration is not shorter than the clips it joins.
            FileNotFoundError: If a referenced clip does not exist.
            RuntimeError: If ffmpeg encounters an error during processing.
        """
        entries = _parse_manifest(manifest)

        resolved_inputs = [safe_input_path(entry.clip) for entry in entries]
        resolved_output = safe_output_path(output_path)

        await log_operation(
            ctx,
            f"Stitching {len(entries)} clips from manifest into {output_path}",
        )

        # Probe every clip for duration, geometry, fps and audio presence.
        durations: list[float] = []
        has_audio: list[bool] = []
        target_width = 0
        target_height = 0
        target_fps = 0.0

        for entry, path in zip(entries, resolved_inputs, strict=True):
            metadata = get_video_metadata(str(path))
            video_meta = metadata.get("video")
            if video_meta is None:
                raise ValueError(f"No video stream found in '{path.name}'")

            full_duration = metadata["duration"]
            if entry.end is not None and entry.end > full_duration:
                raise ValueError(
                    f"'{path.name}' end ({entry.end}s) exceeds its duration "
                    f"({full_duration}s)"
                )

            clip_duration = _clip_duration(entry, full_duration)
            durations.append(clip_duration)
            has_audio.append("audio" in metadata)

            target_width = max(target_width, video_meta["width"])
            target_height = max(target_height, video_meta["height"])
            target_fps = max(target_fps, video_meta["fps"])

        # Validate transition lengths against the clips they blend.
        cumulative = durations[0]
        for index in range(1, len(entries)):
            entry = entries[index]
            if entry.transition is not None:
                transition_duration = (
                    entry.transition_duration
                    if entry.transition_duration is not None
                    else _DEFAULT_TRANSITION_DURATION
                )
                if durations[index] <= transition_duration:
                    raise ValueError(
                        f"transition_duration ({transition_duration}s) must be "
                        f"shorter than clip {index} "
                        f"('{resolved_inputs[index].name}', "
                        f"{durations[index]}s)"
                    )
                if cumulative < transition_duration:
                    raise ValueError(
                        f"transition_duration ({transition_duration}s) is "
                        f"longer than the stitched content preceding clip {index}"
                    )
                cumulative += durations[index] - transition_duration
            else:
                cumulative += durations[index]

        target_width = _even(target_width)
        target_height = _even(target_height)
        fps = round(target_fps) if target_fps > 0 else 30

        try:
            video_stream, audio_stream = _build_manifest_streams(
                entries,
                [str(path) for path in resolved_inputs],
                durations,
                has_audio,
                target_width=target_width,
                target_height=target_height,
                target_fps=fps,
            )

            # The video (xfade/concat) and audio (acrossfade/concat) streams are
            # separate graph outputs, so mux them directly while mirroring the
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
            await run_ffmpeg_async(output, ctx=ctx)
            return (
                f"Stitched {len(entries)} clips from manifest and saved to "
                f"{output_path}"
            )
        except ffmpeg.Error as e:
            await handle_ffmpeg_error(e, ctx)
            raise

    # Ensure the function is registered with MCP.
    del stitch_from_manifest

    @mcp.tool
    async def batch_convert(
        input_paths: list[str],
        output_dir: str,
        format: str = "mp4",
        crf: int | None = None,
        preset: str | None = None,
        ctx: Context | None = None,
    ) -> str:
        """Convert many videos to a target container concurrently.

        Each input is converted to ``<output_dir>/<stem>.<format>`` with the
        standard libx264/aac settings. Conversions are launched together but the
        number of ffmpeg processes actually running at once is bounded by the
        shared ``run_ffmpeg_async`` semaphore (``VFX_MAX_CONCURRENCY``, default
        3), so a large batch cannot overwhelm the host.

        Args:
            input_paths: Paths to the source videos (within the workspace).
            output_dir: Directory that will hold the converted files (created if
                necessary).
            format: Target container/extension (e.g. ``mp4``, ``mkv``,
                ``webm``). Defaults to ``mp4``.
            crf: Optional Constant Rate Factor for the video encoder.
            preset: Optional encoder speed/quality preset.
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message listing the converted output files.

        Raises:
            ValueError: If no inputs are given or a path escapes the workspace.
            FileNotFoundError: If a referenced input does not exist.
            RuntimeError: If any conversion fails.
        """
        if not input_paths:
            raise ValueError("batch_convert requires at least one input path")

        ext = format.lower().lstrip(".")
        faststart = ext in _FASTSTART_FORMATS

        # Resolve/validate everything up front so a bad path fails fast before
        # any encode is launched.
        resolved_inputs = [safe_input_path(path) for path in input_paths]
        resolved_outputs = [
            safe_output_path(str(Path(output_dir) / f"{src.stem}.{ext}"))
            for src in resolved_inputs
        ]

        await log_operation(
            ctx,
            f"Batch converting {len(resolved_inputs)} file(s) to '{ext}' "
            f"in {output_dir}",
        )

        async def _convert_one(src: Path, dst: Path) -> str:
            stream = ffmpeg.input(str(src))
            output = create_standard_output(
                stream,
                str(dst),
                crf=crf,
                preset=preset,
                faststart=faststart,
            )
            try:
                await run_ffmpeg_async(output, ctx=ctx)
            except ffmpeg.Error as e:
                await handle_ffmpeg_error(e, ctx)
                raise
            return str(dst)

        results = await asyncio.gather(
            *(
                _convert_one(src, dst)
                for src, dst in zip(resolved_inputs, resolved_outputs, strict=True)
            ),
            return_exceptions=True,
        )

        succeeded: list[str] = []
        failures: list[str] = []
        for src, result in zip(resolved_inputs, results, strict=True):
            if isinstance(result, BaseException):
                failures.append(f"{src.name}: {result}")
            else:
                succeeded.append(result)

        if failures:
            raise RuntimeError(
                f"batch_convert failed for {len(failures)} of "
                f"{len(resolved_inputs)} file(s): " + "; ".join(failures)
            )

        return f"Converted {len(succeeded)} file(s) to '{ext}': " + ", ".join(succeeded)

    # Ensure the function is registered with MCP.
    del batch_convert
