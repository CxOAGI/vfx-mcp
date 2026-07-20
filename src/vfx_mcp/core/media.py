"""Canonical media stitching, normalization and format helpers.

This module is the single source of truth for the stream-normalization and
format primitives that used to be copy-pasted (and quietly diverge) across
``basic_video_ops``, ``video_transitions`` and ``batch_automation``. Keeping
them here guarantees every stitching/concat/xfade path conforms clips to the
*same* geometry, frame rate, SAR, pixel format and audio layout, so
heterogeneous sources join cleanly.

The helpers perform no probing and no I/O; they only build ffmpeg-python
stream nodes, so any graph assembled from them can be inspected via
``ffmpeg.get_args`` in tests without the ffmpeg binary being present.
"""

import ffmpeg

# Canonical audio target used whenever heterogeneous clips (including injected
# silence) must be conformed before a concat/xfade/acrossfade join. Ints render
# identically to the historical string values inside the ffmpeg filter graph.
TARGET_SAMPLE_RATE: int = 44100
TARGET_CHANNEL_LAYOUT: str = "stereo"

# Map the review's transition vocabulary (as accepted/normalized by
# ``validate_transition_type``) onto the transition names understood by
# FFmpeg's ``xfade`` filter. ``crossfade`` is an alias for a plain fade.
XFADE_TRANSITION_MAP: dict[str, str] = {
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

# Fallback codecs used when neither the caller nor the target ``format`` picks
# a codec. libx264/aac is the most broadly compatible MP4 pairing.
DEFAULT_VIDEO_CODEC: str = "libx264"
DEFAULT_AUDIO_CODEC: str = "aac"

# Per-container codec defaults. Passing ``format`` only sets these DEFAULTS; an
# explicit ``video_codec``/``audio_codec`` argument always wins.
FORMAT_CODECS: dict[str, dict[str, str]] = {
    "mp4": {"video": "libx264", "audio": "aac"},
    "avi": {"video": "libx264", "audio": "mp3"},
    "mkv": {"video": "libx264", "audio": "aac"},
    "webm": {"video": "libvpx-vp9", "audio": "libvorbis"},
    "mov": {"video": "libx264", "audio": "aac"},
}


def even_dimension(value: int) -> int:
    """Round ``value`` DOWN to the nearest even integer, clamped to ``>= 2``.

    libx264 with ``yuv420p`` (4:2:0 chroma subsampling) rejects odd width or
    height, so target canvas dimensions must be even. Rounding *down* (rather
    than up) is the canonical behaviour chosen here: it never enlarges a clip's
    canvas beyond the largest probed input, and the ``pad`` in
    :func:`normalize_video_stream` absorbs the at-most-one-pixel difference.
    The result is never smaller than 2, the minimum valid dimension.

    Args:
        value: The desired dimension in pixels (may be odd or, degenerately,
            less than 2).

    Returns:
        The largest even integer ``<= value`` that is at least ``2``.
    """
    even = value - (value % 2)
    return even if even >= 2 else 2


def normalize_video_stream(
    stream: ffmpeg.Stream,
    *,
    width: int,
    height: int,
    fps: float | int,
) -> ffmpeg.Stream:
    """Conform a video stream to a canonical geometry, fps, SAR and pix_fmt.

    ``xfade`` and the ``concat`` filter both require their inputs to share an
    identical resolution, pixel format, SAR and frame rate, so every clip must
    be conformed to the same target before the join graph is assembled. Aspect
    ratio is preserved by scaling to fit (``force_original_aspect_ratio=
    decrease``) and letterbox/pillarbox padding the remainder, centred.

    Args:
        stream: The input video stream node to normalize.
        width: Target canvas width in pixels (should already be even; see
            :func:`even_dimension`).
        height: Target canvas height in pixels (should already be even).
        fps: Target frame rate the stream is resampled to.

    Returns:
        The normalized video stream node.
    """
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


def normalize_audio_stream(stream: ffmpeg.Stream) -> ffmpeg.Stream:
    """Conform an audio stream to the canonical sample rate / channel layout.

    Aligning every clip (and any injected silence) to
    :data:`TARGET_SAMPLE_RATE` / :data:`TARGET_CHANNEL_LAYOUT` lets the
    ``concat`` and ``acrossfade`` filters mix heterogeneous sources without
    sample-rate or layout mismatches.

    Args:
        stream: The input audio stream node to normalize.

    Returns:
        The normalized audio stream node.
    """
    return ffmpeg.filter(
        stream,
        "aformat",
        sample_rates=TARGET_SAMPLE_RATE,
        channel_layouts=TARGET_CHANNEL_LAYOUT,
    )


def silent_audio_source(duration: float) -> ffmpeg.Stream:
    """Create a bounded silent audio source for a clip with no audio track.

    Backing silent clips with an ``anullsrc`` input of matched duration keeps
    the audio and video segment counts aligned in concat/xfade graphs and gives
    ``acrossfade`` a second audio input to blend.

    Args:
        duration: Length of the silence in seconds (typically the clip's
            effective, post-trim duration).

    Returns:
        An ffmpeg ``lavfi``/``anullsrc`` input stream at the canonical layout
        and sample rate, bounded to ``duration`` seconds.
    """
    return ffmpeg.input(
        f"anullsrc=channel_layout={TARGET_CHANNEL_LAYOUT}"
        f":sample_rate={TARGET_SAMPLE_RATE}",
        f="lavfi",
        t=duration,
    )
