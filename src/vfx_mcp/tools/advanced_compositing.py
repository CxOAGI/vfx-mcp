"""Advanced compositing tools: green screen, motion blur, and complex effects."""

import ffmpeg
from fastmcp import Context, FastMCP

from ..core import (
    create_standard_output,
    log_operation,
    parse_color,
    run_ffmpeg_async,
    safe_input_path,
    safe_output_path,
    validate_range,
)

__all__ = ["register_compositing_tools"]

# Alpha-capable output formats for the transparent-background chroma-key path.
# yuv420p H.264 (the standard output) has no alpha channel, so a transparent
# result requires a container/codec pairing that actually carries alpha. Each
# entry maps an output extension to the (vcodec, pix_fmt) needed to preserve the
# chroma-keyed alpha matte losslessly (QuickTime RLE) or with alpha (VP9).
_ALPHA_OUTPUT_FORMATS: dict[str, tuple[str, str]] = {
    ".mov": ("qtrle", "argb"),
    ".webm": ("libvpx-vp9", "yuva420p"),
    ".mkv": ("libvpx-vp9", "yuva420p"),
}


def register_compositing_tools(
    mcp: FastMCP[None],
) -> None:
    """Register advanced compositing tools with the MCP server."""

    @mcp.tool
    async def create_green_screen_effect(
        input_path: str,
        output_path: str,
        background_path: str | None = None,
        chroma_key_color: str = "green",
        similarity: float = 0.3,
        blend: float = 0.1,
        spill_reduction: float = 0.5,
        crf: int | None = None,
        preset: str | None = None,
        ctx: Context | None = None,
    ) -> str:
        """Remove green/blue screen and composite with a background or alpha.

        Advanced chroma key compositing with adjustable parameters for
        professional green screen removal and background replacement.

        When ``background_path`` is provided, the keyed foreground is overlaid
        (centered) on the background and encoded as a standard H.264/AAC MP4,
        preserving the *source* video's audio track. When ``background_path``
        is ``None`` the tool produces a genuinely transparent result: the
        chroma-keyed alpha matte is written to an alpha-capable container/codec
        chosen from the output extension. Because H.264/yuv420p cannot carry
        an alpha channel, the transparent path requires an alpha-capable output
        extension:

        - ``.mov`` -> QuickTime RLE (``qtrle``, ARGB, lossless)
        - ``.webm`` / ``.mkv`` -> VP9 (``libvpx-vp9``, ``yuva420p``)

        The transparent output is video-only (the alpha matte); audio is not
        carried through since alpha-matte deliverables are composited later.

        Args:
            input_path: Path to the input video with green/blue screen.
            output_path: Path where the composited video will be saved. For the
                transparent path (no background) this must end in ``.mov``,
                ``.webm`` or ``.mkv``.
            background_path: Path to background image/video. If None, produces a
                transparent (alpha) result instead of compositing.
            chroma_key_color: Color to remove ("green", "blue", "red", or hex
                code like "#00FF00").
            similarity: Color similarity threshold (0.0 to 1.0). Lower = more
                precise.
            blend: Edge blending amount (0.0 to 1.0) for smoother edges.
            spill_reduction: Color spill reduction strength (0.0 to 1.0).
            crf: Constant Rate Factor for the composited (H.264) output; lower
                is higher quality. Ignored on the transparent path. Defaults to
                the encoder default when omitted.
            preset: libx264 speed/quality preset for the composited output
                (e.g. "ultrafast", "medium"). Ignored on the transparent path.
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating green screen effect was applied.

        Raises:
            ValueError: If parameter values are out of valid ranges, or the
                transparent path is requested with a non-alpha output extension.
            RuntimeError: If ffmpeg encounters an error during processing.
        """
        validate_range(similarity, 0.0, 1.0, "Similarity")
        validate_range(blend, 0.0, 1.0, "Blend")
        validate_range(
            spill_reduction,
            0.0,
            1.0,
            "Spill reduction",
        )

        resolved_input = safe_input_path(input_path)
        resolved_output = safe_output_path(output_path)
        resolved_background = (
            safe_input_path(background_path) if background_path else None
        )

        # For the transparent path, fail fast (before any encoding) if the
        # requested container/codec cannot carry an alpha channel.
        alpha_format: tuple[str, str] | None = None
        if resolved_background is None:
            suffix = resolved_output.suffix.lower()
            alpha_format = _ALPHA_OUTPUT_FORMATS.get(suffix)
            if alpha_format is None:
                supported = ", ".join(sorted(_ALPHA_OUTPUT_FORMATS))
                raise ValueError(
                    "Transparent background output requires an alpha-capable "
                    f"container/codec. Use one of: {supported}. Got: {output_path}"
                )

        # Parse color to ffmpeg format
        key_color = parse_color(chroma_key_color)

        await log_operation(
            ctx,
            f"Applying chroma key: {chroma_key_color} "
            f"(similarity: {similarity}, blend: {blend})",
        )

        input_stream: ffmpeg.Stream = ffmpeg.input(str(resolved_input))

        # Create chromakey filter (adds an alpha channel to the video)
        keyed: ffmpeg.Stream = ffmpeg.filter(
            input_stream,
            "chromakey",
            color=key_color,
            similarity=similarity,
            blend=blend,
        )

        # Apply spill reduction if needed
        if spill_reduction > 0:
            keyed = ffmpeg.filter(
                keyed,
                "despill",
                type=("green" if "green" in chroma_key_color.lower() else "blue"),
                mix=spill_reduction,
            )

        output: ffmpeg.Stream
        if resolved_background is not None:
            # Composite the keyed foreground over the background, centered.
            background: ffmpeg.Stream = ffmpeg.input(str(resolved_background))
            composited: ffmpeg.Stream = ffmpeg.filter(
                [background, keyed],
                "overlay",
                x="(W-w)/2",
                y="(H-h)/2",
            )
            # Map the *source* video's audio (input index 1: background is
            # input 0, the keyed source is input 1); "?" tolerates silent
            # inputs.
            output = create_standard_output(
                composited,
                str(resolved_output),
                crf=crf,
                preset=preset,
                map="1:a?",
            )
        else:
            # Transparent path: write the alpha matte to an alpha-capable
            # container/codec. Video-only by design.
            assert alpha_format is not None
            vcodec, pix_fmt = alpha_format
            output = ffmpeg.output(
                keyed,
                str(resolved_output),
                vcodec=vcodec,
                pix_fmt=pix_fmt,
            )

        await run_ffmpeg_async(output, ctx=ctx, output_path=str(resolved_output))

        bg_msg = (
            " with custom background"
            if resolved_background is not None
            else " with transparent background"
        )
        return f"Green screen effect applied{bg_msg} and saved to {resolved_output}"

    @mcp.tool
    async def apply_motion_blur(
        input_path: str,
        output_path: str,
        blur_strength: float = 1.0,
        angle: float = 0.0,
        crf: int | None = None,
        preset: str | None = None,
        ctx: Context | None = None,
    ) -> str:
        """Apply motion blur effect to simulate camera or object movement.

        Creates motion blur effect with adjustable strength and direction,
        simulating movement in the specified angle direction.

        Args:
            input_path: Path to the input video file.
            output_path: Path where the motion-blurred video will be saved.
            blur_strength: Blur intensity (0.1 to 3.0). Higher = more blur.
            angle: Blur direction angle in degrees (0 to 360).
            crf: Constant Rate Factor for the H.264 output; lower is higher
                quality. Defaults to the encoder default when omitted.
            preset: libx264 speed/quality preset (e.g. "ultrafast", "medium").
            ctx: MCP context for progress reporting and logging.

        Returns:
            Success message indicating motion blur was applied.

        Raises:
            ValueError: If parameters are out of valid ranges.
            RuntimeError: If ffmpeg encounters an error during processing.
        """
        validate_range(
            blur_strength,
            0.1,
            3.0,
            "Blur strength",
        )
        validate_range(angle, 0, 360, "Angle")

        resolved_input = safe_input_path(input_path)
        resolved_output = safe_output_path(output_path)

        await log_operation(
            ctx,
            f"Applying motion blur (strength: {blur_strength}, angle: {angle}°)...",
        )

        stream: ffmpeg.Stream = ffmpeg.input(str(resolved_input))

        # Convert strength to a valid boxblur radius. boxblur's luma_radius
        # is a single (isotropic) expression and luma_power is separate —
        # the angle is approximated via blur power since boxblur cannot
        # apply a truly directional kernel on its own.
        blur_amount = int(blur_strength * 3) * 2 + 1  # Odd number for kernel size
        stream = ffmpeg.filter(
            stream,
            "boxblur",
            luma_radius=blur_amount,
            luma_power=1,
        )

        output: ffmpeg.Stream = create_standard_output(
            stream,
            str(resolved_output),
            crf=crf,
            preset=preset,
            map="0:a?",
        )
        await run_ffmpeg_async(output, ctx=ctx, output_path=str(resolved_output))

        return (
            f"Motion blur applied (strength: {blur_strength}, angle: {angle}°) "
            f"and saved to {resolved_output}"
        )
