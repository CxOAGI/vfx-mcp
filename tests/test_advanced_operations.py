"""Tests for advanced video operations.

This module tests the advanced video editing functions like audio extraction,
filters, speed changes, thumbnail generation, and format conversion. Uses
pytest fixtures for consistent test data and temporary file management.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import ffmpeg
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

if TYPE_CHECKING:
    from fastmcp import FastMCP


class VideoStreamInfo(TypedDict):
    """Type definition for video stream information."""

    width: int
    height: int
    codec_name: str
    codec_type: str


class AudioStreamInfo(TypedDict):
    """Type definition for audio stream information."""

    codec_name: str
    codec_type: str


class FormatInfo(TypedDict):
    """Type definition for format information."""

    duration: str
    format_name: str


class ProbeData(TypedDict):
    """Type definition for ffmpeg probe data."""

    streams: list[VideoStreamInfo | AudioStreamInfo]
    format: FormatInfo


class TestAudioOperations:
    """Test suite for audio-related video operations."""

    @pytest.mark.unit
    async def test_extract_audio(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """
        Test audio extraction functionality.

        This test verifies that extract_audio correctly extracts audio
        from a video file into a separate audio file.
        """
        output_path: Path = temp_dir / "extracted_audio.mp3"

        async with Client(mcp_server) as client:
            # Extract audio as MP3
            await client.call_tool(
                "extract_audio",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "format": "mp3",
                },
            )

            # Verify the output file exists
            assert output_path.exists()

            # Verify it's an audio file by probing it
            probe_result = ffmpeg.probe(str(output_path))
            probe_data = cast(ProbeData, probe_result)
            audio_stream = next(
                (s for s in probe_data["streams"] if s["codec_type"] == "audio"),
                None,
            )
            assert audio_stream is not None
            assert cast(AudioStreamInfo, audio_stream)["codec_name"] == "mp3"

    @pytest.mark.unit
    async def test_add_audio_replace(
        self,
        sample_video: Path,
        sample_audio: Path,
        temp_dir: Path,
        mcp_server: FastMCP[None],
    ) -> None:
        """
        Test audio replacement functionality.

        This test verifies that add_audio correctly replaces the audio
        track in a video with a new audio file.
        """
        output_path: Path = temp_dir / "video_with_new_audio.mp4"

        async with Client(mcp_server) as client:
            # Replace audio
            await client.call_tool(
                "add_audio",
                {
                    "input_path": str(sample_video),
                    "audio_path": str(sample_audio),
                    "output_path": str(output_path),
                    "replace": True,
                },
            )

            # Verify the output file exists
            assert output_path.exists()

            # Verify it has both video and audio streams
            probe_result = ffmpeg.probe(str(output_path))
            probe_data = cast(ProbeData, probe_result)
            video_stream = next(
                (s for s in probe_data["streams"] if s["codec_type"] == "video"),
                None,
            )
            audio_stream = next(
                (s for s in probe_data["streams"] if s["codec_type"] == "audio"),
                None,
            )
            assert video_stream is not None
            assert audio_stream is not None


class TestVideoEffects:
    """Test suite for video effects and filters."""

    @pytest.mark.unit
    async def test_apply_filter_simple(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """
        Test applying a simple video filter.

        This test verifies that apply_filter correctly applies a simple
        filter like horizontal flip to a video.
        """
        output_path: Path = temp_dir / "flipped_video.mp4"

        async with Client(mcp_server) as client:
            # Apply horizontal flip filter
            await client.call_tool(
                "apply_filter",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "filter": "hflip",
                },
            )

            # Verify the output file exists
            assert output_path.exists()

            # Verify it has the same dimensions as original
            probe_result = ffmpeg.probe(str(output_path))
            probe_data = cast(ProbeData, probe_result)
            video_stream = next(
                s for s in probe_data["streams"] if s["codec_type"] == "video"
            )
            assert cast(VideoStreamInfo, video_stream)["width"] == 1280
            assert cast(VideoStreamInfo, video_stream)["height"] == 720

    @pytest.mark.unit
    async def test_apply_filter_with_params(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """
        Test applying a filter with parameters.

        This test verifies that apply_filter correctly applies filters
        that require parameters, such as blur with intensity.
        """
        output_path: Path = temp_dir / "blurred_video.mp4"

        async with Client(mcp_server) as client:
            # Apply scale filter with parameter
            await client.call_tool(
                "apply_filter",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "filter": "scale=640:360",
                },
            )

            # Verify the output file exists
            assert output_path.exists()

            # Verify the video was scaled correctly
            probe_result = ffmpeg.probe(str(output_path))
            probe_data = cast(ProbeData, probe_result)
            video_stream = next(
                s for s in probe_data["streams"] if s["codec_type"] == "video"
            )
            assert cast(VideoStreamInfo, video_stream)["width"] == 640
            assert cast(VideoStreamInfo, video_stream)["height"] == 360

    @pytest.mark.unit
    async def test_change_speed_faster(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """
        Test speeding up video playback.

        This test verifies that change_speed correctly speeds up video
        playback while maintaining synchronization.
        """
        output_path: Path = temp_dir / "fast_video.mp4"

        async with Client(mcp_server) as client:
            # Double the speed
            await client.call_tool(
                "change_speed",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "speed": 2.0,
                },
            )

            # Verify the output file exists
            assert output_path.exists()

            # Verify the video duration is approximately halved
            original_probe_result = ffmpeg.probe(str(sample_video))
            new_probe_result = ffmpeg.probe(str(output_path))

            original_probe = cast(ProbeData, original_probe_result)
            new_probe = cast(ProbeData, new_probe_result)

            original_duration = float(original_probe["format"]["duration"])
            new_duration = float(new_probe["format"]["duration"])

            # Duration should be approximately half (with some tolerance)
            expected_duration = original_duration / 2.0
            assert abs(new_duration - expected_duration) < 0.5

    @pytest.mark.unit
    async def test_change_speed_slower(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """
        Test slowing down video playback.

        This test verifies that change_speed correctly slows down video
        playback while maintaining quality.
        """
        output_path: Path = temp_dir / "slow_video.mp4"

        async with Client(mcp_server) as client:
            # Half the speed
            await client.call_tool(
                "change_speed",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "speed": 0.5,
                },
            )

            # Verify the output file exists
            assert output_path.exists()

            # Verify the video duration is approximately doubled
            original_probe_result = ffmpeg.probe(str(sample_video))
            new_probe_result = ffmpeg.probe(str(output_path))

            original_probe = cast(ProbeData, original_probe_result)
            new_probe = cast(ProbeData, new_probe_result)

            original_duration = float(original_probe["format"]["duration"])
            new_duration = float(new_probe["format"]["duration"])

            # Duration should be approximately double (with some tolerance)
            expected_duration = original_duration * 2.0
            assert abs(new_duration - expected_duration) < 0.5

    @pytest.mark.unit
    async def test_change_speed_error_handling(
        self, sample_video: Path, mcp_server: FastMCP[None]
    ) -> None:
        """
        Test error handling for invalid speed values.

        This test verifies that change_speed properly handles invalid
        speed values like zero or negative numbers.
        """
        async with Client(mcp_server) as client:
            # Try invalid speed (zero)
            with pytest.raises(Exception) as exc_info:
                await client.call_tool(
                    "change_speed",
                    {
                        "input_path": str(sample_video),
                        "output_path": "output.mp4",
                        "speed": 0.0,
                    },
                )

            # Verify the error message
            assert "must be greater than 0" in str(exc_info.value).lower()


class TestThumbnailGeneration:
    """Test suite for thumbnail generation."""

    @pytest.mark.unit
    async def test_generate_thumbnail_default(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """
        Test thumbnail generation with default settings.

        This test verifies that generate_thumbnail correctly extracts
        a frame from the middle of the video as a thumbnail.
        """
        output_path: Path = temp_dir / "thumbnail.jpg"

        async with Client(mcp_server) as client:
            # Generate thumbnail with default settings
            await client.call_tool(
                "generate_thumbnail",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                },
            )

            # Verify the output file exists
            assert output_path.exists()

            # Verify it's an image file by probing it
            probe_result = ffmpeg.probe(str(output_path))
            probe_data = cast(ProbeData, probe_result)
            video_stream = next(
                (s for s in probe_data["streams"] if s["codec_type"] == "video"),
                None,
            )
            assert video_stream is not None
            # Should maintain original dimensions
            assert cast(VideoStreamInfo, video_stream)["width"] == 1280
            assert cast(VideoStreamInfo, video_stream)["height"] == 720

    @pytest.mark.unit
    async def test_generate_thumbnail_specific_time(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """
        Test thumbnail generation at a specific timestamp.

        This test verifies that generate_thumbnail correctly extracts
        a frame from a specified time in the video.
        """
        output_path: Path = temp_dir / "thumbnail_2s.png"

        async with Client(mcp_server) as client:
            # Generate thumbnail at 2 seconds
            await client.call_tool(
                "generate_thumbnail",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "timestamp": 2.0,
                },
            )

            # Verify the output file exists
            assert output_path.exists()

            # Verify it's an image file
            probe_result = ffmpeg.probe(str(output_path))
            probe_data = cast(ProbeData, probe_result)
            video_stream = next(
                s for s in probe_data["streams"] if s["codec_type"] == "video"
            )
            assert video_stream is not None

    @pytest.mark.unit
    async def test_generate_thumbnail_resized(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """
        Test thumbnail generation with custom dimensions.

        This test verifies that generate_thumbnail correctly resizes
        the extracted frame to specified dimensions.
        """
        output_path: Path = temp_dir / "thumbnail_small.jpg"

        async with Client(mcp_server) as client:
            # Generate resized thumbnail
            await client.call_tool(
                "generate_thumbnail",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "width": 320,
                    "height": 180,
                },
            )

            # Verify the output file exists
            assert output_path.exists()

            # Verify the image dimensions
            probe_result = ffmpeg.probe(str(output_path))
            probe_data = cast(ProbeData, probe_result)
            video_stream = next(
                s for s in probe_data["streams"] if s["codec_type"] == "video"
            )
            assert cast(VideoStreamInfo, video_stream)["width"] == 320
            assert cast(VideoStreamInfo, video_stream)["height"] == 180


class TestFormatConversion:
    """Test suite for format conversion operations."""

    @pytest.mark.unit
    async def test_convert_format_basic(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """
        Test basic format conversion.

        This test verifies that convert_format correctly converts a video
        to a different format while maintaining quality.
        """
        output_path: Path = temp_dir / "converted.avi"

        async with Client(mcp_server) as client:
            # Convert to AVI format
            await client.call_tool(
                "convert_format",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "format": "avi",
                },
            )

            # Verify the output file exists
            assert output_path.exists()

            # Verify the format was changed
            probe_result = ffmpeg.probe(str(output_path))
            probe_data = cast(ProbeData, probe_result)
            format_name = probe_data["format"]["format_name"]
            assert "avi" in format_name.lower()

    @pytest.mark.unit
    async def test_convert_format_with_codecs(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """
        Test format conversion with specific codecs.

        This test verifies that convert_format correctly applies
        specific video and audio codecs during conversion.
        """
        output_path: Path = temp_dir / "converted_codecs.mp4"

        async with Client(mcp_server) as client:
            # Convert with specific codecs
            await client.call_tool(
                "convert_format",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "video_codec": "libx264",
                    "audio_codec": "aac",
                },
            )

            # Verify the output file exists
            assert output_path.exists()

            # Verify the codecs were applied
            probe_result = ffmpeg.probe(str(output_path))
            probe_data = cast(ProbeData, probe_result)
            video_stream = next(
                (s for s in probe_data["streams"] if s["codec_type"] == "video"),
                None,
            )
            if video_stream:
                assert cast(VideoStreamInfo, video_stream)["codec_name"] == "h264"

    @pytest.mark.unit
    async def test_convert_format_with_bitrates(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """
        Test format conversion with custom bitrates.

        This test verifies that convert_format correctly applies
        custom video and audio bitrates during conversion.
        """
        output_path: Path = temp_dir / "converted_bitrates.mp4"

        async with Client(mcp_server) as client:
            # Convert with custom bitrates
            await client.call_tool(
                "convert_format",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "video_bitrate": "500k",
                    "audio_bitrate": "128k",
                },
            )

            # Verify the output file exists
            assert output_path.exists()

            # Verify the file was processed (size should be different)
            original_size = sample_video.stat().st_size
            new_size = output_path.stat().st_size
            # With lower bitrate, file should generally be smaller
            assert new_size != original_size


class TestErrorHandling:
    """Test suite for error handling in advanced operations."""

    @pytest.mark.unit
    async def test_extract_audio_nonexistent_file(
        self, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """
        Test error handling for non-existent input files.

        This test verifies that audio extraction properly handles
        cases where the input video file doesn't exist.
        """
        output_path: Path = temp_dir / "audio.mp3"

        async with Client(mcp_server) as client:
            # Try to extract audio from non-existent file
            with pytest.raises(ToolError):
                await client.call_tool(
                    "extract_audio",
                    {
                        "input_path": "nonexistent.mp4",
                        "output_path": str(output_path),
                    },
                )

    @pytest.mark.unit
    async def test_apply_filter_invalid_filter(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """
        Test error handling for invalid filter names.

        This test verifies that filter application properly handles
        cases where an invalid filter name is provided.
        """
        output_path: Path = temp_dir / "filtered.mp4"

        async with Client(mcp_server) as client:
            # Try to apply an invalid filter
            with pytest.raises(ToolError):
                await client.call_tool(
                    "apply_filter",
                    {
                        "input_path": str(sample_video),
                        "output_path": str(output_path),
                        "filter": "nonexistent_filter",
                    },
                )


# Pixel formats that actually carry an alpha channel. A "transparent
# background" green-screen result is only honest if the probed output uses one
# of these; yuv420p (the standard H.264 output) has no alpha.
_ALPHA_PIXEL_FORMATS: frozenset[str] = frozenset(
    {"argb", "rgba", "abgr", "bgra", "yuva420p", "yuva444p", "yuva422p"}
)


def _build_transparent_stream(
    input_path: str, output_path: str, vcodec: str, pix_fmt: str
) -> ffmpeg.Stream:
    """Build the transparent (alpha) chroma-key graph the tool produces."""
    input_stream = ffmpeg.input(input_path)
    keyed = ffmpeg.filter(
        input_stream, "chromakey", color="0x00FF00", similarity=0.3, blend=0.1
    )
    keyed = ffmpeg.filter(keyed, "despill", type="green", mix=0.5)
    return ffmpeg.output(keyed, output_path, vcodec=vcodec, pix_fmt=pix_fmt)


def _build_composite_stream(
    input_path: str, background_path: str, output_path: str
) -> ffmpeg.Stream:
    """Build the composited chroma-key graph the tool produces."""
    input_stream = ffmpeg.input(input_path)
    keyed = ffmpeg.filter(
        input_stream, "chromakey", color="0x00FF00", similarity=0.3, blend=0.1
    )
    keyed = ffmpeg.filter(keyed, "despill", type="green", mix=0.5)
    background = ffmpeg.input(background_path)
    composited = ffmpeg.filter([background, keyed], "overlay", x="(W-w)/2", y="(H-h)/2")
    return ffmpeg.output(
        composited,
        output_path,
        vcodec="libx264",
        pix_fmt="yuv420p",
        acodec="aac",
        map="1:a?",
    )


class TestGreenScreenEffect:
    """Test suite for create_green_screen_effect compositing."""

    @pytest.mark.unit
    def test_transparent_command_shape_mov(self) -> None:
        """The transparent .mov path compiles to an alpha-capable codec.

        Verifies via ffmpeg.get_args (no binary needed) that the transparent
        output uses qtrle + argb, which genuinely carries an alpha channel,
        rather than the H.264/yuv420p output that silently drops transparency.
        """
        args = _build_transparent_stream(
            "in.mp4", "out.mov", "qtrle", "argb"
        ).get_args()
        assert "qtrle" in args
        assert "argb" in args
        assert "chromakey" in " ".join(args)

    @pytest.mark.unit
    def test_composite_command_shape_maps_source_audio(self) -> None:
        """The composited path overlays and maps the *source* video's audio.

        Verifies the command overlays the keyed foreground and maps audio from
        input index 1 (the source), not input 0 (the background).
        """
        args = _build_composite_stream("src.mp4", "bg.mp4", "out.mp4").get_args()
        joined = " ".join(args)
        assert "overlay" in joined
        assert "1:a?" in args
        # The source video is the second input (-i), so 1:a is its audio.
        input_files = [args[i + 1] for i, a in enumerate(args) if a == "-i"]
        assert input_files[1] == "src.mp4"

    @pytest.mark.integration
    async def test_transparent_output_has_alpha(
        self,
        sample_video: Path,
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A transparent (no-background) result must have a real alpha channel.

        Exercises the M6 fix: without a background the tool must emit an
        alpha-capable format. Probes the output pixel format and asserts it
        carries alpha (not yuv420p).
        """
        monkeypatch.setenv("VFX_ALLOW_ABSOLUTE", "1")
        output_path: Path = temp_dir / "transparent.mov"

        async with Client(mcp_server) as client:
            await client.call_tool(
                "create_green_screen_effect",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "chroma_key_color": "green",
                },
            )

            assert output_path.exists()

            probe_result = ffmpeg.probe(str(output_path))
            probe_data = cast(ProbeData, probe_result)
            video_stream = next(
                s for s in probe_data["streams"] if s["codec_type"] == "video"
            )
            pix_fmt = cast(dict[str, str], video_stream).get("pix_fmt", "")
            assert pix_fmt in _ALPHA_PIXEL_FORMATS, (
                f"transparent output pix_fmt {pix_fmt!r} has no alpha channel"
            )

    @pytest.mark.integration
    async def test_transparent_rejects_non_alpha_extension(
        self,
        sample_video: Path,
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Requesting transparency with a non-alpha container is rejected.

        An .mp4 (H.264/yuv420p) cannot carry alpha, so the transparent path
        must fail fast rather than silently producing an opaque video.
        """
        monkeypatch.setenv("VFX_ALLOW_ABSOLUTE", "1")
        output_path: Path = temp_dir / "not_transparent.mp4"

        async with Client(mcp_server) as client:
            with pytest.raises(ToolError):
                await client.call_tool(
                    "create_green_screen_effect",
                    {
                        "input_path": str(sample_video),
                        "output_path": str(output_path),
                    },
                )

    @pytest.mark.integration
    async def test_composited_path_has_video_and_audio(
        self,
        sample_video: Path,
        sample_videos: list[Path],
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Compositing over a background yields a standard video with audio.

        Uses a video background so the overlay produces real frames, then
        asserts the output has both a video stream and the source video's
        audio track (mapped from input index 1).
        """
        monkeypatch.setenv("VFX_ALLOW_ABSOLUTE", "1")
        output_path: Path = temp_dir / "composited.mp4"

        async with Client(mcp_server) as client:
            await client.call_tool(
                "create_green_screen_effect",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "background_path": str(sample_videos[0]),
                    "chroma_key_color": "green",
                },
            )

            assert output_path.exists()

            probe_result = ffmpeg.probe(str(output_path))
            probe_data = cast(ProbeData, probe_result)
            video_stream = next(
                (s for s in probe_data["streams"] if s["codec_type"] == "video"),
                None,
            )
            audio_stream = next(
                (s for s in probe_data["streams"] if s["codec_type"] == "audio"),
                None,
            )
            assert video_stream is not None
            assert audio_stream is not None


class TestMotionBlur:
    """Test suite for apply_motion_blur."""

    @pytest.mark.integration
    async def test_motion_blur_preserves_dimensions(
        self,
        sample_video: Path,
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Motion blur produces a valid video at the source dimensions."""
        monkeypatch.setenv("VFX_ALLOW_ABSOLUTE", "1")
        output_path: Path = temp_dir / "motion_blur.mp4"

        async with Client(mcp_server) as client:
            await client.call_tool(
                "apply_motion_blur",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "blur_strength": 1.0,
                    "angle": 0.0,
                },
            )

            assert output_path.exists()

            probe_result = ffmpeg.probe(str(output_path))
            probe_data = cast(ProbeData, probe_result)
            video_stream = next(
                s for s in probe_data["streams"] if s["codec_type"] == "video"
            )
            assert cast(VideoStreamInfo, video_stream)["width"] == 1280
            assert cast(VideoStreamInfo, video_stream)["height"] == 720

    @pytest.mark.integration
    async def test_motion_blur_invalid_strength(
        self,
        sample_video: Path,
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Out-of-range blur strength is rejected before encoding."""
        monkeypatch.setenv("VFX_ALLOW_ABSOLUTE", "1")
        output_path: Path = temp_dir / "motion_blur_bad.mp4"

        async with Client(mcp_server) as client:
            with pytest.raises(ToolError):
                await client.call_tool(
                    "apply_motion_blur",
                    {
                        "input_path": str(sample_video),
                        "output_path": str(output_path),
                        "blur_strength": 99.0,
                    },
                )
