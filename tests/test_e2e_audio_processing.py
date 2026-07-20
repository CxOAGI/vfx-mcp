"""End-to-end tests for audio processing operations.

This module provides comprehensive end-to-end testing for audio processing
tools including extract_audio, add_audio, volume/fade effects, mixing, and
loudness normalization. Tests cover realistic workflows and complete
operations from input to output validation, including that "replace" really
replaces (single audio track) and that video-preserving tools keep the video
stream.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import ffmpeg
import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

if TYPE_CHECKING:
    pass


@pytest.fixture(autouse=True)
def _allow_absolute_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Permit absolute (temp-dir) paths through the workspace sandbox.

    The audio tools validate every path with ``safe_input_path`` /
    ``safe_output_path``, which reject absolute paths outside the workspace
    unless ``VFX_ALLOW_ABSOLUTE=1``. The test fixtures live in per-test temp
    directories, so this fixture opts the whole module into absolute paths.
    """
    monkeypatch.setenv("VFX_ALLOW_ABSOLUTE", "1")


def _stream_counts(media_path: Path) -> tuple[int, int]:
    """Return the ``(video_count, audio_count)`` for a media file."""
    probe = ffmpeg.probe(str(media_path))
    streams = probe["streams"]
    video = sum(1 for s in streams if s["codec_type"] == "video")
    audio = sum(1 for s in streams if s["codec_type"] == "audio")
    return video, audio


def _mean_volume_db(media_path: Path, start: float, duration: float) -> float:
    """Return the mean volume (dBFS) of a time window using ``volumedetect``.

    Runs the ``[start, start + duration)`` slice of ``media_path`` through
    ffmpeg's ``volumedetect`` filter to the null muxer and parses the
    ``mean_volume`` line printed to stderr. Silence approaches large negative
    values (e.g. ``-90 dB``); full-scale audio sits near ``0 dB``.
    """
    _, stderr = (
        ffmpeg.input(str(media_path), ss=start, t=duration)
        .filter("volumedetect")
        .output("-", format="null")
        .run(capture_stdout=True, capture_stderr=True)
    )
    for line in stderr.decode().splitlines():
        if "mean_volume:" in line:
            # e.g. "[Parsed_volumedetect_0 @ ...] mean_volume: -3.5 dB"
            value = line.split("mean_volume:")[1].strip().split(" ")[0]
            return float(value)
    raise AssertionError(f"volumedetect produced no mean_volume for {media_path}")


class TestAudioProcessingE2E:
    """End-to-end tests for audio processing operations."""

    @pytest.mark.integration
    async def test_complete_audio_workflow(
        self,
        sample_video: Path,
        sample_audio: Path,
        temp_dir: Path,
        mcp_server: FastMCP[None],
    ) -> None:
        """Test a complete audio processing workflow.

        This test simulates a realistic audio workflow:
        1. Extract audio from a video
        2. Add new audio to the video (replace mode)
        3. Add audio in mix mode
        4. Validate audio properties
        """
        async with Client(mcp_server) as client:
            # Step 1: Extract original audio
            extracted_audio_path = temp_dir / "extracted.mp3"
            _ = await client.call_tool(
                "extract_audio",
                {
                    "input_path": str(sample_video),
                    "output_path": str(extracted_audio_path),
                    "format": "mp3",
                    "bitrate": "192k",
                },
            )
            assert extracted_audio_path.exists()

            # Verify extracted audio properties
            audio_probe = ffmpeg.probe(str(extracted_audio_path))
            audio_stream = next(
                s for s in audio_probe["streams"] if s["codec_type"] == "audio"
            )
            assert audio_stream["codec_name"] == "mp3"

            # Step 2: Replace audio in video
            video_with_new_audio = temp_dir / "video_new_audio.mp4"
            _ = await client.call_tool(
                "add_audio",
                {
                    "input_path": str(sample_video),
                    "audio_path": str(sample_audio),
                    "output_path": str(video_with_new_audio),
                    "replace": True,
                    "audio_volume": 1.0,
                },
            )
            assert video_with_new_audio.exists()

            # Step 3: Mix audio with existing
            video_mixed_audio = temp_dir / "video_mixed_audio.mp4"
            _ = await client.call_tool(
                "add_audio",
                {
                    "input_path": str(sample_video),
                    "audio_path": str(sample_audio),
                    "output_path": str(video_mixed_audio),
                    "replace": False,
                    "audio_volume": 0.5,
                },
            )
            assert video_mixed_audio.exists()

            # Step 4: Verify both outputs have exactly one video and one audio
            # stream. In particular "replace" must not leave the original audio
            # behind (a second audio track).
            for video_path in [video_with_new_audio, video_mixed_audio]:
                video_count, audio_count = _stream_counts(video_path)
                assert video_count == 1
                assert audio_count == 1

    @pytest.mark.integration
    async def test_add_audio_replace_produces_single_track(
        self,
        sample_video: Path,
        sample_audio: Path,
        temp_dir: Path,
        mcp_server: FastMCP[None],
    ) -> None:
        """Replace mode must drop the original audio (single audio stream).

        The input video already carries a 440Hz sine track. After replacing
        the audio, the output must contain exactly one audio stream (the new
        one) and one video stream, not two audio tracks.
        """
        async with Client(mcp_server) as client:
            output_path = temp_dir / "replaced_single_track.mp4"
            _ = await client.call_tool(
                "add_audio",
                {
                    "input_path": str(sample_video),
                    "audio_path": str(sample_audio),
                    "output_path": str(output_path),
                    "replace": True,
                    "audio_volume": 1.0,
                },
            )
            assert output_path.exists()

            video_count, audio_count = _stream_counts(output_path)
            assert video_count == 1
            assert audio_count == 1

    @pytest.mark.integration
    async def test_add_audio_to_silent_video(
        self,
        sample_videos_silent: list[Path],
        sample_audio: Path,
        temp_dir: Path,
        mcp_server: FastMCP[None],
    ) -> None:
        """Adding audio to a silent video should produce a single audio track.

        Silent clips (no original audio) can be given a soundtrack in either
        replace or mix mode; mix gracefully falls back to using the new audio
        alone since there is nothing to mix with.
        """
        silent_video = sample_videos_silent[0]
        async with Client(mcp_server) as client:
            for replace, name in (
                (True, "silent_replace.mp4"),
                (False, "silent_mix.mp4"),
            ):
                output_path = temp_dir / name
                _ = await client.call_tool(
                    "add_audio",
                    {
                        "input_path": str(silent_video),
                        "audio_path": str(sample_audio),
                        "output_path": str(output_path),
                        "replace": replace,
                        "audio_volume": 1.0,
                    },
                )
                assert output_path.exists()
                video_count, audio_count = _stream_counts(output_path)
                assert video_count == 1
                assert audio_count == 1

    @pytest.mark.integration
    async def test_audio_format_conversion_workflow(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """Test extracting audio in different formats.

        This test extracts audio from a video in multiple formats
        and verifies each format has correct properties.
        """
        formats_to_test = [
            ("mp3", "192k"),
            ("wav", ""),  # WAV doesn't use bitrate
            ("aac", "128k"),
            ("ogg", "256k"),
            ("flac", ""),  # FLAC is lossless, no bitrate
        ]

        async with Client(mcp_server) as client:
            for format_name, bitrate in formats_to_test:
                output_path = temp_dir / f"audio.{format_name}"

                extract_args = {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "format": format_name,
                }
                if bitrate:
                    extract_args["bitrate"] = bitrate

                _ = await client.call_tool(
                    "extract_audio",
                    extract_args,
                )
                assert output_path.exists()

                # Verify audio format
                probe = ffmpeg.probe(str(output_path))
                audio_stream = next(
                    s for s in probe["streams"] if s["codec_type"] == "audio"
                )

                # Check codec (some formats have different probe names)
                if format_name == "mp3":
                    assert audio_stream["codec_name"] == "mp3"
                elif format_name == "aac":
                    assert audio_stream["codec_name"] == "aac"
                elif format_name == "ogg":
                    assert audio_stream["codec_name"] == "vorbis"
                elif format_name == "wav":
                    assert audio_stream["codec_name"] == "pcm_s16le"
                elif format_name == "flac":
                    assert audio_stream["codec_name"] == "flac"

    @pytest.mark.integration
    async def test_extract_audio_no_audio_stream(
        self,
        sample_videos_silent: list[Path],
        temp_dir: Path,
        mcp_server: FastMCP[None],
    ) -> None:
        """Extracting audio from a silent video must raise a clear error.

        Silent Veo-2-style clips have no audio stream; extraction should fail
        with a meaningful error rather than an opaque ffmpeg mapping crash.
        """
        async with Client(mcp_server) as client:
            with pytest.raises(ToolError):
                await client.call_tool(
                    "extract_audio",
                    {
                        "input_path": str(sample_videos_silent[0]),
                        "output_path": str(temp_dir / "silent_audio.mp3"),
                        "format": "mp3",
                    },
                )

    @pytest.mark.integration
    async def test_audio_volume_adjustment_workflow(
        self,
        sample_video: Path,
        sample_audio: Path,
        temp_dir: Path,
        mcp_server: FastMCP[None],
    ) -> None:
        """Test audio volume adjustments in various scenarios.

        This test verifies that audio volume adjustments work correctly
        in both replace and mix modes with different volume levels.
        """
        volume_tests = [
            ("replace", 0.5, "quiet_replace.mp4"),
            ("replace", 1.5, "loud_replace.mp4"),
            ("mix", 0.3, "quiet_mix.mp4"),
            ("mix", 0.8, "normal_mix.mp4"),
        ]

        async with Client(mcp_server) as client:
            for mode, volume, output_name in volume_tests:
                output_path = temp_dir / output_name
                replace_mode = mode == "replace"

                _ = await client.call_tool(
                    "add_audio",
                    {
                        "input_path": str(sample_video),
                        "audio_path": str(sample_audio),
                        "output_path": str(output_path),
                        "replace": replace_mode,
                        "audio_volume": volume,
                    },
                )
                assert output_path.exists()

                # Verify a single video and single audio stream exist.
                video_count, audio_count = _stream_counts(output_path)
                assert video_count == 1
                assert audio_count == 1

                probe = ffmpeg.probe(str(output_path))
                video_stream = next(
                    s for s in probe["streams"] if s["codec_type"] == "video"
                )

                # Verify video properties remain unchanged
                assert video_stream["width"] == 1280
                assert video_stream["height"] == 720

    @pytest.mark.integration
    async def test_adjust_audio_volume_preserves_video(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """adjust_audio_volume on a video must keep the video stream."""
        async with Client(mcp_server) as client:
            output_path = temp_dir / "louder_video.mp4"
            _ = await client.call_tool(
                "adjust_audio_volume",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "volume": 1.5,
                },
            )
            assert output_path.exists()
            video_count, audio_count = _stream_counts(output_path)
            assert video_count == 1
            assert audio_count == 1

    @pytest.mark.integration
    async def test_adjust_audio_volume_audio_only(
        self, sample_audio: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """adjust_audio_volume on an audio-only file stays audio-only."""
        async with Client(mcp_server) as client:
            output_path = temp_dir / "louder_audio.mp3"
            _ = await client.call_tool(
                "adjust_audio_volume",
                {
                    "input_path": str(sample_audio),
                    "output_path": str(output_path),
                    "volume": 1.5,
                },
            )
            assert output_path.exists()
            video_count, audio_count = _stream_counts(output_path)
            assert video_count == 0
            assert audio_count == 1

    @pytest.mark.integration
    async def test_audio_fade_preserves_video(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """Fade-in/out on a video must preserve the video stream.

        Regression test for the bug where fade tools mapped only the filtered
        audio, producing an audio-only ".mp4" with no video stream.
        """
        async with Client(mcp_server) as client:
            for tool_name, name in (
                ("audio_fade_in", "faded_in.mp4"),
                ("audio_fade_out", "faded_out.mp4"),
            ):
                output_path = temp_dir / name
                _ = await client.call_tool(
                    tool_name,
                    {
                        "input_path": str(sample_video),
                        "output_path": str(output_path),
                        "duration": 1.0,
                    },
                )
                assert output_path.exists()
                video_count, audio_count = _stream_counts(output_path)
                assert video_count == 1, f"{tool_name} dropped the video stream"
                assert audio_count == 1

    @pytest.mark.integration
    async def test_audio_fade_audio_only(
        self, sample_audio: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """Fade tools on an audio-only file produce an audio-only output."""
        async with Client(mcp_server) as client:
            for tool_name, name in (
                ("audio_fade_in", "audio_faded_in.mp3"),
                ("audio_fade_out", "audio_faded_out.mp3"),
            ):
                output_path = temp_dir / name
                _ = await client.call_tool(
                    tool_name,
                    {
                        "input_path": str(sample_audio),
                        "output_path": str(output_path),
                        "duration": 1.0,
                    },
                )
                assert output_path.exists()
                video_count, audio_count = _stream_counts(output_path)
                assert video_count == 0
                assert audio_count == 1

    @pytest.mark.integration
    async def test_audio_fade_out_is_at_the_end(
        self, sample_audio: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """Fade-out must attenuate the END of the clip, not the start.

        Regression test for the bug where ``afade type=out`` was applied with
        no start time, so ffmpeg defaulted ``st=0`` and the fade happened at
        the very beginning (leaving the rest silent) instead of at the end.

        The sample audio is a steady 3s tone. After a 1s fade-out the output
        must: keep its ~3s duration, still be loud near the start, and be much
        quieter near the end.
        """
        async with Client(mcp_server) as client:
            output_path = temp_dir / "faded_out_placement.wav"
            _ = await client.call_tool(
                "audio_fade_out",
                {
                    "input_path": str(sample_audio),
                    "output_path": str(output_path),
                    "duration": 1.0,
                },
            )
            assert output_path.exists()

            # Duration must be preserved (the fade does not truncate the clip).
            probe = ffmpeg.probe(str(output_path))
            duration = float(probe["format"]["duration"])
            assert 2.8 <= duration <= 3.2, f"unexpected duration {duration}"

            # The start must still carry (near) full-scale audio: if the fade
            # had wrongly landed at st=0 this window would be heavily attenuated.
            start_db = _mean_volume_db(output_path, 0.0, 0.5)
            # The tail (inside the fade region) must be markedly quieter.
            end_db = _mean_volume_db(output_path, 2.5, 0.5)

            # -40 dB is a generous "not silent" floor: the sine fixture's full
            # level sits near -21 dB, while faded-out silence drops far below.
            assert start_db > -40.0, f"start unexpectedly quiet ({start_db} dB)"
            assert end_db < start_db - 10.0, (
                f"fade-out did not attenuate the end: start={start_db} dB, "
                f"end={end_db} dB"
            )

    @pytest.mark.integration
    async def test_audio_fade_in_is_at_the_start(
        self, sample_audio: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """Fade-in must attenuate the START of the clip and leave the end loud."""
        async with Client(mcp_server) as client:
            output_path = temp_dir / "faded_in_placement.wav"
            _ = await client.call_tool(
                "audio_fade_in",
                {
                    "input_path": str(sample_audio),
                    "output_path": str(output_path),
                    "duration": 1.0,
                },
            )
            assert output_path.exists()

            probe = ffmpeg.probe(str(output_path))
            duration = float(probe["format"]["duration"])
            assert 2.8 <= duration <= 3.2, f"unexpected duration {duration}"

            start_db = _mean_volume_db(output_path, 0.0, 0.5)
            end_db = _mean_volume_db(output_path, 2.5, 0.5)

            # -40 dB is a generous "not silent" floor: the sine fixture's full
            # level sits near -21 dB, while the faded-in start drops far below.
            assert end_db > -40.0, f"end unexpectedly quiet ({end_db} dB)"
            assert start_db < end_db - 10.0, (
                f"fade-in did not attenuate the start: start={start_db} dB, "
                f"end={end_db} dB"
            )

    @pytest.mark.integration
    async def test_mix_audio_workflow(
        self, sample_audio: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """mix_audio combines two audio tracks into a single output."""
        # Build a second audio source at a different frequency.
        second_audio = temp_dir / "second_audio.mp3"
        (
            ffmpeg.input("sine=frequency=880:duration=3", f="lavfi")
            .output(str(second_audio), acodec="mp3", audio_bitrate="192k")
            .overwrite_output()
            .run(quiet=True)
        )

        async with Client(mcp_server) as client:
            output_path = temp_dir / "mixed.mp3"
            _ = await client.call_tool(
                "mix_audio",
                {
                    "audio1_path": str(sample_audio),
                    "audio2_path": str(second_audio),
                    "output_path": str(output_path),
                    "audio1_volume": 0.5,
                    "audio2_volume": 0.5,
                },
            )
            assert output_path.exists()
            video_count, audio_count = _stream_counts(output_path)
            assert video_count == 0
            assert audio_count == 1

    @pytest.mark.integration
    async def test_normalize_loudness_preserves_video(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """normalize_loudness on a video keeps the video and normalizes audio."""
        async with Client(mcp_server) as client:
            output_path = temp_dir / "normalized_video.mp4"
            _ = await client.call_tool(
                "normalize_loudness",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "target_i": -14.0,
                    "target_tp": -1.0,
                    "target_lra": 11.0,
                },
            )
            assert output_path.exists()
            video_count, audio_count = _stream_counts(output_path)
            assert video_count == 1
            assert audio_count == 1

    @pytest.mark.integration
    async def test_normalize_loudness_audio_only(
        self, sample_audio: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """normalize_loudness on an audio-only file stays audio-only."""
        async with Client(mcp_server) as client:
            output_path = temp_dir / "normalized_audio.wav"
            _ = await client.call_tool(
                "normalize_loudness",
                {
                    "input_path": str(sample_audio),
                    "output_path": str(output_path),
                },
            )
            assert output_path.exists()
            video_count, audio_count = _stream_counts(output_path)
            assert video_count == 0
            assert audio_count == 1

    @pytest.mark.integration
    async def test_normalize_loudness_invalid_target(
        self, sample_audio: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """Out-of-range loudness targets must be rejected."""
        async with Client(mcp_server) as client:
            with pytest.raises((ValueError, RuntimeError, ToolError)):
                await client.call_tool(
                    "normalize_loudness",
                    {
                        "input_path": str(sample_audio),
                        "output_path": str(temp_dir / "bad.mp3"),
                        "target_i": 10.0,  # Above the -5.0 ceiling
                    },
                )

    @pytest.mark.integration
    async def test_audio_synchronization_workflow(
        self,
        sample_video: Path,
        sample_audio: Path,
        temp_dir: Path,
        mcp_server: FastMCP[None],
    ) -> None:
        """Test audio synchronization and duration matching.

        This test verifies that audio is properly synchronized with video
        and handles duration mismatches correctly.
        """
        async with Client(mcp_server) as client:
            # First, create a longer audio file for testing
            long_audio_path = temp_dir / "long_audio.mp3"
            (
                ffmpeg.input(
                    "sine=frequency=880:duration=10",  # 10-second sine wave
                    f="lavfi",
                )
                .output(
                    str(long_audio_path),
                    acodec="mp3",
                    audio_bitrate="192k",
                )
                .overwrite_output()
                .run(quiet=True)
            )

            # Test replacing audio with longer audio (should be truncated)
            replace_long_path = temp_dir / "replace_long_audio.mp4"
            _ = await client.call_tool(
                "add_audio",
                {
                    "input_path": str(sample_video),
                    "audio_path": str(long_audio_path),
                    "output_path": str(replace_long_path),
                    "replace": True,
                    "audio_volume": 1.0,
                },
            )
            assert replace_long_path.exists()

            # Verify the output duration matches the video duration
            probe = ffmpeg.probe(str(replace_long_path))
            duration = float(probe["format"]["duration"])
            assert 4.9 <= duration <= 5.1  # Should match original video duration

            # Test mixing with longer audio
            mix_long_path = temp_dir / "mix_long_audio.mp4"
            _ = await client.call_tool(
                "add_audio",
                {
                    "input_path": str(sample_video),
                    "audio_path": str(long_audio_path),
                    "output_path": str(mix_long_path),
                    "replace": False,
                    "audio_volume": 0.7,
                },
            )
            assert mix_long_path.exists()

            # Verify the output duration matches the video duration
            probe = ffmpeg.probe(str(mix_long_path))
            duration = float(probe["format"]["duration"])
            assert 4.9 <= duration <= 5.1  # Should match original video duration

    @pytest.mark.integration
    async def test_audio_quality_preservation_workflow(
        self, sample_video: Path, temp_dir: Path, mcp_server: FastMCP[None]
    ) -> None:
        """Test audio quality preservation across operations.

        This test verifies that audio quality is preserved when
        extracting and re-adding audio to videos.
        """
        async with Client(mcp_server) as client:
            # Step 1: Extract high-quality audio
            hq_audio_path = temp_dir / "hq_audio.flac"
            _ = await client.call_tool(
                "extract_audio",
                {
                    "input_path": str(sample_video),
                    "output_path": str(hq_audio_path),
                    "format": "flac",  # Lossless format
                },
            )
            assert hq_audio_path.exists()

            # Step 2: Add the high-quality audio back to video
            hq_video_path = temp_dir / "hq_video.mp4"
            _ = await client.call_tool(
                "add_audio",
                {
                    "input_path": str(sample_video),
                    "audio_path": str(hq_audio_path),
                    "output_path": str(hq_video_path),
                    "replace": True,
                    "audio_volume": 1.0,
                },
            )
            assert hq_video_path.exists()

            # Step 3: Extract audio again to compare
            extracted_again_path = temp_dir / "extracted_again.wav"
            _ = await client.call_tool(
                "extract_audio",
                {
                    "input_path": str(hq_video_path),
                    "output_path": str(extracted_again_path),
                    "format": "wav",
                },
            )
            assert extracted_again_path.exists()

            # Verify audio properties are maintained
            for audio_path in [hq_audio_path, extracted_again_path]:
                probe = ffmpeg.probe(str(audio_path))
                _ = next(s for s in probe["streams"] if s["codec_type"] == "audio")
                # Duration should be approximately the same
                duration = float(probe["format"]["duration"])
                assert 4.9 <= duration <= 5.1

    @pytest.mark.integration
    async def test_audio_processing_with_video_operations(
        self,
        sample_video: Path,
        sample_audio: Path,
        temp_dir: Path,
        mcp_server: FastMCP[None],
    ) -> None:
        """Test audio processing combined with video operations.

        This test verifies that audio processing works correctly
        when combined with other video editing operations.
        """
        async with Client(mcp_server) as client:
            # Step 1: Trim video first
            trimmed_video_path = temp_dir / "trimmed_for_audio.mp4"
            _ = await client.call_tool(
                "trim_video",
                {
                    "input_path": str(sample_video),
                    "output_path": str(trimmed_video_path),
                    "start_time": 1.0,
                    "duration": 3.0,
                },
            )
            assert trimmed_video_path.exists()

            # Step 2: Extract audio from trimmed video
            trimmed_audio_path = temp_dir / "trimmed_audio.mp3"
            _ = await client.call_tool(
                "extract_audio",
                {
                    "input_path": str(trimmed_video_path),
                    "output_path": str(trimmed_audio_path),
                    "format": "mp3",
                    "bitrate": "320k",
                },
            )
            assert trimmed_audio_path.exists()

            # Step 3: Resize the trimmed video
            resized_video_path = temp_dir / "resized_for_audio.mp4"
            _ = await client.call_tool(
                "resize_video",
                {
                    "input_path": str(trimmed_video_path),
                    "output_path": str(resized_video_path),
                    "scale": 0.75,
                },
            )
            assert resized_video_path.exists()

            # Step 4: Add new audio to resized video
            final_video_path = temp_dir / "final_with_audio.mp4"
            _ = await client.call_tool(
                "add_audio",
                {
                    "input_path": str(resized_video_path),
                    "audio_path": str(sample_audio),
                    "output_path": str(final_video_path),
                    "replace": True,
                    "audio_volume": 1.2,
                },
            )
            assert final_video_path.exists()

            # Verify final video properties
            probe = ffmpeg.probe(str(final_video_path))
            video_stream = next(
                s for s in probe["streams"] if s["codec_type"] == "video"
            )
            audio_stream = next(
                s for s in probe["streams"] if s["codec_type"] == "audio"
            )

            # Check video dimensions (should be 75% of original)
            assert video_stream["width"] == 960  # 1280 * 0.75
            assert video_stream["height"] == 540  # 720 * 0.75

            # Check duration (should be ~3 seconds from trim)
            duration = float(probe["format"]["duration"])
            assert 2.9 <= duration <= 3.1

            # Check audio is present
            assert audio_stream is not None
            assert audio_stream["codec_name"] == "aac"

    @pytest.mark.integration
    async def test_audio_error_handling(
        self,
        sample_video: Path,
        temp_dir: Path,
        mcp_server: FastMCP[None],
    ) -> None:
        """Test error handling in audio processing operations.

        This test verifies that audio processing tools handle errors
        correctly and provide meaningful error messages.
        """
        async with Client(mcp_server) as client:
            # Test extracting audio from non-existent file
            with pytest.raises(ToolError):
                await client.call_tool(
                    "extract_audio",
                    {
                        "input_path": "nonexistent.mp4",
                        "output_path": str(temp_dir / "audio.mp3"),
                        "format": "mp3",
                    },
                )

            # Test invalid audio format
            with pytest.raises((ValueError, RuntimeError, ToolError)):
                await client.call_tool(
                    "extract_audio",
                    {
                        "input_path": str(sample_video),
                        "output_path": str(temp_dir / "audio.xyz"),
                        "format": "invalid_format",
                    },
                )

            # Test adding audio with invalid volume
            with pytest.raises((ValueError, RuntimeError, ToolError)):
                await client.call_tool(
                    "add_audio",
                    {
                        "input_path": str(sample_video),
                        "audio_path": str(sample_video),
                        "output_path": str(temp_dir / "output.mp4"),
                        "audio_volume": 5.0,  # Too high
                    },
                )

            # Test adding audio with negative volume
            with pytest.raises((ValueError, RuntimeError, ToolError)):
                await client.call_tool(
                    "add_audio",
                    {
                        "input_path": str(sample_video),
                        "audio_path": str(sample_video),
                        "output_path": str(temp_dir / "output.mp4"),
                        "audio_volume": -0.5,  # Negative
                    },
                )

            # Test adding non-existent audio file
            with pytest.raises(ToolError):
                await client.call_tool(
                    "add_audio",
                    {
                        "input_path": str(sample_video),
                        "audio_path": "nonexistent_audio.mp3",
                        "output_path": str(temp_dir / "output.mp4"),
                        "replace": True,
                    },
                )

            # Test adding audio to non-existent video
            with pytest.raises(ToolError):
                await client.call_tool(
                    "add_audio",
                    {
                        "input_path": "nonexistent_video.mp4",
                        "audio_path": str(sample_video),
                        "output_path": str(temp_dir / "output.mp4"),
                        "replace": True,
                    },
                )

    @pytest.mark.integration
    async def test_audio_bitrate_variations(
        self,
        sample_video: Path,
        temp_dir: Path,
        mcp_server: FastMCP[None],
    ) -> None:
        """Test audio extraction with different bitrate settings.

        This test verifies that different bitrate settings work correctly
        and produce files with expected properties.
        """
        bitrate_tests = [
            ("96k", "low_quality.mp3"),
            ("128k", "standard_quality.mp3"),
            ("192k", "good_quality.mp3"),
            ("320k", "high_quality.mp3"),
        ]

        async with Client(mcp_server) as client:
            file_sizes = []

            for bitrate, filename in bitrate_tests:
                output_path = temp_dir / filename

                _ = await client.call_tool(
                    "extract_audio",
                    {
                        "input_path": str(sample_video),
                        "output_path": str(output_path),
                        "format": "mp3",
                        "bitrate": bitrate,
                    },
                )
                assert output_path.exists()

                # Verify file properties
                probe = ffmpeg.probe(str(output_path))
                audio_stream = next(
                    s for s in probe["streams"] if s["codec_type"] == "audio"
                )
                assert audio_stream["codec_name"] == "mp3"

                # Track file size for comparison
                file_size = output_path.stat().st_size
                file_sizes.append(file_size)

            # Verify that higher bitrates generally produce larger files
            # (with some tolerance for encoding variations)
            for i in range(len(file_sizes) - 1):
                # Each higher bitrate should generally be larger (with 20% tolerance)
                size_ratio = file_sizes[i + 1] / file_sizes[i]
                assert (
                    size_ratio > 0.8
                )  # Allow for some compression efficiency variation
