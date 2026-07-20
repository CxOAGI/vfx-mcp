"""Tests for video transition stitching.

Exercises ``stitch_with_transitions``, which joins N clips with ``xfade``
(video) and ``acrossfade`` (audio) transitions chained pairwise. These are
integration tests: they invoke the real ffmpeg binary (available in Docker/CI)
and assert on the produced file's duration and stream layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import ffmpeg
import pytest
from fastmcp import Client

if TYPE_CHECKING:
    from fastmcp import FastMCP


def _probe_streams(path: Path) -> tuple[int, int, float]:
    """Return ``(video_stream_count, audio_stream_count, duration)`` for a file."""
    probe = cast(dict[str, object], ffmpeg.probe(str(path)))
    streams = cast(list[dict[str, object]], probe["streams"])
    video = sum(1 for s in streams if s.get("codec_type") == "video")
    audio = sum(1 for s in streams if s.get("codec_type") == "audio")
    fmt = cast(dict[str, object], probe["format"])
    duration = float(cast(str, fmt["duration"]))
    return video, audio, duration


class TestStitchWithTransitions:
    """Test suite for ``stitch_with_transitions``."""

    @pytest.mark.integration
    async def test_stitch_two_clips(
        self,
        sample_videos: list[Path],
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two 2s clips with a 1s transition should yield a ~3s A/V file."""
        monkeypatch.setenv("VFX_WORKSPACE", str(temp_dir))
        output_path: Path = temp_dir / "stitched_two.mp4"
        transition_duration = 1.0

        async with Client(mcp_server) as client:
            await client.call_tool(
                "stitch_with_transitions",
                {
                    "input_paths": [str(v) for v in sample_videos[:2]],
                    "output_path": str(output_path),
                    "transition": "fade",
                    "duration": transition_duration,
                },
            )

        assert output_path.exists()
        video, audio, duration = _probe_streams(output_path)
        assert video == 1
        assert audio == 1
        # sum(2, 2) - (2 - 1) * 1 = 3.0
        expected = 4.0 - transition_duration
        assert abs(duration - expected) <= 0.3

    @pytest.mark.integration
    async def test_stitch_three_clips_duration(
        self,
        sample_videos: list[Path],
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Three 2s clips with 1s transitions should total ~4s with both streams."""
        monkeypatch.setenv("VFX_WORKSPACE", str(temp_dir))
        output_path: Path = temp_dir / "stitched_three.mp4"
        transition_duration = 1.0

        async with Client(mcp_server) as client:
            await client.call_tool(
                "stitch_with_transitions",
                {
                    "input_paths": [str(v) for v in sample_videos],
                    "output_path": str(output_path),
                    "transition": "dissolve",
                    "duration": transition_duration,
                },
            )

        assert output_path.exists()
        video, audio, duration = _probe_streams(output_path)
        assert video == 1
        assert audio == 1
        # sum(2, 2, 2) - (3 - 1) * 1 = 4.0
        expected = 6.0 - 2 * transition_duration
        assert abs(duration - expected) <= 0.3

    @pytest.mark.integration
    async def test_stitch_heterogeneous_clips(
        self,
        sample_videos_heterogeneous: list[Path],
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Clips of differing resolution/fps are normalized then stitched."""
        monkeypatch.setenv("VFX_WORKSPACE", str(temp_dir))
        output_path: Path = temp_dir / "stitched_hetero.mp4"
        transition_duration = 1.0

        async with Client(mcp_server) as client:
            await client.call_tool(
                "stitch_with_transitions",
                {
                    "input_paths": [str(v) for v in sample_videos_heterogeneous],
                    "output_path": str(output_path),
                    "transition": "wipe_left",
                    "duration": transition_duration,
                },
            )

        assert output_path.exists()
        video, audio, duration = _probe_streams(output_path)
        assert video == 1
        assert audio == 1
        # Normalized to the largest input (1280x720); all sources are 2s.
        probe = cast(dict[str, object], ffmpeg.probe(str(output_path)))
        streams = cast(list[dict[str, object]], probe["streams"])
        video_stream = next(s for s in streams if s.get("codec_type") == "video")
        assert int(cast(int, video_stream["width"])) == 1280
        assert int(cast(int, video_stream["height"])) == 720
        expected = 6.0 - 2 * transition_duration
        assert abs(duration - expected) <= 0.3

    @pytest.mark.integration
    async def test_stitch_silent_clip(
        self,
        sample_videos: list[Path],
        sample_videos_silent: list[Path],
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A silent clip is backed by anullsrc so acrossfade still succeeds."""
        monkeypatch.setenv("VFX_WORKSPACE", str(temp_dir))
        output_path: Path = temp_dir / "stitched_silent.mp4"
        transition_duration = 1.0

        # One sound clip, one silent clip (from a different temp dir), so
        # copy the silent clip alongside so it is inside the workspace.
        silent_copy = temp_dir / "silent_clip.mp4"
        silent_copy.write_bytes(sample_videos_silent[0].read_bytes())

        async with Client(mcp_server) as client:
            await client.call_tool(
                "stitch_with_transitions",
                {
                    "input_paths": [str(sample_videos[0]), str(silent_copy)],
                    "output_path": str(output_path),
                    "duration": transition_duration,
                },
            )

        assert output_path.exists()
        video, audio, duration = _probe_streams(output_path)
        assert video == 1
        assert audio == 1
        expected = 4.0 - transition_duration
        assert abs(duration - expected) <= 0.3

    @pytest.mark.unit
    async def test_stitch_requires_two_clips(
        self,
        sample_videos: list[Path],
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fewer than two clips is rejected."""
        monkeypatch.setenv("VFX_WORKSPACE", str(temp_dir))
        async with Client(mcp_server) as client:
            with pytest.raises(Exception) as exc_info:
                await client.call_tool(
                    "stitch_with_transitions",
                    {
                        "input_paths": [str(sample_videos[0])],
                        "output_path": str(temp_dir / "nope.mp4"),
                    },
                )
        assert "at least 2 clips" in str(exc_info.value).lower()

    @pytest.mark.unit
    async def test_stitch_rejects_unknown_transition(
        self,
        sample_videos: list[Path],
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unsupported transition name is rejected before encoding."""
        monkeypatch.setenv("VFX_WORKSPACE", str(temp_dir))
        async with Client(mcp_server) as client:
            with pytest.raises(Exception) as exc_info:
                await client.call_tool(
                    "stitch_with_transitions",
                    {
                        "input_paths": [str(v) for v in sample_videos[:2]],
                        "output_path": str(temp_dir / "nope.mp4"),
                        "transition": "explode",
                    },
                )
        assert "transition type" in str(exc_info.value).lower()

    @pytest.mark.integration
    async def test_stitch_rejects_overlong_transition(
        self,
        sample_videos: list[Path],
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A transition at least as long as a clip is rejected."""
        monkeypatch.setenv("VFX_WORKSPACE", str(temp_dir))
        async with Client(mcp_server) as client:
            with pytest.raises(Exception) as exc_info:
                await client.call_tool(
                    "stitch_with_transitions",
                    {
                        "input_paths": [str(v) for v in sample_videos[:2]],
                        "output_path": str(temp_dir / "nope.mp4"),
                        # Clips are only ~2s long.
                        "duration": 5.0,
                    },
                )
        assert "shorter than every clip" in str(exc_info.value).lower()
