"""Tests for batch/manifest automation tools.

Covers ``stitch_from_manifest`` (single-call manifest stitching that mixes plain
cuts with crossfade transitions and per-clip trimming) and ``batch_convert``
(bounded-concurrency format conversion). Manifest validation and filtergraph
shape are exercised as fast unit tests (no ffmpeg binary needed); the stitch /
convert behaviour is verified end-to-end against the real ffmpeg binary
(available in Docker/CI) as integration tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import ffmpeg
import pytest
from fastmcp import Client

from vfx_mcp.tools.batch_automation import (
    _build_manifest_streams,
    _ManifestEntry,
    _parse_manifest,
)

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


def _probe_codecs(path: Path) -> tuple[str, str]:
    """Return ``(video_codec_name, audio_codec_name)`` for a file."""
    probe = cast(dict[str, object], ffmpeg.probe(str(path)))
    streams = cast(list[dict[str, object]], probe["streams"])
    video = next(
        (s for s in streams if s.get("codec_type") == "video"),
        cast(dict[str, object], {}),
    )
    audio = next(
        (s for s in streams if s.get("codec_type") == "audio"),
        cast(dict[str, object], {}),
    )
    return (
        cast(str, video.get("codec_name", "")),
        cast(str, audio.get("codec_name", "")),
    )


class TestManifestValidation:
    """Fast unit tests for manifest structural validation."""

    @pytest.mark.unit
    def test_empty_manifest_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one entry"):
            _parse_manifest([])

    @pytest.mark.unit
    def test_missing_clip_rejected(self) -> None:
        with pytest.raises(ValueError, match="clip is required"):
            _parse_manifest([{"start": 1.0}])

    @pytest.mark.unit
    def test_blank_clip_rejected(self) -> None:
        with pytest.raises(ValueError, match="clip is required"):
            _parse_manifest([{"clip": "   "}])

    @pytest.mark.unit
    def test_non_object_entry_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be an object"):
            _parse_manifest(cast("list[dict[str, object]]", ["not-a-dict"]))

    @pytest.mark.unit
    def test_bad_start_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="start must be a number"):
            _parse_manifest([{"clip": "a.mp4", "start": "oops"}])

    @pytest.mark.unit
    def test_bool_number_rejected(self) -> None:
        # bool is a subclass of int, but a boolean here is a mistake.
        with pytest.raises(ValueError, match="start must be a number"):
            _parse_manifest([{"clip": "a.mp4", "start": True}])

    @pytest.mark.unit
    def test_end_not_after_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be greater than"):
            _parse_manifest([{"clip": "a.mp4", "start": 2.0, "end": 1.0}])

    @pytest.mark.unit
    def test_negative_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="start must be >= 0"):
            _parse_manifest([{"clip": "a.mp4", "start": -1.0}])

    @pytest.mark.unit
    def test_bad_transition_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="transition must be a string"):
            _parse_manifest([{"clip": "a.mp4", "transition": 3}])

    @pytest.mark.unit
    def test_non_positive_transition_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="transition_duration must be"):
            _parse_manifest([{"clip": "a.mp4", "transition_duration": 0.0}])

    @pytest.mark.unit
    def test_valid_manifest_parses(self) -> None:
        entries = _parse_manifest(
            [
                {"clip": "a.mp4"},
                {
                    "clip": "b.mp4",
                    "start": 0.5,
                    "end": 1.5,
                    "transition": "fade",
                    "transition_duration": 0.5,
                },
            ]
        )
        assert len(entries) == 2
        assert entries[0] == _ManifestEntry("a.mp4", None, None, None, None)
        assert entries[1] == _ManifestEntry("b.mp4", 0.5, 1.5, "fade", 0.5)


class TestManifestGraphShape:
    """Unit tests on the compiled ffmpeg command (no ffmpeg binary needed)."""

    @pytest.mark.unit
    def test_mixed_cut_and_transition_graph(self) -> None:
        """A cut + a trimmed crossfade must compile to concat + xfade + trim."""
        entries = [
            _ManifestEntry("a.mp4", None, None, None, None),
            _ManifestEntry("b.mp4", 0.5, 1.5, "fade", 0.5),
            _ManifestEntry("c.mp4", None, None, None, None),
        ]
        video, audio = _build_manifest_streams(
            entries,
            ["a.mp4", "b.mp4", "c.mp4"],
            durations=[2.0, 1.0, 2.0],
            has_audio=[True, True, False],
            target_width=640,
            target_height=480,
            target_fps=24,
        )
        out = ffmpeg.output(
            video,
            audio,
            "out.mp4",
            vcodec="libx264",
            pix_fmt="yuv420p",
            acodec="aac",
        )
        joined = " ".join(out.get_args())

        # Crossfade for the transition junction, concat for the cut junction.
        assert "xfade=" in joined
        assert "acrossfade=" in joined
        assert "concat=" in joined
        # Trim applied to the second (in/out-pointed) clip.
        assert "trim=" in joined
        assert "atrim=" in joined
        # The silent third clip is backed by anullsrc.
        assert "anullsrc" in joined
        # xfade offset == cumulative duration so far (clip0=2.0) minus the
        # transition length (0.5) == 1.5.
        assert "offset=1.5" in joined

    @pytest.mark.unit
    def test_all_cuts_graph(self) -> None:
        """A manifest with no transitions compiles to pure concat cuts."""
        entries = [
            _ManifestEntry("a.mp4", None, None, None, None),
            _ManifestEntry("b.mp4", None, None, None, None),
        ]
        video, audio = _build_manifest_streams(
            entries,
            ["a.mp4", "b.mp4"],
            durations=[2.0, 2.0],
            has_audio=[True, True],
            target_width=640,
            target_height=480,
            target_fps=24,
        )
        out = ffmpeg.output(video, audio, "out.mp4")
        joined = " ".join(out.get_args())
        assert "concat=" in joined
        assert "xfade=" not in joined


class TestStitchFromManifest:
    """End-to-end tests for ``stitch_from_manifest``."""

    @pytest.mark.integration
    async def test_three_entry_mixed_manifest(
        self,
        sample_videos: list[Path],
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """3 entries: cut, trimmed crossfade, cut -> ~4.5s single A/V file.

        clip0 (2s) --cut--> [clip1 trimmed to 1s, 0.5s fade] --cut--> clip2 (2s)
        Expected duration: 2.0 + (1.0 - 0.5) + 2.0 = 4.5s.
        """
        monkeypatch.setenv("VFX_WORKSPACE", str(temp_dir))
        output_path = temp_dir / "manifest_mixed.mp4"

        manifest = [
            {"clip": str(sample_videos[0])},
            {
                "clip": str(sample_videos[1]),
                "start": 0.5,
                "end": 1.5,
                "transition": "fade",
                "transition_duration": 0.5,
            },
            {"clip": str(sample_videos[2])},
        ]

        async with Client(mcp_server) as client:
            await client.call_tool(
                "stitch_from_manifest",
                {"manifest": manifest, "output_path": str(output_path)},
            )

        assert output_path.exists()
        video, audio, duration = _probe_streams(output_path)
        assert video == 1
        assert audio == 1
        assert abs(duration - 4.5) <= 0.3

    @pytest.mark.integration
    async def test_manifest_with_silent_clip(
        self,
        sample_videos: list[Path],
        sample_videos_silent: list[Path],
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A silent clip mixed with audio clips still yields one audio track."""
        monkeypatch.setenv("VFX_WORKSPACE", str(temp_dir))
        output_path = temp_dir / "manifest_silent.mp4"

        manifest = [
            {"clip": str(sample_videos[0])},
            {
                "clip": str(sample_videos_silent[0]),
                "transition": "dissolve",
                "transition_duration": 0.5,
            },
        ]

        async with Client(mcp_server) as client:
            await client.call_tool(
                "stitch_from_manifest",
                {"manifest": manifest, "output_path": str(output_path)},
            )

        assert output_path.exists()
        video, audio, duration = _probe_streams(output_path)
        assert video == 1
        assert audio == 1
        # 2.0 + (2.0 - 0.5) = 3.5s
        assert abs(duration - 3.5) <= 0.3

    @pytest.mark.integration
    async def test_empty_manifest_errors(
        self,
        mcp_server: FastMCP[None],
    ) -> None:
        """An empty manifest is rejected before any ffmpeg work."""
        async with Client(mcp_server) as client:
            with pytest.raises(Exception, match="at least one entry"):
                await client.call_tool(
                    "stitch_from_manifest",
                    {"manifest": [], "output_path": "out.mp4"},
                )

    @pytest.mark.integration
    async def test_start_beyond_duration_errors(
        self,
        sample_videos: list[Path],
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A ``start`` at or past a clip's duration is rejected clearly.

        The 2s sample clip cannot supply content starting at 5s, so the tool
        must raise before emitting a 0-length clip that fails opaquely later.
        """
        monkeypatch.setenv("VFX_WORKSPACE", str(temp_dir))
        output_path = temp_dir / "manifest_bad_start.mp4"

        manifest = [{"clip": str(sample_videos[0]), "start": 5.0}]

        async with Client(mcp_server) as client:
            with pytest.raises(Exception, match="start.*past its duration"):
                await client.call_tool(
                    "stitch_from_manifest",
                    {"manifest": manifest, "output_path": str(output_path)},
                )


class TestBatchConvert:
    """End-to-end tests for ``batch_convert``."""

    @pytest.mark.integration
    async def test_batch_convert_three_clips(
        self,
        sample_videos: list[Path],
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Converting 3 clips produces 3 output files in the target dir."""
        monkeypatch.setenv("VFX_WORKSPACE", str(temp_dir))
        output_dir = temp_dir / "converted"

        async with Client(mcp_server) as client:
            await client.call_tool(
                "batch_convert",
                {
                    "input_paths": [str(v) for v in sample_videos],
                    "output_dir": str(output_dir),
                    "format": "mkv",
                },
            )

        outputs = sorted(output_dir.glob("*.mkv"))
        assert len(outputs) == 3
        for out in outputs:
            _, _, duration = _probe_streams(out)
            assert abs(duration - 2.0) <= 0.3

    @pytest.mark.integration
    async def test_batch_convert_webm_uses_webm_codecs(
        self,
        sample_videos: list[Path],
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Converting to webm must pick a webm-valid codec pairing.

        The webm muxer rejects H.264/AAC, so ``format='webm'`` must select the
        shared per-container defaults (VP9 video + Vorbis audio) and produce a
        playable file rather than failing at runtime.
        """
        monkeypatch.setenv("VFX_WORKSPACE", str(temp_dir))
        output_dir = temp_dir / "converted_webm"

        async with Client(mcp_server) as client:
            await client.call_tool(
                "batch_convert",
                {
                    "input_paths": [str(sample_videos[0])],
                    "output_dir": str(output_dir),
                    "format": "webm",
                },
            )

        outputs = sorted(output_dir.glob("*.webm"))
        assert len(outputs) == 1
        video_codec, audio_codec = _probe_codecs(outputs[0])
        # VP8/VP9/AV1 video + Vorbis/Opus audio are the webm-valid pairings.
        assert video_codec in {"vp8", "vp9", "av1"}
        assert audio_codec in {"vorbis", "opus"}

    @pytest.mark.integration
    async def test_batch_convert_same_stem_collision_errors(
        self,
        sample_videos: list[Path],
        temp_dir: Path,
        mcp_server: FastMCP[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two inputs with the same stem map to one output and must be rejected.

        ``a/clip.mp4`` and ``b/clip.mp4`` both target ``<out>/clip.mkv``; the
        tool must raise a clear ``ValueError`` naming the colliding sources
        rather than racing two concurrent encodes onto one file.
        """
        monkeypatch.setenv("VFX_WORKSPACE", str(temp_dir))

        dir_a = temp_dir / "a"
        dir_b = temp_dir / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        clip_a = dir_a / "clip.mp4"
        clip_b = dir_b / "clip.mp4"
        clip_a.write_bytes(sample_videos[0].read_bytes())
        clip_b.write_bytes(sample_videos[1].read_bytes())

        output_dir = temp_dir / "converted_collision"

        async with Client(mcp_server) as client:
            with pytest.raises(Exception, match="same output path"):
                await client.call_tool(
                    "batch_convert",
                    {
                        "input_paths": [str(clip_a), str(clip_b)],
                        "output_dir": str(output_dir),
                        "format": "mkv",
                    },
                )

    @pytest.mark.integration
    async def test_batch_convert_empty_errors(
        self,
        mcp_server: FastMCP[None],
    ) -> None:
        """An empty input list is rejected."""
        async with Client(mcp_server) as client:
            with pytest.raises(Exception, match="at least one input"):
                await client.call_tool(
                    "batch_convert",
                    {"input_paths": [], "output_dir": "out"},
                )
