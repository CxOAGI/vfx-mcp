"""Unit tests for core media/utilities helpers.

Covers the canonical ``even_dimension`` rounding, ``create_standard_output``
flag emission (inspected via ``ffmpeg.get_args`` so no ffmpeg binary is
required), and the failure/cancellation cleanup behaviour of
``run_ffmpeg_async``. The ffmpeg subprocess is faked throughout, so these tests
never launch ffmpeg and are safe to run without the binary installed.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import ffmpeg
import pytest

from vfx_mcp.core import utilities
from vfx_mcp.core.media import even_dimension
from vfx_mcp.core.utilities import create_standard_output, run_ffmpeg_async


class TestEvenDimension:
    """The canonical ``even_dimension`` rounds DOWN to even, clamped to >= 2."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1920, 1920),  # already even -> unchanged
            (1921, 1920),  # odd -> rounds down
            (3, 2),  # odd small -> down to 2
            (2, 2),  # minimum valid dimension
            (1, 2),  # below minimum -> clamps to 2
            (0, 2),  # degenerate -> clamps to 2
        ],
    )
    def test_rounds_down_and_clamps(self, value: int, expected: int) -> None:
        assert even_dimension(value) == expected


class TestCreateStandardOutput:
    """``create_standard_output`` emits the expected ffmpeg flags."""

    @pytest.mark.unit
    def test_quality_flags_emitted(self) -> None:
        stream = ffmpeg.input("in.mp4")
        args = create_standard_output(
            stream,
            "out.mp4",
            crf=23,
            preset="medium",
            video_bitrate="2M",
            audio_bitrate="128k",
            faststart=True,
        ).get_args()

        assert args[-1] == "out.mp4"
        assert "libx264" in args
        assert "aac" in args
        assert "yuv420p" in args
        # CRF, preset and bitrates are threaded through verbatim.
        assert args[args.index("-crf") + 1] == "23"
        assert args[args.index("-preset") + 1] == "medium"
        assert args[args.index("-b:v") + 1] == "2M"
        assert args[args.index("-b:a") + 1] == "128k"
        # faststart relocates the moov atom for progressive playback.
        assert args[args.index("-movflags") + 1] == "+faststart"

    @pytest.mark.unit
    def test_copy_mode_avoids_reencode_flags(self) -> None:
        stream = ffmpeg.input("in.mkv")
        args = create_standard_output(
            stream,
            "out.mkv",
            copy_video=True,
            copy_audio=True,
        ).get_args()

        # Both streams are copied; no encoder or pixel-format flags appear.
        assert args[args.index("-vcodec") + 1] == "copy"
        assert args[args.index("-acodec") + 1] == "copy"
        assert "libx264" not in args
        assert "-pix_fmt" not in args
        assert "-crf" not in args

    @pytest.mark.unit
    def test_faststart_disabled_by_default(self) -> None:
        stream = ffmpeg.input("in.mp4")
        args = create_standard_output(stream, "out.mp4").get_args()
        assert "-movflags" not in args


class _FakeProcess:
    """Stand-in for the ``subprocess.Popen`` returned by ffmpeg.run_async.

    ``communicate`` optionally blocks on an event so a test can cancel or time
    out the awaiting task mid-encode; ``kill``/``wait`` release that event so
    the worker thread unwinds promptly.
    """

    def __init__(
        self,
        *,
        block: bool,
        returncode: int = 0,
        stderr: bytes = b"",
    ) -> None:
        self._release = threading.Event()
        self._block = block
        self._returncode = returncode
        self._stderr = stderr
        self.killed = False
        self.wait_called = False

    def communicate(self) -> tuple[bytes, bytes]:
        if self._block:
            # Bounded so a misbehaving test can never hang the suite forever.
            self._release.wait(timeout=10)
        return b"", self._stderr

    def kill(self) -> None:
        self.killed = True
        self._release.set()

    def wait(self) -> int:
        self.wait_called = True
        self._release.set()
        return self._returncode

    def poll(self) -> int:
        return self._returncode


def _patch_run_async(monkeypatch: pytest.MonkeyPatch, process: _FakeProcess) -> None:
    """Make ``ffmpeg.run_async`` return ``process`` without launching ffmpeg."""

    def _fake_run_async(*_args: object, **_kwargs: object) -> _FakeProcess:
        return process

    monkeypatch.setattr(utilities.ffmpeg, "run_async", _fake_run_async)


def _dummy_stream() -> ffmpeg.Stream:
    """A throwaway output node; never actually executed (run_async is faked)."""
    return ffmpeg.input("in.mp4").output("out.mp4")


class TestRunFfmpegAsyncCleanup:
    """``run_ffmpeg_async`` kills the process and cleans up partial output."""

    @pytest.mark.unit
    async def test_cancellation_kills_process_and_unlinks_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        process = _FakeProcess(block=True)
        _patch_run_async(monkeypatch, process)

        partial = tmp_path / "out.mp4"
        partial.write_bytes(b"partial")

        task = asyncio.create_task(
            run_ffmpeg_async(_dummy_stream(), output_path=str(partial))
        )
        # Let the task reach the blocking communicate() await before cancelling.
        await asyncio.sleep(0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert process.killed is True
        # The half-written deliverable must be removed on cancellation.
        assert not partial.exists()

    @pytest.mark.unit
    async def test_cancellation_reraised_without_output_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        process = _FakeProcess(block=True)
        _patch_run_async(monkeypatch, process)

        task = asyncio.create_task(run_ffmpeg_async(_dummy_stream()))
        await asyncio.sleep(0.1)
        task.cancel()

        # Cancellation is re-raised (never swallowed) even with no output_path.
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.killed is True

    @pytest.mark.unit
    async def test_timeout_kills_process_and_unlinks_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        process = _FakeProcess(block=True)
        _patch_run_async(monkeypatch, process)

        partial = tmp_path / "out.mp4"
        partial.write_bytes(b"partial")

        with pytest.raises(RuntimeError, match="timed out"):
            await run_ffmpeg_async(
                _dummy_stream(),
                timeout=0.05,
                output_path=str(partial),
            )

        assert process.killed is True
        assert not partial.exists()

    @pytest.mark.unit
    async def test_nonzero_exit_unlinks_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        process = _FakeProcess(block=False, returncode=1, stderr=b"boom")
        _patch_run_async(monkeypatch, process)

        partial = tmp_path / "out.mp4"
        partial.write_bytes(b"partial")

        with pytest.raises(RuntimeError, match="boom"):
            await run_ffmpeg_async(_dummy_stream(), output_path=str(partial))

        # A failed encode leaves no corrupt file behind.
        assert not partial.exists()

    @pytest.mark.unit
    async def test_success_leaves_output_in_place(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        process = _FakeProcess(block=False, returncode=0)
        _patch_run_async(monkeypatch, process)

        output = tmp_path / "out.mp4"
        output.write_bytes(b"final")

        await run_ffmpeg_async(_dummy_stream(), output_path=str(output))

        # A clean exit must not delete the deliverable.
        assert output.exists()
        assert process.killed is False
