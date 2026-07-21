"""Pytest configuration and fixtures for VFX MCP tests.

This module provides shared fixtures and configuration for all tests.
Includes fixtures for creating temporary directories, sample videos,
sample audio files, and MCP server instances for testing.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import ffmpeg
import pytest

if TYPE_CHECKING:
    from fastmcp import FastMCP


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Create a temporary directory for test files.

    This fixture creates a temporary directory that is automatically
    cleaned up after the test completes. Used by other fixtures and
    tests that need file system access.

    Yields:
        Path to the temporary directory.
    """
    temp_path: str = tempfile.mkdtemp()
    try:
        yield Path(temp_path)
    finally:
        # Cleanup after test completes
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_video(temp_dir: Path) -> Path:
    """Create a sample test video using ffmpeg.

    This fixture generates a 5-second test video with color bars pattern
    and a 440Hz sine wave audio track. The video is encoded with H.264
    and AAC for broad compatibility.

    Args:
        temp_dir: Temporary directory fixture for file storage.

    Returns:
        Path to the generated test video file.
    """
    output_path: Path = temp_dir / "test_video.mp4"

    # Generate test video using lavfi (libavfilter) virtual input devices
    # testsrc creates color bars pattern, sine creates audio tone
    video_input = ffmpeg.input(
        "testsrc=duration=5:size=1280x720:rate=30",
        f="lavfi",
    )
    audio_input = ffmpeg.input("sine=frequency=440:duration=5", f="lavfi")

    # Combine video and audio into MP4 container
    stream = ffmpeg.output(
        video_input,
        audio_input,
        str(output_path),
        vcodec="libx264",
        acodec="aac",
        audio_bitrate="128k",
        **{"f": "mp4"},
    )
    ffmpeg.run(stream, overwrite_output=True, quiet=True)

    return output_path


@pytest.fixture
def sample_videos(temp_dir: Path) -> list[Path]:
    """Create multiple sample test videos with audio.

    This fixture generates three test videos with different visual patterns
    for testing concatenation and batch operations. Each video is 2 seconds
    long with distinct visual content and carries a 440Hz sine-wave audio
    track so that audio-aware concatenation can be exercised. The three
    clips share identical resolution (640x480), frame rate (24fps) and
    codecs, giving a combined duration of ~6 seconds when concatenated.

    Args:
        temp_dir: Temporary directory fixture for file storage.

    Returns:
        List of paths to the generated test video files.
    """
    videos: list[Path] = []

    # Different test patterns for visual variety
    patterns: list[str] = [
        "testsrc",  # Standard color bars
        "testsrc2",  # Advanced test pattern
        "rgbtestsrc",  # RGB color test
    ]

    for i, pattern in enumerate(patterns):
        output_path: Path = temp_dir / f"test_video_{i}.mp4"

        # Generate 2-second test videos with different patterns
        # Using ultrafast preset for faster test execution
        video_input = ffmpeg.input(
            f"{pattern}=duration=2:size=640x480:rate=24",
            f="lavfi",
        )
        # Sine-wave audio so concatenation with audio streams is testable
        audio_input = ffmpeg.input(
            "sine=frequency=440:duration=2",
            f="lavfi",
        )
        stream = ffmpeg.output(
            video_input,
            audio_input,
            str(output_path),
            vcodec="libx264",
            acodec="aac",
            audio_bitrate="128k",
            preset="ultrafast",
            **{"f": "mp4"},
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)

        videos.append(output_path)

    return videos


@pytest.fixture
def sample_videos_silent(temp_dir: Path) -> list[Path]:
    """Create multiple sample test videos WITHOUT audio.

    This fixture generates three 2-second test videos with no audio stream,
    mirroring the output of silent generators such as Veo 2. It is used to
    verify that audio-aware tools (concatenation, speed changes, audio
    extraction) degrade gracefully when inputs carry no audio track. The
    clips are homogeneous (640x480@24fps) and total ~6 seconds when joined.

    Args:
        temp_dir: Temporary directory fixture for file storage.

    Returns:
        List of paths to the generated silent test video files.
    """
    videos: list[Path] = []

    patterns: list[str] = [
        "testsrc",  # Standard color bars
        "testsrc2",  # Advanced test pattern
        "rgbtestsrc",  # RGB color test
    ]

    for i, pattern in enumerate(patterns):
        output_path: Path = temp_dir / f"test_video_silent_{i}.mp4"

        # Video-only input; no audio stream is added to the output
        video_input = ffmpeg.input(
            f"{pattern}=duration=2:size=640x480:rate=24",
            f="lavfi",
        )
        stream = ffmpeg.output(
            video_input,
            str(output_path),
            vcodec="libx264",
            preset="ultrafast",
            **{"f": "mp4"},
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)

        videos.append(output_path)

    return videos


@pytest.fixture
def sample_videos_heterogeneous(temp_dir: Path) -> list[Path]:
    """Create sample test videos with mismatched resolutions and frame rates.

    This fixture generates three 2-second test videos that deliberately differ
    in resolution and frame rate (640x480@24, 1280x720@30, 854x480@25), each
    carrying a 440Hz sine-wave audio track. It is used to verify that
    concatenation performs a normalization pre-pass (scale/pad + fps + SAR +
    pix_fmt) rather than assuming identical input properties. The clips total
    ~6 seconds of content when concatenated.

    Args:
        temp_dir: Temporary directory fixture for file storage.

    Returns:
        List of paths to the generated heterogeneous test video files.
    """
    videos: list[Path] = []

    # (pattern, size, rate) tuples with intentionally differing geometry/fps
    specs: list[tuple[str, str, int]] = [
        ("testsrc", "640x480", 24),
        ("testsrc2", "1280x720", 30),
        ("rgbtestsrc", "854x480", 25),
    ]

    for i, (pattern, size, rate) in enumerate(specs):
        output_path: Path = temp_dir / f"test_video_hetero_{i}.mp4"

        video_input = ffmpeg.input(
            f"{pattern}=duration=2:size={size}:rate={rate}",
            f="lavfi",
        )
        audio_input = ffmpeg.input(
            "sine=frequency=440:duration=2",
            f="lavfi",
        )
        stream = ffmpeg.output(
            video_input,
            audio_input,
            str(output_path),
            vcodec="libx264",
            acodec="aac",
            audio_bitrate="128k",
            preset="ultrafast",
            **{"f": "mp4"},
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)

        videos.append(output_path)

    return videos


@pytest.fixture
def sample_video_odd_dims(temp_dir: Path) -> Path:
    """Create a sample test video with odd (non-even) dimensions.

    This fixture generates a 2-second 641x481 video with a 440Hz sine-wave
    audio track. The odd width and height are used to verify that resize and
    other re-encoding operations force even output dimensions, since yuv420p
    H.264 rejects odd width/height ("not divisible by 2").

    Args:
        temp_dir: Temporary directory fixture for file storage.

    Returns:
        Path to the generated odd-dimension test video file.
    """
    output_path: Path = temp_dir / "test_video_odd.mp4"

    # testsrc permits arbitrary sizes; 641x481 is odd on both axes. libx264
    # with the default yuv420p rejects odd dimensions (chroma subsampling),
    # so encode with yuv444p, which allows odd width/height and keeps the
    # frame genuinely 641x481 for downstream even-dimension resize handling.
    video_input = ffmpeg.input(
        "testsrc=duration=2:size=641x481:rate=24",
        f="lavfi",
    )
    audio_input = ffmpeg.input(
        "sine=frequency=440:duration=2",
        f="lavfi",
    )
    stream = ffmpeg.output(
        video_input,
        audio_input,
        str(output_path),
        vcodec="libx264",
        pix_fmt="yuv444p",
        acodec="aac",
        audio_bitrate="128k",
        preset="ultrafast",
        **{"f": "mp4"},
    )
    ffmpeg.run(stream, overwrite_output=True, quiet=True)

    return output_path


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a sample still image for image-to-video tests.

    This fixture generates a single-frame 1280x720 PNG using the lavfi color
    source. It is used by tools that build video from a still image (e.g.
    ``image_to_video``).

    Args:
        temp_dir: Temporary directory fixture for file storage.

    Returns:
        Path to the generated PNG image file.
    """
    output_path: Path = temp_dir / "test_image.png"

    # A solid-color single frame is sufficient for image-to-video tests
    image_input = ffmpeg.input(
        "color=color=blue:size=1280x720",
        f="lavfi",
    )
    stream = ffmpeg.output(
        image_input,
        str(output_path),
        vframes=1,
    )
    ffmpeg.run(stream, overwrite_output=True, quiet=True)

    return output_path


@pytest.fixture
def sample_audio(temp_dir: Path) -> Path:
    """Create a sample audio file.

    This fixture generates a 3-second test audio file containing a 440Hz
    sine wave tone (A4 musical note). The audio is encoded as MP3 at 192kbps.

    Args:
        temp_dir: Temporary directory fixture for file storage.

    Returns:
        Path to the generated audio file.
    """
    output_path: Path = temp_dir / "test_audio.mp3"

    # Generate a 3-second 440Hz sine wave (A4 musical note)
    # Using lavfi sine generator for consistent test audio
    audio_input = ffmpeg.input(
        "sine=frequency=440:duration=3",
        f="lavfi",
    )
    stream = ffmpeg.output(
        audio_input,
        str(output_path),
        acodec="mp3",
        audio_bitrate="192k",
    )
    ffmpeg.run(stream, overwrite_output=True, quiet=True)

    return output_path


@pytest.fixture
def mcp_server() -> FastMCP[None]:
    """Create an MCP server instance for testing.

    This fixture provides a configured FastMCP server instance with all
    video editing tools and resource endpoints loaded. Imports the server
    from main module to avoid circular import issues.

    Returns:
        Configured FastMCP server instance ready for testing.
    """
    # Import here to avoid circular imports during module loading
    from main import mcp

    return mcp


# Pytest configuration options
def pytest_configure(
    config: pytest.Config,
) -> None:
    """Configure pytest with custom markers.

    Registers custom markers for categorizing tests by type and performance
    characteristics. This allows selective test execution using pytest's
    marker filtering.

    Args:
        config: Pytest configuration object.
    """
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests",
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests",
    )
