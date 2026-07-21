"""Parameter validation functions for VFX operations."""

import os
import re
from pathlib import Path
from typing import overload

# ffmpeg accepts many protocol/URL inputs which turn an unvalidated ``input_path``
# into an SSRF / arbitrary-file-read primitive. These prefixes are always
# rejected for input paths.
_BLOCKED_INPUT_PROTOCOLS: tuple[str, ...] = (
    "http:",
    "https:",
    "concat:",
    "pipe:",
)

# Generic ``scheme://`` URL detector (catches ftp://, rtmp://, etc.).
_URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def resolve_workspace() -> Path:
    """Return the sandbox root directory for all file operations.

    Reads the ``VFX_WORKSPACE`` environment variable and falls back to the
    current working directory. The path is expanded and resolved so that
    containment checks compare fully-normalized paths.
    """
    root = os.environ.get("VFX_WORKSPACE")
    base = Path(root) if root else Path.cwd()
    return base.expanduser().resolve()


def _containment_enforced() -> bool:
    """Whether path containment is enforced.

    The sandbox is *opt-in*: containment is only enforced when ``VFX_WORKSPACE``
    is explicitly set (the intended posture for pipeline/server deployments).
    When it is unset — library, dev, and test usage — paths are resolved but not
    jailed. ``VFX_ALLOW_ABSOLUTE=1`` disables containment even when a workspace
    is set. Protocol/URL input rejection is always active regardless.
    """
    if os.environ.get("VFX_ALLOW_ABSOLUTE") == "1":
        return False
    return bool(os.environ.get("VFX_WORKSPACE"))


def _reject_protocol_input(path: str) -> None:
    """Raise ``ValueError`` if ``path`` looks like an ffmpeg protocol/URL input."""
    stripped = path.strip()
    lowered = stripped.lower()

    for proto in _BLOCKED_INPUT_PROTOCOLS:
        if lowered.startswith(proto):
            raise ValueError(f"Protocol/URL inputs are not allowed: {path}")

    if _URL_SCHEME_RE.match(stripped):
        raise ValueError(f"Protocol/URL inputs are not allowed: {path}")

    # lavfi is a virtual input device (e.g. testsrc, anullsrc) which is useful
    # internally but must be opt-in for caller-supplied paths.
    if lowered == "lavfi" or lowered.startswith("lavfi:"):
        if os.environ.get("VFX_ALLOW_LAVFI") != "1":
            raise ValueError(
                f"lavfi inputs are disabled (set VFX_ALLOW_LAVFI=1 to allow): {path}"
            )


def _resolve_within_workspace(path: str, workspace: Path) -> Path:
    """Resolve ``path`` against ``workspace`` and enforce containment.

    Absolute paths and ``..`` traversal that escape the workspace root are
    rejected unless ``VFX_ALLOW_ABSOLUTE=1`` is set.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (workspace / candidate).resolve()

    if not _containment_enforced():
        return resolved

    if resolved != workspace and not resolved.is_relative_to(workspace):
        raise ValueError(f"Path escapes the workspace sandbox ({workspace}): {path}")
    return resolved


def safe_input_path(path: str, *, workspace: Path | None = None) -> Path:
    """Resolve and validate an input file path within the workspace sandbox.

    Rejects protocol/URL inputs and path traversal outside the workspace, then
    verifies the resolved path exists and is a regular file.

    Args:
        path: Caller-supplied input path.
        workspace: Sandbox root; defaults to :func:`resolve_workspace`.

    Returns:
        The resolved, contained path.

    Raises:
        ValueError: On protocol inputs or workspace escape.
        FileNotFoundError: If the resolved path does not exist.
    """
    _reject_protocol_input(path)
    ws = workspace if workspace is not None else resolve_workspace()
    resolved = _resolve_within_workspace(path, ws)
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return resolved


@overload
def safe_output_path(path: str, *, workspace: Path | None = ...) -> Path: ...
@overload
def safe_output_path(
    path: list[str], *, workspace: Path | None = ...
) -> list[Path]: ...
def safe_output_path(
    path: str | list[str], *, workspace: Path | None = None
) -> Path | list[Path]:
    """Resolve and validate output path(s) within the workspace sandbox.

    Enforces the same containment rules as :func:`safe_input_path` (traversal
    outside the workspace rejected unless ``VFX_ALLOW_ABSOLUTE=1``) and creates
    parent directories for the destination(s). Accepts a single path or a list.

    Args:
        path: A single output path or a list of output paths.
        workspace: Sandbox root; defaults to :func:`resolve_workspace`.

    Returns:
        A resolved :class:`~pathlib.Path`, or a list of them when given a list.
    """
    ws = workspace if workspace is not None else resolve_workspace()
    if isinstance(path, list):
        return [_prepare_output(p, ws) for p in path]
    return _prepare_output(path, ws)


def _prepare_output(path: str, workspace: Path) -> Path:
    """Resolve a single output path and create its parent directories."""
    resolved = _resolve_within_workspace(path, workspace)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str,
) -> None:
    """Validate that a parameter is within the specified range."""
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}")


def validate_file_path(file_path: str) -> Path:
    """Validate that a file path exists and is readable."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    return path


def validate_output_path(
    output_path: str,
) -> Path:
    """Validate that an output path is writable."""
    path = Path(output_path)
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_video_paths(video_paths: list[str], min_count: int = 1) -> list[Path]:
    """Validate a list of video file paths."""
    if len(video_paths) < min_count:
        raise ValueError(f"At least {min_count} video file(s) required")

    validated_paths: list[Path] = []
    for path_str in video_paths:
        path = validate_file_path(path_str)
        validated_paths.append(path)

    return validated_paths


def validate_transition_type(
    transition_type: str,
) -> str:
    """Validate transition type parameter."""
    valid_transitions = [
        "fade",
        "wipe_left",
        "wipe_right",
        "wipe_up",
        "wipe_down",
        "slide_left",
        "slide_right",
        "dissolve",
        "crossfade",
    ]
    if transition_type not in valid_transitions:
        raise ValueError(
            f"Transition type must be one of: {', '.join(valid_transitions)}"
        )
    return transition_type


def validate_filter_name(filter_name: str) -> str:
    """Validate video filter name."""
    common_filters = [
        "blur",
        "sharpen",
        "brightness",
        "contrast",
        "saturation",
        "vintage",
        "sepia",
        "grayscale",
        "hflip",
    ]
    # Allow scale filters with parameters
    if filter_name.startswith("scale="):
        return filter_name
    if filter_name not in common_filters:
        raise ValueError(
            f"Filter must be one of: {', '.join(common_filters)} or scale=WIDTHxHEIGHT"
        )
    return filter_name


def validate_animation_type(
    animation_type: str,
) -> str:
    """Validate text animation type."""
    valid_animations = [
        "fade_in",
        "slide_in_left",
        "slide_in_right",
        "slide_in_top",
        "slide_in_bottom",
        "zoom_in",
        "rotate_in",
        "typewriter",
    ]
    if animation_type not in valid_animations:
        raise ValueError(
            f"Animation type must be one of: {', '.join(valid_animations)}"
        )
    return animation_type
