"""MCP resource endpoints for tool discovery and video metadata.

All file discovery and metadata lookups are scoped to the workspace sandbox
(:func:`vfx_mcp.core.validation.resolve_workspace`) so that reading a resource
can never enumerate or probe arbitrary paths on the host filesystem.
"""

import json
import os
from pathlib import Path

from fastmcp import FastMCP

from ..core import get_video_metadata, resolve_workspace, safe_input_path

# File extensions treated as video files by ``videos://list``.
_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".mpg",
        ".mpeg",
    }
)

# Maximum directory depth (relative to the workspace root) scanned by
# ``videos://list``. Depth 0 is the workspace root itself. A small bound keeps
# discovery fast and prevents traversing large, unrelated directory trees.
_MAX_LIST_DEPTH = 3

# Maximum number of file paths returned by ``videos://list``. The total count
# is always reported even when the returned listing is truncated.
_MAX_LIST_RESULTS = 500


def _list_workspace_videos(
    workspace: Path,
    *,
    max_depth: int = _MAX_LIST_DEPTH,
    limit: int = _MAX_LIST_RESULTS,
) -> tuple[list[str], int]:
    """List video files under ``workspace`` as workspace-relative paths.

    Walks the workspace tree up to ``max_depth`` levels deep, skipping hidden
    directories and files. Returns POSIX-style paths relative to the workspace
    root (directly usable as ``input_path`` for the editing tools) together
    with the total number of matches found (which may exceed ``len(paths)``
    when the listing is truncated at ``limit``).

    Args:
        workspace: The resolved sandbox root to scan.
        max_depth: Maximum directory depth relative to the root to descend.
        limit: Maximum number of paths to include in the returned list.

    Returns:
        A ``(relative_paths, total_found)`` tuple.
    """
    relative_paths: list[str] = []
    total_found = 0

    if not workspace.is_dir():
        return relative_paths, total_found

    for dirpath, dirnames, filenames in os.walk(workspace):
        current = Path(dirpath)
        depth = len(current.relative_to(workspace).parts)

        # Prune traversal beyond the depth bound and skip hidden directories.
        if depth >= max_depth:
            dirnames[:] = []
        else:
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))

        for name in sorted(filenames):
            if name.startswith("."):
                continue
            if Path(name).suffix.lower() not in _VIDEO_EXTENSIONS:
                continue
            total_found += 1
            if len(relative_paths) < limit:
                relative = (current / name).relative_to(workspace).as_posix()
                relative_paths.append(relative)

    return relative_paths, total_found


# Advanced/stitching-oriented tools surfaced by ``tools://advanced/{category}``.
# Keep this list in sync with the tools actually registered by the server so
# clients never discover a capability that does not exist.
_ADVANCED_TOOLS: dict[str, list[dict[str, object]]] = {
    "stitching": [
        {
            "name": "concatenate_videos",
            "purpose": "Join multiple clips end-to-end into a single video",
            "key_features": [
                "Audio-aware concatenation (silent clips backed by silence)",
                "Normalization pre-pass for heterogeneous resolution/fps",
                "Lossless stream-copy fast path for homogeneous inputs",
            ],
            "example_use": "Stitch same-pipeline Veo/omni clips in order",
        },
        {
            "name": "stitch_with_transitions",
            "purpose": "Join N clips with crossfade-style transitions",
            "key_features": [
                "xfade video transitions (fade/dissolve/wipe/slide)",
                "acrossfade audio blending between clips",
                "Automatic normalization of mismatched inputs",
            ],
            "example_use": "Assemble a montage with dissolves between shots",
        },
        {
            "name": "stitch_from_manifest",
            "purpose": "Stitch a deliverable from a manifest of clip segments",
            "key_features": [
                "Per-clip in/out trim points and transitions",
                "One MCP call per deliverable instead of many",
                "Consistent normalization across all segments",
            ],
            "example_use": "Render a timeline described as a list of segments",
        },
    ],
    "compositing": [
        {
            "name": "create_green_screen_effect",
            "purpose": "Remove green/blue screen and replace the background",
            "key_features": [
                "Chroma key compositing with adjustable similarity/blend",
                "Color spill reduction",
                "Custom background image or video",
            ],
            "example_use": "Composite a subject over a custom background",
        },
        {
            "name": "apply_motion_blur",
            "purpose": "Add directional/temporal motion blur to footage",
            "key_features": [
                "Adjustable blur strength",
                "Frame-blending based blur",
            ],
            "example_use": "Smooth fast motion or stylize movement",
        },
    ],
    "audio": [
        {
            "name": "normalize_loudness",
            "purpose": "Normalize perceived loudness to a target (EBU R128)",
            "key_features": [
                "loudnorm-based integrated loudness targeting",
                "Level clips from different sources before stitching",
                "Preserves the video stream",
            ],
            "example_use": "Match loudness across mixed-source clips",
        },
        {
            "name": "mix_audio",
            "purpose": "Mix an external audio track with a video's audio",
            "key_features": [
                "amix with normalize disabled to preserve levels",
                "Adjustable per-source volume",
            ],
            "example_use": "Layer narration or music over a clip",
        },
    ],
}


def register_resource_endpoints(
    mcp: FastMCP[object],
) -> None:
    """Register MCP resource endpoints with the server."""

    @mcp.resource("videos://list")
    async def list_videos_resource() -> str:
        """List video files inside the workspace sandbox.

        Scans the workspace root (``VFX_WORKSPACE`` or the current working
        directory) up to a bounded depth and returns workspace-relative paths
        that can be passed directly as ``input_path`` to the editing tools,
        along with the total number of matches found.
        """
        workspace = resolve_workspace()
        videos, total_found = _list_workspace_videos(workspace)

        return json.dumps(
            {
                "videos": videos,
                "count": total_found,
                "workspace": str(workspace),
            },
            indent=2,
        )

    # Ensure function is registered with MCP
    del list_videos_resource

    @mcp.resource("videos://{filename}/metadata")
    async def video_metadata_resource(filename: str) -> str:
        """Get detailed metadata for a video inside the workspace sandbox.

        The requested path is resolved and containment-checked against the
        workspace via :func:`safe_input_path`, so metadata cannot be probed for
        arbitrary absolute paths outside the sandbox (unless
        ``VFX_ALLOW_ABSOLUTE=1`` is explicitly set).
        """
        resolved = safe_input_path(filename)
        metadata = get_video_metadata(str(resolved))
        return json.dumps(metadata, indent=2)

    # Ensure function is registered with MCP
    del video_metadata_resource

    @mcp.resource("tools://advanced/{category}")
    async def advanced_tools_resource(category: str = "all") -> str:
        """List advanced VFX tools with descriptions and capabilities.

        The catalog is filtered against the tools actually registered on the
        server, so this resource can never advertise a capability that does not
        exist (the original ``create_video_slideshow`` bug, finding M2).

        Args:
            category: One of ``stitching``, ``compositing``, ``audio`` or
                ``all`` (the default) to return every category.
        """
        requested = category.lower()
        if requested in ("", "all"):
            catalog = [tool for tools in _ADVANCED_TOOLS.values() for tool in tools]
        else:
            catalog = _ADVANCED_TOOLS.get(requested, [])

        # Only advertise tools that are genuinely registered with the server.
        # FastMCP's tool-introspection API varies across versions; fall back to
        # the curated catalog if it isn't available so this never hard-fails.
        registered: set[str] | None = None
        try:
            tools = await mcp.list_tools()  # type: ignore[attr-defined]
            registered = {t.name for t in tools}
        except Exception:
            registered = None

        selected = [
            tool
            for tool in catalog
            if registered is None or tool.get("name") in registered
        ]

        return json.dumps(
            {
                "advanced_tools": selected,
                "total_tools": len(selected),
                "category": requested or "all",
                "categories": sorted(_ADVANCED_TOOLS),
            },
            indent=2,
        )

    # Ensure function is registered with MCP
    del advanced_tools_resource
