# Code Review: vfx-mcp — Readiness for Omni/Veo Stitching Pipelines

Reviewed: 2026-07-20 (branch `main`, commit `8183b23`).
Scope: full-repo review focused on using this MCP server to combine and stitch
videos in automated omni/veo pipelines. Findings verified by compiling the
actual ffmpeg command lines produced by the code (ffmpeg-python `get_args`).

**Bottom line: do not integrate as-is.** The one tool the pipeline depends on
most — `concatenate_videos` — does not concatenate. Several audio tools produce
wrong output, the PyPI entry point crashes, and CI never runs the test suite,
which is how the broken concat merged. All of this is fixable; the core is a
thin (~2,100 line) wrapper over ffmpeg-python, and Phase 1 below is roughly a
day or two of work.

---

## Critical findings (pipeline blockers)

### C1. `concatenate_videos` does not concatenate — CONFIRMED
`src/vfx_mcp/tools/basic_video_ops.py:240`

```python
inputs = [ffmpeg.input(path) for path in input_paths]
stream = ffmpeg.concat(*inputs, v=1, a=1)
```

`ffmpeg.concat` with `v=1, a=1` expects **2 streams per input** (video + audio),
in order. Passing whole input nodes instead of split streams means
ffmpeg-python computes `n = len(inputs) / 2`. Verified compiled output:

- **2 inputs** → `[0][1]concat=a=1:n=1:v=1` — that is *one* segment whose video
  comes from file A and whose audio comes from file B. The "concatenated"
  output is clip A's video with clip B's soundtrack, at clip A's duration.
- **3+ inputs** → `ValueError: Expected concat input streams to have length
  multiple of 2 (v=1, a=1); got 3` — the tool throws before ffmpeg even runs.

Introduced by fork commit `ce5a68e` ("Fix audio output in video concatenate").
The upstream code (`ffmpeg.concat(*inputs)`, defaults `v=1, a=0`) concatenated
video correctly but dropped audio — the "fix" replaced a partial behavior with
a fully broken one. Correct shape:

```python
streams = []
for i in inputs:
    streams += [i.video, i.audio]   # or i["a"] guarded by a probe
stream = ffmpeg.concat(*streams, v=1, a=1)
```

### C2. Concat has no handling for silent or heterogeneous inputs
`basic_video_ops.py:206-246`

Directly relevant to Veo/omni output:

- **Silent clips**: Veo 2 output (and Veo 3 with audio disabled) has no audio
  stream. Even a correctly-written `a=1` concat fails on any silent input, and
  mixing silent + sound clips fails. Needs a probe pass that injects
  `anullsrc` for silent inputs (or falls back to video-only concat when *no*
  input has audio).
- **Heterogeneous clips**: the ffmpeg `concat` filter requires identical
  resolution, and mismatched fps/SAR/pix_fmt produce broken timing. The
  docstring claims "Videos with different properties will be automatically
  converted" — nothing in the code does this. Needs a normalization pre-pass
  (scale/pad + fps + SAR + pix_fmt).
- **No lossless fast path**: clips from the same Veo pipeline are homogeneous
  (same codec/resolution/fps). Those should be stitched with the concat
  *demuxer* + `-c copy` — near-instant and zero generational quality loss —
  instead of always re-encoding through libx264 at default CRF 23.

### C3. PyPI console script crashes on launch
`pyproject.toml:37` declares `vfx-mcp = "vfx_mcp.core.server:main"`, but
`src/vfx_mcp/core/server.py` defines no `main` — only `create_mcp_server`.
`pip install vfx-mcp && vfx-mcp` dies with an import error. Also: README and
CLAUDE.md document `MCP_TRANSPORT`/`MCP_HOST`/`MCP_PORT` env vars, but no code
reads them — the server is stdio-only via `main.py`, which itself relies on a
`sys.path` hack (`main.py:18`).

### C4. `add_audio` "replace" mode doesn't replace — CONFIRMED
`src/vfx_mcp/tools/audio_processing.py:215-222`

Compiled command: `-i video.mp4 -i audio.mp3 -map 0 -map 1 -acodec aac -shortest -vcodec copy`.
`-map 0` maps *all* streams of the video, including its original audio, so the
output has **two audio tracks** with the original first — most players play the
original audio, making "replace" a no-op. Mix mode (`:238`) has the same flaw:
`-map 0 -map [s0]` emits original + mixed tracks. Replace needs `-map 0:v -map 1:a`;
mix needs `-map 0:v -map [mixed]`.

### C5. Audio fade/volume tools silently discard the video stream — CONFIRMED
`audio_processing.py:285-288, 388-391, 430-433`

`adjust_audio_volume`, `audio_fade_in`, `audio_fade_out` are documented for
"audio/video" files, but when given a video the compiled command is
`-filter_complex [0]afade=...[s0] -map [s0] out.mp4` — only the filtered audio
is mapped. The output "video" has no video stream. If used to fade a stitched
video's audio, the pipeline gets an audio-only .mp4.

---

## High-severity findings

### H1. The test suite cannot pass, and CI never runs it
- `tests/conftest.py:104-122` — the `sample_videos` fixture generates clips
  **without audio**; `tests/test_basic_operations.py:222` concatenates all
  three. With current code this raises the C1 `ValueError`; even with C1 fixed,
  `a=1` fails on the silent fixtures. So the primary concat test is red.
- `.github/workflows/ci.yaml` runs only on `pull_request` and delegates to an
  external reusable Nix workflow (`conneroisu/ci`); nothing in `flake.nix`
  wires pytest into `nix flake check`. The CD workflow (`cd.yaml`) publishes to
  PyPI with **no test gate**. This is how a broken concat merged through two PRs.

### H2. No path validation or sandboxing on any tool
`src/vfx_mcp/core/validation.py` contains `validate_file_path` /
`validate_video_paths` / `validate_output_path` — **none are called by any
tool**. Every tool accepts arbitrary filesystem paths and runs with
`overwrite_output=True`, so any MCP client (i.e., any LLM in the loop) can
overwrite any file the process can write. ffmpeg also accepts URLs and
device/protocol inputs (`lavfi`, `concat:`, `http:`), so unvalidated
`input_path` is an SSRF/local-file-read primitive if the server is ever
exposed beyond a trusted desktop. A pipeline deployment needs a workspace-root
sandbox with containment checks.

### H3. Synchronous `ffmpeg.run()` inside async tools; no timeouts or limits
Every tool calls blocking `ffmpeg.run()` from an `async def`, freezing the
FastMCP event loop for the duration of the encode. Long stitches make the
server unresponsive (no progress, no heartbeats, no concurrent calls). There
are no timeouts and no concurrency caps, so a single bad request (e.g.,
10-minute 4K re-encode) stalls everything. Use `asyncio.to_thread` (or
`run_async` + polling), add per-call timeouts and a semaphore, and surface
real progress via `ctx.report_progress` by parsing `-progress` output.

### H4. Multiple tools crash on silent (audio-less) video
- `change_speed` (`video_effects.py:255`) unconditionally maps `stream["a"]`.
- `extract_audio` (`audio_processing.py:114`) same.
Veo-2-style silent clips will fail these tools with an opaque ffmpeg mapping
error.

### H5. `resize_video` produces odd dimensions that libx264 rejects
`basic_video_ops.py:169-187` uses `scale=W:-1` / `iw*scale`. Any odd result
(e.g., width 500 from 1920×1080 → height 281) fails with yuv420p ("height not
divisible by 2"). Use `-2` for the auto axis and round the scaled axis to even.

### H6. `trim_video` is keyframe-inaccurate
`basic_video_ops.py:78-89` uses input-side seek + `c="copy"`: cuts snap to
keyframes, so trims can start with frozen/black frames or be off by up to a
GOP (~2s on typical web encodes). Fine as a fast path, wrong as the only path —
stitching workflows need an optional frame-accurate (re-encode) mode. Also,
`if duration:` treats `duration=0` as "to end", and there's no check that
`start_time` is inside the file.

### H7. `videos://list` resource recursively scans the user's home directories
`src/vfx_mcp/resources/mcp_endpoints.py:31-45` does `rglob("*")` over `cwd`,
`~/Videos`, `~/Movies`, and `~/Desktop` on every read — potentially minutes on
large home dirs, a privacy leak in shared deployments, and it returns bare
filenames without paths, so results can't even be fed back into other tools.
`videos://{filename}/metadata` happily probes any absolute path.

---

## Medium-severity findings

- **M1. No quality controls.** `create_standard_output`
  (`core/utilities.py:120`) hardcodes libx264/aac/yuv420p with no
  CRF/preset/bitrate parameters and no way to preserve the source codec.
  Every operation is a lossy default-CRF-23 re-encode — generational loss on
  every pipeline step. No `-movflags +faststart` for mp4 delivery either.
- **M2. Advertised features are empty stubs.** `video_transitions.py`,
  `batch_automation.py`, `video_analysis.py`, `text_animation.py` are
  `pass`-body TODOs, yet README documents a `transition` parameter on
  `concatenate_videos` (README.md:137) that doesn't exist, and the
  `tools://advanced` resource (`mcp_endpoints.py:70`) advertises
  `create_video_slideshow` etc. If you want stitching *with transitions*
  (xfade), you'd be building it, not configuring it.
- **M3. Dockerfile issues.** Base image `python:3.12-slim` vs
  `requires-python = ">=3.13"` (works only because uv downloads a 3.13
  toolchain at build time); `COPY . .` with no `.dockerignore` (ships `.git`,
  docs, caches); runs as root; `uv sync` without `--frozen`.
- **M4. `convert_format` silently overrides caller's codec choices** whenever
  `format` is passed (`format_conversion.py:87-100`) — passing
  `format="mp4", video_codec="libx265"` gets you libx264.
- **M5. `add_audio`/`mix_audio` loudness behavior.** `amix` halves input
  volumes by default (no `normalize=0`), so mixed narration/music comes out
  quiet; no loudness normalization (`loudnorm`) anywhere, which matters when
  stitching clips from different sources.
- **M6. Green-screen "transparent background" is impossible as coded**
  (`advanced_compositing.py:106-116`): output is yuv420p H.264, which has no
  alpha channel. `image_to_video` also fails on odd-dimension images (same
  even-dimension issue as H5).
- **M7. Stale/incorrect docs.** CLAUDE.md describes a "single-file
  architecture" in `main.py` (code moved to `src/vfx_mcp/` long ago), lists
  tools that don't exist, and documents env vars that aren't read.
  `pyproject.toml` says 0.1.0; README/docs say v0.1.1. Note this repo is a
  fork (CxOAGI) of conneroisu/vfx-mcp — the PyPI package is upstream's, so
  `pip install vfx-mcp` does **not** get your fork's fixes.
- **M8. Error hygiene.** `handle_ffmpeg_error` returns full raw ffmpeg stderr
  in the RuntimeError — huge payloads back through MCP; should truncate.
  The post-call `raise` statements after it are dead code kept for type
  checkers (harmless, but the pattern plus `del tool_name` / `_ = (...)`
  lint-appeasement is worth cleaning).

---

## Remediation plan (no implementation yet)

### Phase 0 — Decision & setup (½ day)
1. Decide integration mode: run as MCP server (stdio in a container) vs.
   import `vfx_mcp` as a library from the pipeline. Either way, **pin your
   fork** — do not depend on the upstream PyPI package (M7).
2. Add pytest to CI on push/PR (plain GitHub Actions job with apt ffmpeg is
   enough; don't wait on the Nix reusable workflow) and gate `cd.yaml` on it.
   Land this first so every following fix is provable. (H1)

### Phase 1 — Correctness blockers (1–2 days)
3. Rewrite `concatenate_videos` (C1, C2):
   - probe each input; split into `i.video` / `i.audio`, injecting `anullsrc`
     for silent clips (or video-only concat when none have audio);
   - normalization pre-pass (scale/pad to target resolution, fps, SAR,
     pix_fmt) when inputs are heterogeneous;
   - fast path: concat demuxer + `-c copy` when probes show homogeneous
     codec/resolution/fps/pix_fmt — the common case for same-pipeline Veo clips;
   - tests: 2 and 3+ clips, with-audio, silent, mixed, mismatched resolution;
     assert output *duration*, *stream counts*, and (for a marker test) that
     frames from each source appear.
4. Fix `add_audio` mapping (`-map 0:v -map 1:a` for replace; `-map 0:v -map
   [mixed]` for mix; `amix normalize=0` + volume semantics documented). (C4, M5)
5. Fix fade/volume tools to preserve video (`-map 0:v -c:v copy` + filtered
   audio). (C5)
6. Add a `main()` to `core/server.py` (reading `MCP_TRANSPORT`/host/port),
   point the console script at it, drop the `sys.path` hack in `main.py`. (C3)
7. Guard `stream["a"]` uses behind a probe in `change_speed`/`extract_audio`. (H4)
8. `resize_video`: use `-2` auto-axis / force-even scaling. (H5)
9. `trim_video`: add `accurate: bool = False` re-encode mode; validate
   `start_time`/`duration`. (H6)

### Phase 2 — Pipeline hardening (2–3 days)
10. Workspace sandbox: `VFX_WORKSPACE` root, resolve + containment-check every
    input/output path (actually use `validation.py`), reject URLs/protocol
    inputs unless explicitly enabled. (H2)
11. Async execution: wrap ffmpeg in `asyncio.to_thread` with per-call timeout,
    global concurrency semaphore, and `ctx.report_progress` wired to ffmpeg
    `-progress` output. (H3)
12. Quality controls: add `crf`/`preset`/`copy_codec` options to
    `create_standard_output` and expose them on concat/trim/convert; add
    `-movflags +faststart`. Fix `convert_format` codec-override precedence. (M1, M4)
13. Replace the home-directory scan in `videos://list` with a workspace-scoped
    listing returning relative paths. (H7)

### Phase 3 — Features your pipeline actually wants (2–4 days, optional)
14. `stitch_with_transitions`: xfade/acrossfade-based N-clip stitcher
    (fade/dissolve/wipe), the thing README already advertises. (M2)
15. Loudness normalization tool (`loudnorm`) for mixed-source audio.
16. Batch/manifest stitching (list of {clip, in, out, transition}) so the
    pipeline makes one MCP call per deliverable instead of N.

### Phase 4 — Docs & packaging cleanup (½ day)
17. Rewrite CLAUDE.md/README to match reality (modular layout, real tool list,
    real env vars); align version; fix Dockerfile (3.13 base, `.dockerignore`,
    non-root, `uv sync --frozen`, CMD via console script). (M3, M7)

### Suggested acceptance gate for pipeline adoption
- Stitch 3 heterogeneous clips (one silent) → correct order, correct total
  duration ±0.1s, single audio track, no re-encode when homogeneous.
- Kill/timeout behavior verified on a deliberately long encode.
- Path escape attempts (`../`, absolute, `http://`) rejected.
