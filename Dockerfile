# cxoagi-vfx-mcp container image
# The project requires Python >=3.13 (see pyproject.toml), so the base image
# must ship a matching interpreter.
FROM python:3.13-slim

# ffmpeg is a hard runtime dependency for every tool in this server.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv drives dependency resolution and the console-script entry point.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy the dependency manifests first so the dependency layer is cached
# independently of source changes.
COPY pyproject.toml uv.lock ./

# Install ONLY the locked dependencies here (not the project itself), so this
# layer stays cached across source edits. --no-install-project skips building
# cxoagi-vfx-mcp, which would otherwise need README.md/src (not yet copied).
# --frozen fails the build if uv.lock is out of date rather than re-resolving.
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the project (see .dockerignore for exclusions; README.md is
# force-included because pyproject references it).
COPY . .

# Now build + install the project itself against the full source tree.
RUN uv sync --frozen --no-dev

# Run as an unprivileged user rather than root.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Launch the installed console script directly (no runtime re-sync).
CMD ["uv", "run", "--no-sync", "vfx-mcp"]
