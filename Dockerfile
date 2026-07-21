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

# Copy the dependency manifests first so the sync layer is cached independently
# of source changes.
COPY pyproject.toml uv.lock ./

# Install dependencies exactly as locked; --frozen fails the build if uv.lock
# is out of date rather than silently re-resolving.
RUN uv sync --frozen --no-dev

# Copy the rest of the project (see .dockerignore for exclusions).
COPY . .

# Run as an unprivileged user rather than root.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Launch via the console script defined in pyproject.toml.
CMD ["uv", "run", "vfx-mcp"]
