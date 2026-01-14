FROM python:3.12-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install uv
WORKDIR /app
COPY . .
RUN uv sync
CMD ["uv", "run", "python", "main.py"]
