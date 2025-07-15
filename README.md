## Tech Stack

| Area           | Tool / Service        |
|----------------|-----------------------|
| Web framework  | Django                |
| Background     | Celery + Redis        |
| Computer vision| OpenCV, YOLOv8        |
| OCR            | EasyOCR               |
| Embeddings     | CLIP, OpenAI          |
| Vector DB      | ChromaDB              |
| Streaming      | mediamtx (RTSP)       |
| Packaging      | uv + pyproject.toml   |

## Quick start (local)

```bash
# 1. Install Python deps
uv sync                      # installs locked deps from pyproject/uv.lock

# 2. Bring up Redis broker (background)
docker compose up -d redis

# 3. Start RTSP broker and demo stream (two terminals)
mediamtx mediamtx.yaml       # RTSP server
ffmpeg -re -stream_loop -1 -i sample.mp4 \
      -c:v libx264 -preset veryfast -tune zerolatency -g 30 -keyint_min 30 \
      -b:v 2.5M -f rtsp rtsp://localhost:8554/overwatch

# 4. Django database & server
uv run python manage.py migrate       # apply migrations
uv run python manage.py runserver     # web UI on http://127.0.0.1:8000

# 5. Background worker (new terminal)
uv run celery -A overwatch worker -P solo --loglevel=info
```

VS Code users can launch steps 3 & 5 via **Tasks → Run Stream** and **Tasks → Run Celery Worker** (see `.vscode/tasks.json`).

Create cameras at `/cameras/new`, set the RTSP URL above, and frames will start processing automatically.
