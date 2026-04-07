# Local TTS App

A fully local, GPU-accelerated text-to-speech web application. No external APIs — all inference runs on your own hardware.

**Backends**
- **Fish Speech v1.5** — high-quality neural TTS with optional voice cloning via reference audio
- **Coqui XTTS v2** — zero-shot voice cloning from a reference speaker WAV

**Stack:** FastAPI · SQLAlchemy (async PostgreSQL) · asyncio queue · ThreadPoolExecutor · vanilla-JS frontend

---

## Features

- Submit text and get back WAV + MP3 audio
- Upload voice profiles (reference WAV + transcript) for Fish Speech voice cloning
- Job queue — one GPU inference at a time, non-blocking HTTP
- Generation history with pagination
- Per-job status polling (`pending → processing → done | error`)
- Lazy model loading to conserve VRAM when both backends are enabled

---

## Requirements

| Component | Version |
|-----------|---------|
| Python | 3.12+ |
| PyTorch | 2.3.0 (CUDA 12.1) |
| PostgreSQL | 14+ |
| ffmpeg | any recent (for MP3 conversion) |
| CUDA | 12.x (CPU and MPS also supported) |
| VRAM | ≥ 8 GB recommended (RTX 3060 / 4060 or better) |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/tts-app.git
cd tts-app
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.3.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

> For CPU-only: `pip install torch==2.3.0 torchaudio==2.3.0`  
> For CUDA 11.8: replace `cu121` with `cu118`

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Fish Speech from source

```bash
pip install "fish-speech @ git+https://github.com/fishaudio/fish-speech.git@v1.5.0"
```

### 6. Download Fish Speech checkpoints

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="fishaudio/fish-speech-1.5",
    local_dir="checkpoints/fish-speech-1.5",
)
EOF
```

The checkpoint directory should contain at minimum:
```
checkpoints/fish-speech-1.5/
  config.json
  model.pth
  firefly-gan-vq-fsq-8x1024-21hz-generator.pth
```

### 7. Set up PostgreSQL

```sql
CREATE USER tts_user WITH PASSWORD 'tts_password';
CREATE DATABASE tts_db OWNER tts_user;
```

### 8. Configure environment

Create a `.env` file in the project root:

```dotenv
DATABASE_URL=postgresql+asyncpg://tts_user:tts_password@localhost:5432/tts_db

# Optional overrides (defaults shown)
FISH_SPEECH_CKPT=checkpoints/fish-speech-1.5
DEVICE=cuda
AUDIO_OUTPUT_DIR=app/static/audio
LOG_LEVEL=INFO

# Lazy loading — set to true to defer model loading to first request
LAZY_FISH_SPEECH=false
LAZY_XTTS=true

# XTTS-specific (only needed if using the XTTS backend)
XTTS_SPEAKER_WAV=app/static/reference_speaker.wav
```

### 9. (Optional) Install ffmpeg

Required for WAV → MP3 conversion. Without it, only WAV output is available.

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

---

## Running

### Start the API server

```bash
./start-server.sh
# default: http://0.0.0.0:8000
# logs → logs/server.log
```

Options:
```bash
./start-server.sh --host 127.0.0.1 --port 8080
```

### Start the frontend

```bash
./start-frontend.sh
# default: http://127.0.0.1:3000
# logs → logs/frontend.log
```

Options:
```bash
./start-frontend.sh --port 4000
```

### Stop services

```bash
./kill-server.sh
./kill-frontend.sh
```

### Run manually (development)

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --reload
```

> `--workers 1` is required — the GPU queue is in-process and not safe for multiple workers.

---

## API Reference

All endpoints are documented interactively at **`http://localhost:8000/docs`** (Swagger UI).

### POST `/generate`

Submit a TTS job. Returns immediately with a `job_id`.

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "model": "fish_speech", "voice": null}'
```

```json
{"job_id": "3f2a1b4c-..."}
```

**Body fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Text to synthesise (1–4096 chars) |
| `model` | string | `"fish_speech"` | `"fish_speech"` or `"xtts"` |
| `voice` | string \| null | `null` | Voice profile ID (stem of an uploaded WAV) |

### GET `/status/{job_id}`

Poll for job completion.

```bash
curl http://localhost:8000/status/3f2a1b4c-...
```

```json
{
  "status": "done",
  "file_path": "/static/audio/3f2a1b4c-....wav",
  "error": null
}
```

`status` values: `pending` → `processing` → `done` | `error`

### GET `/history`

List recent generation jobs (newest first).

```bash
curl "http://localhost:8000/history?limit=20&offset=0"
```

### DELETE `/audio/{id}`

Delete a job record and its audio files from disk.

```bash
curl -X DELETE http://localhost:8000/audio/3f2a1b4c-...
```

### GET `/voices`

List available voice profiles.

### POST `/voices`

Upload a new voice profile (multipart form).

```bash
curl -X POST http://localhost:8000/voices \
  -F "name=My Voice" \
  -F "transcript=Text spoken in the reference audio" \
  -F "audio=@/path/to/reference.wav"
```

### DELETE `/voices/{voice_id}`

Delete a voice profile.

### GET `/health`

Liveness probe — returns `{"status": "ok", "queue_depth": N}`.

---

## Voice Profiles (Fish Speech)

Voice profiles are WAV + transcript pairs stored under `app/static/voices/`:

```
app/static/voices/
  my_voice.wav   ← reference audio (~5–30 seconds of clean speech)
  my_voice.txt   ← transcript of what is spoken in the WAV
```

Upload them via the web UI or the `POST /voices` API endpoint. The transcript is optional but significantly improves cloning accuracy.

---

## Project Structure

```
tts-app/
├── app/
│   ├── domain/               # Pure Python entities + ABC interfaces (no framework deps)
│   │   ├── entities.py       # GenerationJob, VoiceProfile dataclasses
│   │   └── interfaces.py     # JobRepository, TTSPort, VoiceRepository, … ABCs
│   ├── use_cases/            # Application business rules
│   │   ├── audio.py          # Enqueue, GetStatus, ListHistory, DeleteAudio
│   │   └── voices.py         # ListVoices, CreateVoice, DeleteVoice
│   ├── infrastructure/       # Concrete adapters (DB, TTS engines, filesystem)
│   │   ├── database.py       # SQLAlchemy ORM + SQLAlchemyJobRepository
│   │   ├── tts_engine.py     # FishSpeechBackend, XTTSBackend, TTSService
│   │   ├── voice_storage.py  # FilesystemVoiceRepository
│   │   ├── job_cache.py      # InMemoryJobStatusCache, LocalFileStorage
│   │   └── worker.py         # asyncio inference queue consumer
│   ├── api/                  # HTTP layer (FastAPI router, schemas, dependencies)
│   │   ├── router.py         # Endpoint handlers
│   │   ├── schemas.py        # Pydantic DTOs
│   │   └── dependencies.py   # FastAPI Depends() wiring
│   ├── config.py             # Pydantic Settings (env / .env)
│   └── main.py               # Composition root + FastAPI app factory
├── frontend/
│   └── index.html            # Single-page UI (vanilla JS, no build step)
├── checkpoints/              # Model weights (not committed)
├── logs/                     # Server + frontend logs (not committed)
├── requirements.txt
├── start-server.sh
├── start-frontend.sh
├── kill-server.sh
├── kill-frontend.sh
└── run_linters.sh
```

---

## Concurrency Model

```
HTTP handler (async)  ──enqueue──►  asyncio.Queue
                                         │
                                    inference_worker (async task)
                                         │
                              loop.run_in_executor()
                                         │
                                  ThreadPoolExecutor
                                   (max_workers=1)
                                         │
                                  TTSService.synthesise()
                                  (blocks GPU thread)
```

- Exactly one GPU inference runs at a time — no Celery, no Redis
- HTTP handlers stay non-blocking; clients poll `/status/{job_id}`
- DB writes happen in the asyncio event loop after the executor returns (thread-safe)

---

## Development

### Linting

```bash
./run_linters.sh
# runs: black · isort · flake8 · ruff · pylint · mypy
```

### Individual tools

```bash
source .venv/bin/activate
black app/
isort app/
ruff check app/
pylint app
mypy app
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | **Required.** asyncpg DSN: `postgresql+asyncpg://user:pass@host:5432/db` |
| `FISH_SPEECH_CKPT` | `/models/fish-speech-1.5` | Path to Fish Speech checkpoint directory |
| `XTTS_MODEL_ID` | `tts_models/multilingual/multi-dataset/xtts_v2` | Coqui TTS model identifier |
| `XTTS_SPEAKER_WAV` | `/app/static/reference_speaker.wav` | Reference WAV for XTTS zero-shot cloning |
| `AUDIO_OUTPUT_DIR` | `/app/static/audio` | Directory for generated audio files |
| `DEVICE` | `cuda` | PyTorch device: `cuda`, `cpu`, or `mps` |
| `LAZY_FISH_SPEECH` | `false` | Defer Fish Speech loading to first request |
| `LAZY_XTTS` | `false` | Defer XTTS loading to first request |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## License

MIT
