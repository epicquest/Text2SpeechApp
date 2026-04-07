"""
main.py — FastAPI application entry point.

Responsibilities
----------------
1. Lifespan management  — create directories, initialise DB, load TTS models,
                          start the inference worker task.
2. asyncio.Queue        — serialise GPU jobs so only 1 runs at a time.
3. ThreadPoolExecutor   — isolate blocking GPU inference from the event loop.
4. In-memory job store  — fast O(1) status lookups between DB writes.
5. REST endpoints       — POST /generate, GET /status/{job_id},
                          GET /history, DELETE /audio/{id}.

Concurrency Model
-----------------
  ┌──────────────────┐          ┌────────────────────────────┐
  │  HTTP Handlers   │ enqueue  │   inference_worker() task  │
  │  (async, event   │ ──────►  │   (asyncio, event loop)    │
  │   loop)          │          │                            │
  └──────────────────┘          │  await run_in_executor()   │
                                │    │                       │
                                │    ▼                       │
                                │  ThreadPoolExecutor        │
                                │  (max_workers=1)           │
                                │    │  TTSService.generate()│
                                │    │  (blocks GPU thread)  │
                                └────┴───────────────────────┘

The executor has exactly one worker thread, which guarantees that at most one
GPU inference call executes at any moment, without requiring Celery or Redis.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import AsyncSessionLocal, AudioGeneration, get_session, init_db
from app.tts_engine import TTSService

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.getLevelName(settings.LOG_LEVEL),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory job state
# ---------------------------------------------------------------------------


class JobState(TypedDict):
    """Ephemeral job record kept in process memory for fast status lookups."""

    status: Literal["pending", "processing", "done", "error"]
    file_path: str | None  # relative URL path returned to the frontend
    error: str | None


# Module-level store; mutated exclusively by inference_worker() and POST /generate.
job_store: dict[str, JobState] = {}


# ---------------------------------------------------------------------------
# Inference worker
# ---------------------------------------------------------------------------


async def inference_worker(
    queue: asyncio.Queue[tuple[str, str, str, str | None]],
    executor: ThreadPoolExecutor,
) -> None:
    """
    Consume jobs from ``queue`` one at a time and run GPU inference.

    Each queue item is a ``(job_id, text, model_name, voice_id)`` tuple.  The
    blocking ``TTSService.generate_with_voice()`` call is dispatched to the
    executor's single worker thread so the event loop remains responsive.

    DB writes happen in this coroutine (async context) after the executor
    returns — never inside the executor thread — to avoid thread-safety issues
    with asyncpg connections.
    """
    loop = asyncio.get_running_loop()
    logger.info("inference_worker: started")

    while True:
        job_id, text, model_name, voice_id = await queue.get()
        logger.info(
            "inference_worker: starting job %s (model=%s voice=%s)",
            job_id,
            model_name,
            voice_id,
        )

        # ---- Update state: pending → processing ----
        job_store[job_id]["status"] = "processing"
        async with AsyncSessionLocal() as session:
            record = await session.get(AudioGeneration, uuid.UUID(job_id))
            if record is not None:
                record.status = "processing"
                await session.commit()

        # ---- GPU inference (blocking — runs in the single executor thread) ----
        try:
            wav_path, _mp3_path = await loop.run_in_executor(
                executor,
                TTSService.get().generate_with_voice,
                text,
                model_name,
                job_id,
                voice_id,
            )

            # Derive a relative URL path for the frontend  (e.g. /static/audio/…)
            relative_path = f"/static/audio/{wav_path.name}"

            # ---- Update state: processing → done ----
            job_store[job_id]["status"] = "done"
            job_store[job_id]["file_path"] = relative_path

            async with AsyncSessionLocal() as session:
                record = await session.get(AudioGeneration, uuid.UUID(job_id))
                if record is not None:
                    record.status = "done"
                    record.file_path = str(wav_path)  # absolute path stored in DB
                    await session.commit()

            logger.info("inference_worker: job %s done → %s", job_id, wav_path)

        except Exception as exc:  # pylint: disable=broad-exception-caught
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.exception("inference_worker: job %s failed — %s", job_id, error_msg)

            job_store[job_id]["status"] = "error"
            job_store[job_id]["error"] = error_msg

            async with AsyncSessionLocal() as session:
                record = await session.get(AudioGeneration, uuid.UUID(job_id))
                if record is not None:
                    record.status = "error"
                    await session.commit()

        finally:
            queue.task_done()


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Manage application startup and shutdown.

    Startup sequence:
      1. Ensure audio output directory exists.
      2. Create DB tables (idempotent).
      3. Load TTS models into GPU memory.
      4. Start the inference worker task.

    Shutdown sequence:
      5. Cancel worker task gracefully.
      6. Shutdown the executor (waits for any running GPU job to finish).
    """
    # 1. Ensure audio directory exists
    output_dir = settings.AUDIO_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("lifespan: audio output dir → %s", output_dir)

    # 2. Initialise database
    await init_db()
    logger.info("lifespan: database tables ready")

    # 3. Load TTS models
    await TTSService.get().load_models()

    # 4. Create queue, executor, and worker task
    queue: asyncio.Queue[tuple[str, str, str, str | None]] = asyncio.Queue()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-worker")

    worker_task = asyncio.create_task(
        inference_worker(queue, executor),
        name="inference-worker",
    )

    # Attach to app.state so endpoints can access them
    _app.state.queue = queue
    _app.state.executor = executor

    logger.info("lifespan: application ready")
    yield  # ← server is running

    # 5. Cancel worker task
    logger.info("lifespan: shutting down inference worker")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

    # 6. Shutdown executor (drain any running GPU job)
    executor.shutdown(wait=True)
    logger.info("lifespan: shutdown complete")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Local TTS Service",
    description=(
        "GPU-accelerated text-to-speech API powered by Fish Speech v1.5 "
        "and Coqui XTTS v2.  No external APIs — fully local inference."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Derived paths for static file directories
VOICES_DIR: Path = settings.AUDIO_OUTPUT_DIR.parent / "voices"

# Ensure directories exist (StaticFiles raises RuntimeError if directory is missing)
settings.AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# Serve generated audio files at /static/audio/
app.mount(
    "/static/audio", StaticFiles(directory=str(settings.AUDIO_OUTPUT_DIR)), name="audio"
)
# Serve voice reference audio at /static/voices/
app.mount("/static/voices", StaticFiles(directory=str(VOICES_DIR)), name="voices")

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""

    text: str = Field(
        ..., min_length=1, max_length=4096, description="Text to synthesise"
    )
    model: str = Field(
        default="fish_speech",
        description='TTS backend: "fish_speech" | "xtts"',
    )
    voice: str | None = Field(
        default=None,
        description="Voice profile ID (stem of the .wav file in app/static/voices/)",
    )


class GenerateResponse(BaseModel):
    """Response body returned after queueing a generation job."""

    job_id: str


class StatusResponse(BaseModel):
    """Polling response for a generation job's current state."""

    status: Literal["pending", "processing", "done", "error"]
    file_path: str | None = None
    error: str | None = None


class AudioGenerationOut(BaseModel):
    """Read model for history endpoint — safe subset of AudioGeneration columns."""

    id: str
    input_text: str
    model_name: str
    voice_id: str | None
    file_path: str | None
    status: str
    created_at: str | None

    @classmethod
    def from_orm(cls, obj: AudioGeneration) -> AudioGenerationOut:
        return cls(**obj.to_dict())


class VoiceOut(BaseModel):
    """Voice profile summary returned by the /voices endpoint."""

    id: str
    name: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify_voice_name(name: str) -> str:
    """
    Convert a display name to a safe filename stem (alphanumeric + dash/underscore).
    """
    slug = re.sub(r"[^\w\-]", "_", name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:64]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a TTS generation job",
)
async def generate(
    req: GenerateRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> GenerateResponse:
    """
    Accept a text synthesis request and enqueue it for GPU processing.

    Returns immediately with a ``job_id``.  Poll ``GET /status/{job_id}``
    to track progress.
    """
    job_id = str(uuid.uuid4())

    # 1. Persist a "pending" record in the DB
    record = AudioGeneration(
        id=uuid.UUID(job_id),
        input_text=req.text,
        model_name=req.model,
        voice_id=req.voice,
        status="pending",
    )
    session.add(record)
    await session.commit()

    # 2. Initialise in-memory state
    job_store[job_id] = JobState(status="pending", file_path=None, error=None)

    # 3. Enqueue the job
    await app.state.queue.put((job_id, req.text, req.model, req.voice))
    logger.info(
        "POST /generate: enqueued job %s (model=%s voice=%s)",
        job_id,
        req.model,
        req.voice,
    )

    return GenerateResponse(job_id=job_id)


@app.get(
    "/status/{job_id}",
    response_model=StatusResponse,
    summary="Poll the status of a generation job",
)
async def get_status(
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> StatusResponse:
    """
    Return the current status of a queued or completed job.

    Status values: ``pending`` → ``processing`` → ``done`` | ``error``

    ``file_path`` is a relative URL (e.g. ``/static/audio/<uuid>.wav``) that
    the frontend can use to fetch or play the audio once status is ``done``.

    Falls back to a DB lookup if the job is not in the in-memory store (e.g.
    after a server restart).
    """
    # Fast path — in-memory lookup
    if job_id in job_store:
        state = job_store[job_id]
        return StatusResponse(
            status=state["status"],
            file_path=state["file_path"],
            error=state["error"],
        )

    # Slow path — DB lookup (covers server-restart scenario)
    try:
        uid = uuid.UUID(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid job_id format") from exc

    record = await session.get(AudioGeneration, uid)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")

    file_path: str | None = None
    if record.file_path and record.status == "done":
        file_path = f"/static/audio/{Path(record.file_path).name}"

    return StatusResponse(
        status=record.status,  # type: ignore[arg-type]
        file_path=file_path,
        error=None,
    )


@app.get(
    "/history",
    response_model=list[AudioGenerationOut],
    summary="List previous generation jobs",
)
async def get_history(
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: int = 100,
    offset: int = 0,
) -> list[AudioGenerationOut]:
    """
    Return up to ``limit`` previous generation jobs, newest first.

    Supports basic pagination via ``offset`` and ``limit`` query parameters.
    """
    result = await session.execute(
        select(AudioGeneration)
        .order_by(AudioGeneration.created_at.desc())
        .limit(min(limit, 500))  # hard cap to prevent runaway queries
        .offset(offset)
    )
    records = result.scalars().all()
    return [AudioGenerationOut.from_orm(r) for r in records]


@app.delete(
    "/audio/{audio_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an audio record and its files",
)
async def delete_audio(
    audio_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> None:
    """
    Delete the DB record **and** the associated audio files from disk.

    Both the ``.wav`` and the ``.mp3`` side-car are removed if they exist.
    Returns 404 if the record is not found; always returns 204 on success.
    """
    try:
        uid = uuid.UUID(audio_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid audio_id format") from exc

    record = await session.get(AudioGeneration, uid)
    if record is None:
        raise HTTPException(status_code=404, detail="Audio record not found")

    # ---- Remove files from disk ----
    if record.file_path:
        wav_path = Path(record.file_path)
        mp3_path = wav_path.with_suffix(".mp3")

        if wav_path.is_file():
            wav_path.unlink()
            logger.info("DELETE /audio/%s: removed %s", audio_id, wav_path)

        if mp3_path.is_file():
            mp3_path.unlink()
            logger.info("DELETE /audio/%s: removed %s", audio_id, mp3_path)

    # ---- Remove from in-memory store ----
    job_store.pop(audio_id, None)

    # ---- Remove DB record ----
    await session.delete(record)
    await session.commit()
    logger.info("DELETE /audio/%s: DB record deleted", audio_id)


# ---------------------------------------------------------------------------
# Voice profile endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/voices",
    response_model=list[VoiceOut],
    summary="List available voice profiles",
)
async def list_voices() -> list[VoiceOut]:
    """Return all voice profiles stored in the voices directory."""
    voices = []
    for wav in sorted(VOICES_DIR.glob("*.wav")):
        display = wav.stem.replace("_", " ").replace("-", " ").title()
        voices.append(VoiceOut(id=wav.stem, name=display))
    return voices


@app.post(
    "/voices",
    response_model=VoiceOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a new voice profile",
)
async def create_voice(
    name: str = Form(..., min_length=1, max_length=100),
    transcript: str = Form(
        default="", description="Text spoken in the reference audio"
    ),
    audio: UploadFile = File(..., description="Reference WAV audio file"),
) -> VoiceOut:
    """
    Upload a WAV reference audio file to create a voice profile.

    The ``name`` becomes the display label; the URL-safe slugified form is
    used as the ``id``.  The optional ``transcript`` (what is spoken in the
    audio) improves voice-cloning accuracy for Fish Speech.
    """
    voice_id = _slugify_voice_name(name)
    if not voice_id:
        raise HTTPException(status_code=422, detail="Voice name produced an empty slug")

    # Accept only WAV files
    is_wav_content = (audio.content_type or "").lower() in (
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
    )
    is_wav_name = (audio.filename or "").lower().endswith(".wav")
    if not (is_wav_content or is_wav_name):
        raise HTTPException(
            status_code=422,
            detail="audio must be a WAV file (.wav)",
        )

    content = await audio.read()
    (VOICES_DIR / f"{voice_id}.wav").write_bytes(content)
    (VOICES_DIR / f"{voice_id}.txt").write_text(transcript, encoding="utf-8")

    logger.info("POST /voices: created voice %r (%d bytes)", voice_id, len(content))
    return VoiceOut(id=voice_id, name=name)


@app.delete(
    "/voices/{voice_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a voice profile",
)
async def delete_voice(voice_id: str) -> None:
    """Remove the WAV and transcript files for a voice profile."""
    if not re.fullmatch(r"[\w\-]{1,64}", voice_id):
        raise HTTPException(status_code=422, detail="Invalid voice_id")

    wav_path = VOICES_DIR / f"{voice_id}.wav"
    if not wav_path.is_file():
        raise HTTPException(status_code=404, detail="Voice not found")

    wav_path.unlink()
    txt_path = VOICES_DIR / f"{voice_id}.txt"
    txt_path.unlink(missing_ok=True)
    logger.info("DELETE /voices/%s: removed", voice_id)


# ---------------------------------------------------------------------------
# Health check (optional but useful for load-balancers / Docker healthchecks)
# ---------------------------------------------------------------------------


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, Any]:
    """Lightweight liveness probe — does not touch the DB or GPU."""
    return {
        "status": "ok",
        "queue_depth": app.state.queue.qsize() if hasattr(app.state, "queue") else -1,
    }
