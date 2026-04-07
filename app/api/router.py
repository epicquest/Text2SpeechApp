"""
api/router.py — FastAPI router: thin HTTP adapters over the use-case layer.

Each handler:
  1. Receives HTTP input (path params, query params, request body).
  2. Constructs the appropriate use case with injected dependencies.
  3. Calls the use case.
  4. Maps the domain result to a Pydantic response schema.

No business logic lives here.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import (
    get_cache,
    get_file_storage,
    get_job_repo,
    get_queue,
    get_voice_repo,
)
from app.api.schemas import (
    AudioGenerationOut,
    GenerateRequest,
    GenerateResponse,
    StatusResponse,
    VoiceOut,
)
from app.domain.interfaces import (
    FileStorage,
    JobRepository,
    JobStatusCache,
    VoiceRepository,
)
from app.use_cases.audio import (
    DeleteAudioUseCase,
    EnqueueJobUseCase,
    GetJobStatusUseCase,
    ListHistoryUseCase,
)
from app.use_cases.voices import (
    CreateVoiceUseCase,
    DeleteVoiceUseCase,
    ListVoicesUseCase,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify_voice_name(name: str) -> str:
    """Convert a display name to a safe filename stem (alphanumeric + dash/underscore)."""
    slug = re.sub(r"[^\w\-]", "_", name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:64]


# ---------------------------------------------------------------------------
# Audio generation endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a TTS generation job",
)
async def generate(
    req: GenerateRequest,
    repo: Annotated[JobRepository, Depends(get_job_repo)],
    cache: Annotated[JobStatusCache, Depends(get_cache)],
    queue: Annotated[
        asyncio.Queue[tuple[str, str, str, str | None]], Depends(get_queue)
    ],
) -> GenerateResponse:
    """
    Accept a text synthesis request and enqueue it for GPU processing.

    Returns immediately with a ``job_id``.  Poll ``GET /status/{job_id}``
    to track progress.
    """
    use_case = EnqueueJobUseCase(repo, cache, queue)
    job_id = await use_case.execute(req.text, req.model, req.voice)
    logger.info(
        "POST /generate: enqueued job %s (model=%s voice=%s)",
        job_id,
        req.model,
        req.voice,
    )
    return GenerateResponse(job_id=job_id)


@router.get(
    "/status/{job_id}",
    response_model=StatusResponse,
    summary="Poll the status of a generation job",
)
async def get_status(
    job_id: str,
    repo: Annotated[JobRepository, Depends(get_job_repo)],
    cache: Annotated[JobStatusCache, Depends(get_cache)],
) -> StatusResponse:
    """
    Return the current status of a queued or completed job.

    Status values: ``pending`` → ``processing`` → ``done`` | ``error``

    ``file_path`` is a relative URL (e.g. ``/static/audio/<uuid>.wav``) that
    the frontend can use to fetch or play the audio once status is ``done``.
    """
    use_case = GetJobStatusUseCase(repo, cache)
    job = await use_case.execute(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Build the correct relative path when serving from DB (absolute stored path)
    file_path = job.file_path
    if file_path and not file_path.startswith("/static/"):
        file_path = f"/static/audio/{Path(file_path).name}"

    return StatusResponse(status=job.status, file_path=file_path, error=job.error)


@router.get(
    "/history",
    response_model=list[AudioGenerationOut],
    summary="List previous generation jobs",
)
async def get_history(
    repo: Annotated[JobRepository, Depends(get_job_repo)],
    limit: int = 100,
    offset: int = 0,
) -> list[AudioGenerationOut]:
    """Return up to *limit* previous generation jobs, newest first."""
    use_case = ListHistoryUseCase(repo)
    jobs = await use_case.execute(limit, offset)
    return [AudioGenerationOut.from_domain(j) for j in jobs]


@router.delete(
    "/audio/{audio_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an audio record and its files",
)
async def delete_audio(
    audio_id: str,
    repo: Annotated[JobRepository, Depends(get_job_repo)],
    cache: Annotated[JobStatusCache, Depends(get_cache)],
    storage: Annotated[FileStorage, Depends(get_file_storage)],
) -> None:
    """
    Delete the DB record **and** the associated audio files from disk.

    Returns 404 if the record is not found; always returns 204 on success.
    """
    use_case = DeleteAudioUseCase(repo, cache, storage)
    job = await use_case.execute(audio_id)

    if job is None:
        # use case returns None for both invalid-UUID and not-found
        raise HTTPException(status_code=404, detail="Audio record not found")
    logger.info("DELETE /audio/%s: removed", audio_id)


# ---------------------------------------------------------------------------
# Voice profile endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/voices",
    response_model=list[VoiceOut],
    summary="List available voice profiles",
)
async def list_voices(
    voice_repo: Annotated[VoiceRepository, Depends(get_voice_repo)],
) -> list[VoiceOut]:
    """Return all voice profiles stored in the voices directory."""
    use_case = ListVoicesUseCase(voice_repo)
    profiles = await use_case.execute()
    return [VoiceOut(id=p.id, name=p.name) for p in profiles]


@router.post(
    "/voices",
    response_model=VoiceOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a new voice profile",
)
async def create_voice(
    voice_repo: Annotated[VoiceRepository, Depends(get_voice_repo)],
    name: str = Form(..., min_length=1, max_length=100),
    transcript: str = Form(
        default="", description="Text spoken in the reference audio"
    ),
    audio: UploadFile = File(..., description="Reference WAV audio file"),
) -> VoiceOut:
    """
    Upload a WAV reference audio file to create a voice profile.

    The ``name`` becomes the display label; the URL-safe slug is the ``id``.
    The optional ``transcript`` improves voice-cloning accuracy for Fish Speech.
    """
    voice_id = _slugify_voice_name(name)
    if not voice_id:
        raise HTTPException(status_code=422, detail="Voice name produced an empty slug")

    is_wav_content = (audio.content_type or "").lower() in (
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
    )
    is_wav_name = (audio.filename or "").lower().endswith(".wav")
    if not (is_wav_content or is_wav_name):
        raise HTTPException(status_code=422, detail="audio must be a WAV file (.wav)")

    content = await audio.read()
    use_case = CreateVoiceUseCase(voice_repo)
    profile = await use_case.execute(voice_id, name, content, transcript)
    logger.info("POST /voices: created voice %r (%d bytes)", voice_id, len(content))
    return VoiceOut(id=profile.id, name=profile.name)


@router.delete(
    "/voices/{voice_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a voice profile",
)
async def delete_voice(
    voice_id: str,
    voice_repo: Annotated[VoiceRepository, Depends(get_voice_repo)],
) -> None:
    """Remove the WAV and transcript files for a voice profile."""
    if not re.fullmatch(r"[\w\-]{1,64}", voice_id):
        raise HTTPException(status_code=422, detail="Invalid voice_id")

    use_case = DeleteVoiceUseCase(voice_repo)
    deleted = await use_case.execute(voice_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Voice not found")
    logger.info("DELETE /voices/%s: removed", voice_id)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@router.get("/health", include_in_schema=False)
async def health(request: Any = None) -> dict[str, Any]:
    """Lightweight liveness probe — does not touch the DB or GPU."""
    queue_depth = -1
    if request is not None and hasattr(request.app.state, "queue"):
        queue_depth = request.app.state.queue.qsize()
    return {"status": "ok", "queue_depth": queue_depth}
