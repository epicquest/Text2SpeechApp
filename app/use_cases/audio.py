"""
use_cases/audio.py — Application business rules for audio-generation jobs.

Depends only on domain entities and interfaces.
No SQLAlchemy, no FastAPI, no framework-specific imports.
"""

from __future__ import annotations

import asyncio
import uuid

from app.domain.entities import GenerationJob
from app.domain.interfaces import FileStorage, JobRepository, JobStatusCache


class EnqueueJobUseCase:  # pylint: disable=too-few-public-methods
    """Create a pending job record and push it onto the inference queue."""

    def __init__(
        self,
        repo: JobRepository,
        cache: JobStatusCache,
        queue: asyncio.Queue[tuple[str, str, str, str | None]],
    ) -> None:
        """Inject repository, cache, and the async inference queue."""
        self._repo = repo
        self._cache = cache
        self._queue = queue

    async def execute(self, text: str, model_name: str, voice_id: str | None) -> str:
        """Persist the job, seed the cache, push onto the queue, return job_id."""
        job_id = uuid.uuid4()
        job = GenerationJob(
            id=job_id,
            text=text,
            model_name=model_name,
            voice_id=voice_id,
            status="pending",
        )
        await self._repo.create(job)
        self._cache.put(str(job_id), "pending")
        await self._queue.put((str(job_id), text, model_name, voice_id))
        return str(job_id)


class GetJobStatusUseCase:  # pylint: disable=too-few-public-methods
    """Return the current status of a generation job."""

    def __init__(self, repo: JobRepository, cache: JobStatusCache) -> None:
        """Inject repository (slow path) and cache (fast path)."""
        self._repo = repo
        self._cache = cache

    async def execute(self, job_id: str) -> GenerationJob | None:
        """Return the job entity from cache if live, else from the DB."""
        cached = self._cache.get(job_id)
        if cached is not None:
            return cached
        try:
            uid = uuid.UUID(job_id)
        except ValueError:
            return None
        return await self._repo.get(uid)


class ListHistoryUseCase:  # pylint: disable=too-few-public-methods
    """Return a paginated list of recent generation jobs."""

    def __init__(self, repo: JobRepository) -> None:
        """Inject the job repository."""
        self._repo = repo

    async def execute(self, limit: int, offset: int) -> list[GenerationJob]:
        """Hard-caps limit at 500 to prevent runaway queries."""
        return await self._repo.list_recent(limit=min(limit, 500), offset=offset)


class DeleteAudioUseCase:  # pylint: disable=too-few-public-methods
    """Delete a generation-job record, its audio files, and its cache entry."""

    def __init__(
        self, repo: JobRepository, cache: JobStatusCache, storage: FileStorage
    ) -> None:
        """Inject repository, cache, and file-system storage port."""
        self._repo = repo
        self._cache = cache
        self._storage = storage

    async def execute(self, audio_id: str) -> GenerationJob | None:
        """
        Delete the job and return the entity, or None if the id is invalid.

        The caller (router) is responsible for raising 404 when None is returned
        due to a record not being found.
        """
        try:
            uid = uuid.UUID(audio_id)
        except ValueError:
            return None
        job = await self._repo.delete(uid)
        if job is not None:
            if job.file_path:
                self._storage.delete_audio_files(job.file_path)
            self._cache.remove(audio_id)
        return job
