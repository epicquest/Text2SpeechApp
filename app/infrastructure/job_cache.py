"""
infrastructure/job_cache.py — In-memory job-status cache and local file storage.

Both are thin infrastructure helpers with no database or network I/O.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TypedDict

from app.domain.entities import GenerationJob, JobStatus
from app.domain.interfaces import FileStorage, JobStatusCache


class _CacheEntry(TypedDict):
    """Ephemeral job state kept in process memory."""

    status: JobStatus
    file_path: str | None
    error: str | None


class InMemoryJobStatusCache(JobStatusCache):
    """
    ``JobStatusCache`` implementation backed by a plain Python dict.

    Thread-safety note: mutations happen exclusively from the asyncio event
    loop, so no locking is required.
    """

    def __init__(self) -> None:
        """Initialise an empty cache."""
        self._store: dict[str, _CacheEntry] = {}

    def put(
        self,
        job_id: str,
        status: str,
        file_path: str | None = None,
        error: str | None = None,
    ) -> None:
        """Upsert the ephemeral state for *job_id*."""
        self._store[job_id] = _CacheEntry(
            status=status,  # type: ignore[typeddict-item]
            file_path=file_path,
            error=error,
        )

    def get(self, job_id: str) -> GenerationJob | None:
        """Return the cached state as a domain entity, or None."""
        entry = self._store.get(job_id)
        if entry is None:
            return None
        return GenerationJob(
            id=uuid.UUID(job_id),
            text="",
            model_name="",
            voice_id=None,
            status=entry["status"],
            file_path=entry["file_path"],
            error=entry["error"],
        )

    def remove(self, job_id: str) -> None:
        """Evict *job_id* from the cache."""
        self._store.pop(job_id, None)


class LocalFileStorage(FileStorage):  # pylint: disable=too-few-public-methods
    """``FileStorage`` implementation that deletes files from the local disk."""

    def delete_audio_files(self, wav_path: str) -> None:
        """Delete the WAV file and its MP3 side-car if they exist on disk."""
        path = Path(wav_path)
        for candidate in (path, path.with_suffix(".mp3")):
            if candidate.is_file():
                candidate.unlink()
