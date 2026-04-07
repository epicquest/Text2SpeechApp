"""
api/dependencies.py — FastAPI dependency functions.

These thin helpers wire the infrastructure concrete implementations to the
use-case layer, keeping endpoint handlers free of construction logic.
"""

from __future__ import annotations

import asyncio

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.domain.interfaces import (
    FileStorage,
    JobRepository,
    JobStatusCache,
    VoiceRepository,
)
from app.infrastructure.database import SQLAlchemyJobRepository, get_session
from app.infrastructure.job_cache import LocalFileStorage
from app.infrastructure.voice_storage import FilesystemVoiceRepository

_VOICES_DIR = settings.AUDIO_OUTPUT_DIR.parent / "voices"


def get_job_repo(
    session: AsyncSession = Depends(get_session),
) -> JobRepository:
    """Provide a request-scoped ``SQLAlchemyJobRepository``."""
    return SQLAlchemyJobRepository(session)


def get_cache(request: Request) -> JobStatusCache:
    """Return the process-wide ``InMemoryJobStatusCache`` from app state."""
    return request.app.state.cache  # type: ignore[no-any-return]


def get_queue(
    request: Request,
) -> asyncio.Queue[tuple[str, str, str, str | None]]:
    """Return the inference queue from app state."""
    return request.app.state.queue  # type: ignore[no-any-return]


def get_voice_repo() -> VoiceRepository:
    """Provide a ``FilesystemVoiceRepository`` scoped to the voices directory."""
    return FilesystemVoiceRepository(_VOICES_DIR)


def get_file_storage() -> FileStorage:
    """Provide a ``LocalFileStorage`` instance."""
    return LocalFileStorage()
