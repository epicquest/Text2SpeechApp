"""
domain/interfaces.py — Abstract port definitions (Dependency Inversion Principle).

Only standard-library and intra-domain imports allowed here.
Every interface is an output port that the use-case layer depends on;
infrastructure provides concrete adapters that implement these ABCs.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from app.domain.entities import GenerationJob, VoiceProfile


class JobRepository(ABC):
    """Persistence interface for GenerationJob entities."""

    @abstractmethod
    async def create(self, job: GenerationJob) -> None:
        """Persist a new job record."""

    @abstractmethod
    async def get(self, job_id: uuid.UUID) -> GenerationJob | None:
        """Return the job with the given id, or None."""

    @abstractmethod
    async def update_status(
        self,
        job_id: uuid.UUID,
        status: str,
        *,
        file_path: str | None = None,
        error: str | None = None,
    ) -> None:
        """Update status and optionally file_path/error of an existing job."""

    @abstractmethod
    async def list_recent(self, limit: int, offset: int) -> list[GenerationJob]:
        """Return up to *limit* jobs, ordered newest first."""

    @abstractmethod
    async def delete(self, job_id: uuid.UUID) -> GenerationJob | None:
        """Delete the job record and return the entity, or None if not found."""


class JobStatusCache(ABC):
    """Fast in-memory status cache for live job polling."""

    @abstractmethod
    def put(
        self,
        job_id: str,
        status: str,
        file_path: str | None = None,
        error: str | None = None,
    ) -> None:
        """Upsert a job's ephemeral state."""

    @abstractmethod
    def get(self, job_id: str) -> GenerationJob | None:
        """Return the cached state as a domain entity, or None."""

    @abstractmethod
    def remove(self, job_id: str) -> None:
        """Evict a job from the cache."""


class VoiceRepository(ABC):
    """Storage interface for voice profiles."""

    @abstractmethod
    async def list(self) -> list[VoiceProfile]:
        """Return all available voice profiles."""

    @abstractmethod
    async def save(
        self,
        voice_id: str,
        name: str,
        audio_content: bytes,
        transcript: str,
    ) -> VoiceProfile:
        """Persist a new voice profile and return it."""

    @abstractmethod
    async def delete(self, voice_id: str) -> bool:
        """Delete a voice profile. Returns True if it existed."""

    @abstractmethod
    def exists(self, voice_id: str) -> bool:
        """Return True if a voice profile with this id is stored."""


class FileStorage(ABC):  # pylint: disable=too-few-public-methods
    """File-system operations port."""

    @abstractmethod
    def delete_audio_files(self, wav_path: str) -> None:
        """Delete the WAV file and its MP3 side-car if they exist."""


class TTSPort(ABC):
    """Infrastructure port for text-to-speech synthesis."""

    @abstractmethod
    async def load(self) -> None:
        """Load all model weights into memory. Called once at startup."""

    @abstractmethod
    def synthesise(
        self,
        text: str,
        model_name: str,
        job_id: str,
        voice_id: str | None,
    ) -> tuple[Path, Path]:
        """
        Synthesise *text* and return ``(wav_path, mp3_path)``.

        Blocking call — must be dispatched to a ``ThreadPoolExecutor``.
        """
