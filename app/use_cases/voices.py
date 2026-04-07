"""
use_cases/voices.py — Application business rules for voice-profile management.

Depends only on domain entities and interfaces.
No filesystem paths, no FastAPI, no framework-specific imports.
"""

from __future__ import annotations

from app.domain.entities import VoiceProfile
from app.domain.interfaces import VoiceRepository


class ListVoicesUseCase:  # pylint: disable=too-few-public-methods
    """Return all available voice profiles."""

    def __init__(self, repo: VoiceRepository) -> None:
        """Inject the voice repository."""
        self._repo = repo

    async def execute(self) -> list[VoiceProfile]:
        """Return all stored voice profiles."""
        return await self._repo.list()


class CreateVoiceUseCase:  # pylint: disable=too-few-public-methods
    """Persist a new voice profile from raw audio bytes."""

    def __init__(self, repo: VoiceRepository) -> None:
        """Inject the voice repository."""
        self._repo = repo

    async def execute(
        self,
        voice_id: str,
        name: str,
        audio_content: bytes,
        transcript: str,
    ) -> VoiceProfile:
        """Save and return the new voice profile."""
        return await self._repo.save(voice_id, name, audio_content, transcript)


class DeleteVoiceUseCase:  # pylint: disable=too-few-public-methods
    """Remove a voice profile from storage."""

    def __init__(self, repo: VoiceRepository) -> None:
        """Inject the voice repository."""
        self._repo = repo

    async def execute(self, voice_id: str) -> bool:
        """Return True if the profile was deleted, False if it did not exist."""
        return await self._repo.delete(voice_id)
