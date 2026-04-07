"""
infrastructure/voice_storage.py — Filesystem-backed VoiceRepository adapter.
"""

from __future__ import annotations

from pathlib import Path

from app.domain.entities import VoiceProfile
from app.domain.interfaces import VoiceRepository


class FilesystemVoiceRepository(VoiceRepository):
    """
    ``VoiceRepository`` that stores voice profiles as WAV + TXT file pairs.

    Directory layout::

        <voices_dir>/
          <voice_id>.wav    ← reference audio
          <voice_id>.txt    ← transcript (may be empty)
    """

    def __init__(self, voices_dir: Path) -> None:
        """Accept the directory where voice files are stored."""
        self._dir = voices_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    async def list(self) -> list[VoiceProfile]:
        """Return all voice profiles ordered by id."""
        return [
            VoiceProfile(
                id=wav.stem,
                name=wav.stem.replace("_", " ").replace("-", " ").title(),
            )
            for wav in sorted(self._dir.glob("*.wav"))
        ]

    async def save(
        self,
        voice_id: str,
        name: str,
        audio_content: bytes,
        transcript: str,
    ) -> VoiceProfile:
        """Write the WAV and transcript files; return the new profile."""
        (self._dir / f"{voice_id}.wav").write_bytes(audio_content)
        (self._dir / f"{voice_id}.txt").write_text(transcript, encoding="utf-8")
        return VoiceProfile(id=voice_id, name=name)

    async def delete(self, voice_id: str) -> bool:
        """Delete the WAV (and transcript if present). Returns True if WAV existed."""
        wav = self._dir / f"{voice_id}.wav"
        if not wav.is_file():
            return False
        wav.unlink()
        txt = self._dir / f"{voice_id}.txt"
        txt.unlink(missing_ok=True)
        return True

    def exists(self, voice_id: str) -> bool:
        """Return True if the WAV file for *voice_id* exists."""
        return (self._dir / f"{voice_id}.wav").is_file()
