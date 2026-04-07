"""
config.py — Application-wide configuration via Pydantic BaseSettings.

All values are read from environment variables (or a .env file).
A single module-level ``settings`` instance is imported everywhere else so
there is exactly one source of truth and the values are validated at startup.

Environment Variables
---------------------
DATABASE_URL            Full asyncpg DSN, e.g.
                        postgresql+asyncpg://user:pass@localhost:5432/tts_db
FISH_SPEECH_CKPT        Path to Fish Speech v1.5 checkpoint directory.
                        Default: /models/fish-speech-1.5
XTTS_MODEL_ID           Coqui TTS model identifier.
                        Default: tts_models/multilingual/multi-dataset/xtts_v2
XTTS_SPEAKER_WAV        Reference WAV file used by XTTS for zero-shot cloning.
                        Default: /app/static/reference_speaker.wav
AUDIO_OUTPUT_DIR        Directory where generated .wav/.mp3 files are stored.
                        Default: /app/static/audio
DEVICE                  PyTorch device string. Default: cuda
LAZY_XTTS               When "true", XTTS is loaded on first request instead
                        of at startup (useful when VRAM is tight). Default: false
LOG_LEVEL               Python logging level string. Default: INFO
"""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    DATABASE_URL: str

    # ------------------------------------------------------------------ #
    # Model paths / identifiers
    # ------------------------------------------------------------------ #
    FISH_SPEECH_CKPT: str = "/models/fish-speech-1.5"
    XTTS_MODEL_ID: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    XTTS_SPEAKER_WAV: str = "/app/static/reference_speaker.wav"

    # ------------------------------------------------------------------ #
    # Audio output
    # ------------------------------------------------------------------ #
    AUDIO_OUTPUT_DIR: Path = Path("/app/static/audio")

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    DEVICE: str = "cuda"
    LAZY_FISH_SPEECH: bool = False
    LAZY_XTTS: bool = False

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("DATABASE_URL")
    @classmethod
    def _validate_db_url(cls, v: str) -> str:
        if not v.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must use the asyncpg driver, e.g. "
                "postgresql+asyncpg://user:pass@host:5432/dbname"
            )
        return v

    @field_validator("DEVICE")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        allowed = {"cuda", "cpu", "mps"}
        if v not in allowed:
            raise ValueError(f"DEVICE must be one of {allowed}, got {v!r}")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        import logging

        level = logging.getLevelName(v.upper())
        if not isinstance(level, int):
            raise ValueError(f"LOG_LEVEL {v!r} is not a valid Python logging level")
        return v.upper()


# Module-level singleton — import this everywhere.
settings = Settings()  # type: ignore[call-arg]  # DATABASE_URL must be set via env
