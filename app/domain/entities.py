"""
domain/entities.py — Core domain entities.

Zero external dependencies: standard library only.
Entities represent enterprise business rules and must never import from
SQLAlchemy, Pydantic, FastAPI, or any other framework.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

JobStatus = Literal["pending", "processing", "done", "error"]


@dataclass
class GenerationJob:  # pylint: disable=too-many-instance-attributes
    """Enterprise business entity representing one TTS synthesis job."""

    id: uuid.UUID
    text: str
    model_name: str
    voice_id: str | None
    status: JobStatus
    file_path: str | None = None
    error: str | None = None
    created_at: datetime | None = None


@dataclass
class VoiceProfile:
    """A saved reference-audio voice profile used for voice cloning."""

    id: str
    name: str
