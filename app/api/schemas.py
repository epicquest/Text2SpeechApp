"""
api/schemas.py — Pydantic request/response schemas (HTTP contract layer).

These are data-transfer objects (DTOs) for the API and must not be passed
into use cases or domain code.  Mapping to/from domain entities is done
in the router.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.domain.entities import GenerationJob


class GenerateRequest(BaseModel):
    """Request body for the POST /generate endpoint."""

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

    @classmethod
    def from_domain(cls, job: GenerationJob) -> StatusResponse:
        """Build a StatusResponse from a domain entity."""
        return cls(
            status=job.status,
            file_path=job.file_path,
            error=job.error,
        )


class AudioGenerationOut(BaseModel):
    """Read model for the GET /history endpoint."""

    id: str
    input_text: str
    model_name: str
    voice_id: str | None
    file_path: str | None
    status: str
    created_at: str | None

    @classmethod
    def from_domain(cls, job: GenerationJob) -> AudioGenerationOut:
        """Build a history record from a domain entity."""
        return cls(
            id=str(job.id),
            input_text=job.text,
            model_name=job.model_name,
            voice_id=job.voice_id,
            file_path=job.file_path,
            status=job.status,
            created_at=job.created_at.isoformat() if job.created_at else None,
        )


class VoiceOut(BaseModel):
    """Voice profile summary returned by the /voices endpoint."""

    id: str
    name: str
