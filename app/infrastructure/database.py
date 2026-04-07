"""
infrastructure/database.py — SQLAlchemy ORM models, engine, and concrete
JobRepository adapter.

Exports
-------
Base                        Declarative base class.
AudioGeneration             ORM model (one row per TTS job).
async_engine                Async SQLAlchemy engine (singleton).
AsyncSessionLocal           Async session factory.
init_db()                   Creates tables on first startup (idempotent).
get_session()               FastAPI dependency yielding an AsyncSession.
SQLAlchemyJobRepository     Concrete implementation of JobRepository.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Text, func, select
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings
from app.domain.entities import GenerationJob
from app.domain.interfaces import JobRepository

# ---------------------------------------------------------------------------
# Engine + session factory
# ---------------------------------------------------------------------------

async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = (  # pylint: disable=invalid-name
    async_sessionmaker(
        bind=async_engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )
)


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):  # pylint: disable=too-few-public-methods
    """Shared declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------


class AudioGeneration(Base):
    """
    ORM model representing one TTS generation job.

    This is an *infrastructure* class — domain code must not import it.
    Domain code uses the ``GenerationJob`` dataclass instead.
    """

    __tablename__ = "audio_generations"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    voice_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        server_default="pending",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),  # pylint: disable=not-callable
        default=lambda: datetime.now(timezone.utc),
    )

    def to_domain(self) -> GenerationJob:
        """Map this ORM record to the pure domain entity."""
        return GenerationJob(
            id=self.id,
            text=self.input_text,
            model_name=self.model_name,
            voice_id=self.voice_id,
            status=self.status,  # type: ignore[arg-type]
            file_path=self.file_path,
            created_at=self.created_at,
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of this record."""
        return {
            "id": str(self.id),
            "input_text": self.input_text,
            "model_name": self.model_name,
            "voice_id": self.voice_id,
            "file_path": self.file_path,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"<AudioGeneration id={self.id!s} model={self.model_name!r}"
            f" status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# Database lifecycle helpers
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """
    Create all tables that do not yet exist (idempotent).

    Should be called once inside the FastAPI lifespan.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, checkfirst=True)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an ``AsyncSession`` per request.

    Usage::

        @router.get("/example")
        async def example(session: AsyncSession = Depends(get_session)):
            ...
    """
    async with AsyncSessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# Concrete repository adapter
# ---------------------------------------------------------------------------


class SQLAlchemyJobRepository(JobRepository):
    """
    Concrete ``JobRepository`` backed by SQLAlchemy async sessions.

    The session is injected at construction time, making this class fully
    testable without touching the database.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Accept an externally-managed AsyncSession."""
        self._session = session

    async def create(self, job: GenerationJob) -> None:
        """Insert a new AudioGeneration row from the domain entity."""
        record = AudioGeneration(
            id=job.id,
            input_text=job.text,
            model_name=job.model_name,
            voice_id=job.voice_id,
            status=job.status,
        )
        self._session.add(record)
        await self._session.commit()

    async def get(self, job_id: uuid.UUID) -> GenerationJob | None:
        """Return the domain entity for the given id, or None."""
        record = await self._session.get(AudioGeneration, job_id)
        return record.to_domain() if record is not None else None

    async def update_status(
        self,
        job_id: uuid.UUID,
        status: str,
        *,
        file_path: str | None = None,
        error: str | None = None,
    ) -> None:
        """Update the status (and optionally file_path) of an existing record."""
        record = await self._session.get(AudioGeneration, job_id)
        if record is None:
            return
        record.status = status
        if file_path is not None:
            record.file_path = file_path
        await self._session.commit()

    async def list_recent(self, limit: int, offset: int) -> list[GenerationJob]:
        """Return up to *limit* jobs ordered by created_at descending."""
        result = await self._session.execute(
            select(AudioGeneration)
            .order_by(AudioGeneration.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [row.to_domain() for row in result.scalars().all()]

    async def delete(self, job_id: uuid.UUID) -> GenerationJob | None:
        """Delete the record and return the domain entity, or None."""
        record = await self._session.get(AudioGeneration, job_id)
        if record is None:
            return None
        domain_obj = record.to_domain()
        await self._session.delete(record)
        await self._session.commit()
        return domain_obj
