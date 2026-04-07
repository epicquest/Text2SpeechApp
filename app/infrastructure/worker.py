"""
infrastructure/worker.py — Async inference worker that consumes the GPU queue.

This module is the only place that combines async event-loop code (queue reads,
DB writes) with blocking GPU work dispatched to a ThreadPoolExecutor.

It depends directly on infrastructure (AsyncSessionLocal, SQLAlchemyJobRepository,
TTSPort) and uses the domain interfaces for type annotations, keeping the
worker's coupling explicit and localised to the infrastructure layer.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

from app.domain.interfaces import JobStatusCache, TTSPort
from app.infrastructure.database import AsyncSessionLocal, SQLAlchemyJobRepository

logger = logging.getLogger(__name__)


async def inference_worker(  # pylint: disable=too-many-locals
    queue: asyncio.Queue[tuple[str, str, str, str | None]],
    executor: ThreadPoolExecutor,
    tts: TTSPort,
    cache: JobStatusCache,
) -> None:
    """
    Consume jobs from *queue* one at a time and run GPU inference.

    Each queue item is a ``(job_id, text, model_name, voice_id)`` tuple.
    The blocking ``tts.synthesise()`` call is dispatched to *executor*'s
    single worker thread so the event loop stays responsive.

    DB writes happen in this coroutine (async context) after the executor
    returns — never inside the executor thread — to avoid thread-safety
    issues with asyncpg connections.
    """
    loop = asyncio.get_running_loop()
    logger.info("inference_worker: started")

    while True:
        job_id, text, model_name, voice_id = await queue.get()
        logger.info(
            "inference_worker: starting job %s (model=%s voice=%s)",
            job_id,
            model_name,
            voice_id,
        )

        # ---- pending → processing ----
        cache.put(job_id, "processing")
        async with AsyncSessionLocal() as session:
            repo = SQLAlchemyJobRepository(session)
            await repo.update_status(uuid.UUID(job_id), "processing")

        try:
            wav_path, _mp3_path = await loop.run_in_executor(
                executor,
                tts.synthesise,
                text,
                model_name,
                job_id,
                voice_id,
            )

            relative_path = f"/static/audio/{wav_path.name}"

            # ---- processing → done ----
            cache.put(job_id, "done", file_path=relative_path)
            async with AsyncSessionLocal() as session:
                repo = SQLAlchemyJobRepository(session)
                await repo.update_status(
                    uuid.UUID(job_id), "done", file_path=str(wav_path)
                )

            logger.info("inference_worker: job %s done → %s", job_id, wav_path)

        except Exception as exc:  # pylint: disable=broad-exception-caught
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.exception("inference_worker: job %s failed — %s", job_id, error_msg)

            cache.put(job_id, "error", error=error_msg)
            async with AsyncSessionLocal() as session:
                repo = SQLAlchemyJobRepository(session)
                await repo.update_status(uuid.UUID(job_id), "error")

        finally:
            queue.task_done()
