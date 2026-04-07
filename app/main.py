"""
main.py — Composition root and FastAPI application factory.

Responsibilities
----------------
1. Logging configuration.
2. Lifespan management — create directories, initialise DB, load TTS models,
   start the inference worker task, and wire all dependencies onto app.state.
3. Middleware and static-file mounts.
4. Include the API router.

Concurrency Model
-----------------
  ┌──────────────────┐          ┌────────────────────────────┐
  │  HTTP Handlers   │ enqueue  │   inference_worker() task  │
  │  (async, event   │ ──────►  │   (asyncio, event loop)    │
  │   loop)          │          │                            │
  └──────────────────┘          │  await run_in_executor()   │
                                │    │                       │
                                │    ▼                       │
                                │  ThreadPoolExecutor        │
                                │  (max_workers=1)           │
                                │    │  TTSService.synthesise │
                                │    │  (blocks GPU thread)  │
                                └────┴───────────────────────┘

The executor has exactly one worker thread, which guarantees that at most one
GPU inference call executes at any moment, without requiring Celery or Redis.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.router import router
from app.config import settings
from app.infrastructure.database import init_db
from app.infrastructure.job_cache import InMemoryJobStatusCache
from app.infrastructure.tts_engine import TTSService
from app.infrastructure.worker import inference_worker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.getLevelName(settings.LOG_LEVEL),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastAPI lifespan (composition root — wires all infrastructure)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown.

    Startup sequence:
      1. Ensure audio output directory exists.
      2. Create DB tables (idempotent).
      3. Create in-memory job-status cache.
      4. Instantiate and load TTS models into GPU memory.
      5. Create queue, executor, and inference worker task.
      6. Attach everything to app.state for dependency injection.

    Shutdown sequence:
      7. Cancel worker task gracefully.
      8. Shutdown the executor (waits for any running GPU job to finish).
    """
    # 1. Ensure audio directory exists
    output_dir = settings.AUDIO_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("lifespan: audio output dir → %s", output_dir)

    voices_dir = output_dir.parent / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)

    # 2. Initialise database
    await init_db()
    logger.info("lifespan: database tables ready")

    # 3. In-memory job-status cache
    cache = InMemoryJobStatusCache()

    # 4. TTS service (no singleton — injected via app.state)
    tts = TTSService()
    await tts.load()

    # 5. Inference queue + executor + worker task
    queue: asyncio.Queue[tuple[str, str, str, str | None]] = asyncio.Queue()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-worker")

    worker_task = asyncio.create_task(
        inference_worker(queue, executor, tts, cache),
        name="inference-worker",
    )

    # 6. Expose on app.state (read by API dependency functions)
    _app.state.queue = queue
    _app.state.cache = cache
    _app.state.tts = tts

    logger.info("lifespan: application ready")
    yield

    # 7. Cancel worker task
    logger.info("lifespan: shutting down inference worker")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

    # 8. Drain the executor (waits for any in-flight GPU job)
    executor.shutdown(wait=True)
    logger.info("lifespan: shutdown complete")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Local TTS Service",
    description=(
        "GPU-accelerated text-to-speech API powered by Fish Speech v1.5 "
        "and Coqui XTTS v2.  No external APIs — fully local inference."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static-file mounts (directories must exist before mounting)
settings.AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_voices_dir = settings.AUDIO_OUTPUT_DIR.parent / "voices"
_voices_dir.mkdir(parents=True, exist_ok=True)

app.mount(
    "/static/audio",
    StaticFiles(directory=str(settings.AUDIO_OUTPUT_DIR)),
    name="audio",
)
app.mount(
    "/static/voices",
    StaticFiles(directory=str(_voices_dir)),
    name="voices",
)

# All routes are defined in the API layer
app.include_router(router)
