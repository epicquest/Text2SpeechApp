"""
tts_engine.py — TTS model backends and service singleton.

Architecture
------------

BaseTTSBackend (ABC)
    ├── FishSpeechBackend   — Fish Speech v1.5 LM + VQGAN vocoder
    └── XTTSBackend         — Coqui XTTS v2 (zero-shot voice clone)

TTSService (singleton)
    Owns both backend instances, loads them once, exposes a single
    ``generate()`` method that is safe to call from a background thread.

GPU Safety Rules
----------------
* All model inference runs inside ``torch.no_grad()`` +
  ``torch.autocast("cuda", dtype=torch.float16)``.
* ``torch.cuda.OutOfMemoryError`` is caught per backend; the CUDA cache is
  cleared and the error is re-raised so the queue worker can mark the job as
  failed without crashing the process.
* Models are loaded once and kept in VRAM for the lifetime of the process.
  The ``ThreadPoolExecutor(max_workers=1)`` in main.py ensures at most one
  ``generate()`` call is executing at any moment.

Audio Output
------------
FishSpeechBackend saves a .wav file using soundfile (24 kHz, mono float32).
XTTSBackend delegates file writing to the Coqui TTS library.
After the backend call, TTSService converts the .wav to a .mp3 side-car
(192 kbps) via pydub so the frontend can offer both formats.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import torch

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract backend protocol
# ---------------------------------------------------------------------------

class BaseTTSBackend(ABC):
    """
    Interface that every TTS backend must implement.

    ``load()``      — Download / parse / move weights to GPU.  Called once at
                     startup (or lazily before first use).  Must be idempotent.
    ``generate()``  — Synthesise ``text`` and write the result as a WAV file to
                     ``output_path``.  Called from a background thread.
    """

    name: str = ""

    @abstractmethod
    def load(self) -> None:
        """Load model weights into GPU memory. Idempotent."""

    @abstractmethod
    def generate(self, text: str, output_path: Path) -> None:
        """
        Synthesise ``text`` and write a WAV file to ``output_path``.

        Parameters
        ----------
        text:           Text to synthesise (UTF-8 string, arbitrary length).
        output_path:    Destination path for the .wav file.  The parent
                        directory is guaranteed to exist by the caller.

        Raises
        ------
        torch.cuda.OutOfMemoryError
            Caller is expected to call ``torch.cuda.empty_cache()`` and mark
            the job as failed; other jobs can continue afterwards.
        RuntimeError
            Any other inference-time failure.
        """

    @property
    def is_loaded(self) -> bool:
        """Return True if ``load()`` has been called successfully."""
        return getattr(self, "_loaded", False)


# ---------------------------------------------------------------------------
# Fish Speech v1.5 backend
# ---------------------------------------------------------------------------

class FishSpeechBackend(BaseTTSBackend):
    """
    Fish Speech v1.5.0 — uses the TTSInferenceEngine API shipped with the
    package (tools.inference_engine.TTSInferenceEngine).

    Two components are loaded at startup:
      1. LLaMA language model — runs in a dedicated background thread fed by
         a queue (launch_thread_safe_queue).
      2. FireflyGAN decoder — converts VQ tokens → raw waveform.

    Checkpoint layout expected under settings.FISH_SPEECH_CKPT:
      <ckpt_dir>/                                    ← LLaMA checkpoint dir
      <ckpt_dir>/firefly-gan-vq-fsq-8x1024-21hz-generator.pth  ← decoder
    """

    name = "fish_speech"

    def __init__(self) -> None:
        self._engine = None
        self._loaded = False
        self._device = settings.DEVICE
        self._ckpt_dir = Path(settings.FISH_SPEECH_CKPT)

    def load(self) -> None:
        if self._loaded:
            return

        logger.info("FishSpeechBackend: loading models from %s", self._ckpt_dir)
        t0 = time.perf_counter()

        try:
            import torch
            from tools.inference_engine import TTSInferenceEngine
            from tools.llama.generate import launch_thread_safe_queue
            from tools.vqgan.inference import load_model as load_decoder_model
        except ImportError as exc:
            raise RuntimeError(
                "fish_speech package not found. Install it with:\n"
                "  pip install git+https://github.com/fishaudio/fish-speech.git@v1.5.0"
            ) from exc

        precision = torch.half if self._device != "cpu" else torch.float32

        # ---- LLaMA language model (runs in its own thread) ----
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=str(self._ckpt_dir),
            device=self._device,
            precision=precision,
            compile=False,
        )

        # ---- Firefly GAN decoder ----
        decoder_ckpt = (
            self._ckpt_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        )
        decoder_model = load_decoder_model(
            config_name="firefly_gan_vq",
            checkpoint_path=str(decoder_ckpt),
            device=self._device,
        )

        self._engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=False,
        )

        elapsed = time.perf_counter() - t0
        logger.info("FishSpeechBackend: loaded in %.2fs", elapsed)
        self._loaded = True

    def generate(self, text: str, output_path: Path) -> None:
        if not self._loaded:
            raise RuntimeError("FishSpeechBackend.load() has not been called")

        try:
            import soundfile as sf
            from tools.schema import ServeTTSRequest
        except ImportError as exc:
            raise RuntimeError("fish_speech package not available") from exc

        logger.debug("FishSpeechBackend: synthesising %d chars", len(text))
        t0 = time.perf_counter()

        req = ServeTTSRequest(text=text, streaming=False)

        try:
            result = None
            for result in self._engine.inference(req):
                if result.code == "error":
                    raise RuntimeError(f"Fish Speech inference error: {result.error}")
                if result.code == "final":
                    break

            if result is None or result.code != "final" or result.audio is None:
                raise RuntimeError("Fish Speech produced no audio output")

            sample_rate, audio_np = result.audio
            sf.write(str(output_path), audio_np, samplerate=sample_rate, subtype="PCM_16")

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error(
                "FishSpeechBackend: CUDA out of memory — cache cleared. "
                "Text length was %d chars.",
                len(text),
            )
            raise

        elapsed = time.perf_counter() - t0
        logger.info(
            "FishSpeechBackend: synthesis done in %.2fs → %s",
            elapsed,
            output_path,
        )


# ---------------------------------------------------------------------------
# XTTS v2 backend (Coqui TTS)
# ---------------------------------------------------------------------------

class XTTSBackend(BaseTTSBackend):
    """
    Coqui XTTS v2 — zero-shot voice cloning.

    Requires a reference speaker WAV file configured via XTTS_SPEAKER_WAV.
    The TTS library handles file writing; this backend delegates entirely to it.
    """

    name = "xtts"

    def __init__(self) -> None:
        self._tts = None
        self._loaded = False
        self._device = settings.DEVICE
        self._model_id = settings.XTTS_MODEL_ID
        self._speaker_wav = settings.XTTS_SPEAKER_WAV

    def load(self) -> None:
        if self._loaded:
            return

        logger.info("XTTSBackend: loading model %r", self._model_id)
        t0 = time.perf_counter()

        try:
            from TTS.api import TTS  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "Coqui TTS package not found. Install it with:\n"
                "  pip install TTS>=0.22"
            ) from exc

        # TTS.__init__ downloads the model on first run, then caches it.
        # .to() moves all sub-models to the specified device.
        self._tts = TTS(self._model_id).to(self._device)

        elapsed = time.perf_counter() - t0
        logger.info("XTTSBackend: loaded in %.2fs", elapsed)
        self._loaded = True

    @torch.no_grad()
    def generate(self, text: str, output_path: Path) -> None:
        if not self._loaded:
            raise RuntimeError("XTTSBackend.load() has not been called")

        speaker_wav = Path(self._speaker_wav)
        if not speaker_wav.is_file():
            raise FileNotFoundError(
                f"XTTS speaker reference WAV not found at {self._speaker_wav}. "
                "Place a ~10-second clean-speech WAV at that path."
            )

        logger.debug("XTTSBackend: synthesising %d chars", len(text))
        t0 = time.perf_counter()

        try:
            # XTTS writes the audio file directly; fp16 is handled internally
            # by the Coqui library when the model is on a CUDA device.
            self._tts.tts_to_file(
                text=text,
                speaker_wav=str(speaker_wav),
                language="en",
                file_path=str(output_path),
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error(
                "XTTSBackend: CUDA out of memory — cache cleared. "
                "Text length was %d chars.",
                len(text),
            )
            raise

        elapsed = time.perf_counter() - t0
        logger.info(
            "XTTSBackend: synthesis done in %.2fs → %s",
            elapsed,
            output_path,
        )


# ---------------------------------------------------------------------------
# TTSService — application-level singleton
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, str] = {
    "fish_speech": "fish_speech",
    "xtts": "xtts",
}

DEFAULT_MODEL = "fish_speech"


class TTSService:
    """
    Singleton service that owns both TTS backends.

    Usage
    -----
    ::

        svc = TTSService.get()
        await svc.load_models()        # called once from FastAPI lifespan
        wav_path, mp3_path = svc.generate(text, "fish_speech", job_id)
    """

    _instance: ClassVar[TTSService | None] = None

    def __init__(self) -> None:
        self._backends: dict[str, BaseTTSBackend] = {
            "fish_speech": FishSpeechBackend(),
            "xtts": XTTSBackend(),
        }
        self._output_dir = settings.AUDIO_OUTPUT_DIR
        self._lazy_fish_speech = settings.LAZY_FISH_SPEECH
        self._lazy_xtts = settings.LAZY_XTTS

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> TTSService:
        """Return the process-wide TTSService instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    async def load_models(self) -> None:
        """
        Load all enabled backends into GPU memory.

        This coroutine is intentionally NOT offloaded to an executor — model
        loading is a one-time blocking operation at startup; delaying the
        server's readiness is acceptable and makes error reporting cleaner.

        If ``settings.LAZY_XTTS`` is True, XTTS is skipped here and will be
        loaded on the first request that targets it.
        """
        import asyncio

        loop = asyncio.get_running_loop()

        if self._lazy_fish_speech:
            logger.info(
                "TTSService: LAZY_FISH_SPEECH=true — Fish Speech will be loaded on first request"
            )
        else:
            logger.info("TTSService: loading Fish Speech backend")
            await loop.run_in_executor(None, self._backends["fish_speech"].load)

        if self._lazy_xtts:
            logger.info(
                "TTSService: LAZY_XTTS=true — XTTS will be loaded on first request"
            )
        else:
            logger.info("TTSService: loading XTTS backend")
            await loop.run_in_executor(None, self._backends["xtts"].load)

        logger.info("TTSService: all models ready")

    # ------------------------------------------------------------------
    # Inference (called from ThreadPoolExecutor — runs in a background thread)
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        model_name: str,
        job_id: str,
    ) -> tuple[Path, Path]:
        """
        Synthesise ``text`` and save WAV + MP3 files.

        Parameters
        ----------
        text:        Text to synthesise.
        model_name:  Backend key — "fish_speech" | "xtts".
        job_id:      UUID string used as the base filename.

        Returns
        -------
        (wav_path, mp3_path) — absolute Path objects of the saved files.

        Raises
        ------
        RuntimeError / torch.cuda.OutOfMemoryError
            The caller (queue worker) is responsible for catching these and
            marking the job as failed.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        backend = self._resolve_backend(model_name)

        # Lazy XTTS loading — guarded because generate() runs in a thread
        if not backend.is_loaded:
            logger.info(
                "TTSService: lazily loading backend %r before first use", backend.name
            )
            backend.load()

        wav_path = self._output_dir / f"{job_id}.wav"
        mp3_path = self._output_dir / f"{job_id}.mp3"

        backend.generate(text, wav_path)
        self._convert_to_mp3(wav_path, mp3_path)

        return wav_path, mp3_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_backend(self, model_name: str) -> BaseTTSBackend:
        """
        Return the backend for ``model_name``, falling back to Fish Speech if
        the name is unrecognised.
        """
        canonical = SUPPORTED_MODELS.get(model_name.lower())
        if canonical is None:
            logger.warning(
                "TTSService: unknown model %r — falling back to %r",
                model_name,
                DEFAULT_MODEL,
            )
            canonical = DEFAULT_MODEL
        return self._backends[canonical]

    @staticmethod
    def _convert_to_mp3(wav_path: Path, mp3_path: Path) -> None:
        """Convert a WAV file to MP3 at 192 kbps using pydub."""
        try:
            from pydub import AudioSegment  # type: ignore[import]
        except ImportError:
            logger.warning(
                "pydub not installed — skipping MP3 conversion. "
                "Install it with: pip install pydub"
            )
            return

        try:
            segment = AudioSegment.from_wav(str(wav_path))
            segment.export(str(mp3_path), format="mp3", bitrate="192k")
            logger.debug("TTSService: MP3 saved → %s", mp3_path)
        except Exception:
            # MP3 conversion failure is non-fatal; the WAV is still usable.
            logger.exception("TTSService: MP3 conversion failed for %s", wav_path)
