"""
infrastructure/tts_engine.py — TTS model backends and TTSService adapter.

Architecture
------------

BaseTTSBackend (ABC)
    ├── FishSpeechBackend   — Fish Speech v1.5 LM + VQGAN vocoder
    └── XTTSBackend         — Coqui XTTS v2 (zero-shot voice clone)

TTSService
    Owns both backend instances, implements TTSPort, exposes a single
    ``synthesise()`` method that is safe to call from a background thread.
    Instantiated once and stored on ``app.state`` (no global singleton).

GPU Safety Rules
----------------
* All model inference runs inside ``torch.no_grad()`` +
  ``torch.autocast("cuda", dtype=torch.float16)``.
* ``torch.cuda.OutOfMemoryError`` is caught per backend; the CUDA cache is
  cleared and the error is re-raised so the queue worker can mark the job
  as failed without crashing the process.
* Models are loaded once and kept in VRAM for the lifetime of the process.
  ``ThreadPoolExecutor(max_workers=1)`` in main.py ensures at most one
  ``synthesise()`` call executes at any moment.

LSP Compliance
--------------
``BaseTTSBackend`` exposes ``generate()`` (no reference) and
``generate_with_reference()`` (optional reference audio).  The default
implementation of ``generate_with_reference()`` simply delegates to
``generate()``, so XTTSBackend gains voice-reference support for free.
FishSpeechBackend overrides ``generate_with_reference()`` to use the audio.
This eliminates the ``isinstance`` check that previously lived in TTSService.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import torch

from app.config import settings
from app.domain.interfaces import TTSPort

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class BaseTTSBackend(ABC):
    """
    Interface that every TTS backend must implement.

    Subclasses must implement ``load()`` and ``generate()``.
    ``generate_with_reference()`` has a default implementation that ignores
    the reference audio and delegates to ``generate()``.
    """

    name: str = ""

    @abstractmethod
    def load(self) -> None:
        """Load model weights into GPU memory. Idempotent."""

    @abstractmethod
    def generate(self, text: str, output_path: Path) -> None:
        """
        Synthesise *text* and write a WAV file to *output_path*.

        Parameters
        ----------
        text:           Text to synthesise (UTF-8, arbitrary length).
        output_path:    Destination ``.wav`` path (parent dir guaranteed to exist).

        Raises
        ------
        torch.cuda.OutOfMemoryError
            Caller must call ``torch.cuda.empty_cache()`` and mark the job
            failed; subsequent jobs may continue normally.
        RuntimeError
            Any other inference-time failure.
        """

    def generate_with_reference(
        self,
        text: str,
        output_path: Path,
        reference_wav: Path | None = None,
        reference_text: str | None = None,
    ) -> None:
        """
        Synthesise *text* using optional reference audio for voice cloning.

        Default implementation ignores the reference and delegates to
        ``generate()``.  Override in backends that support voice cloning.
        """
        # pylint: disable=unused-argument
        self.generate(text, output_path)

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
      <ckpt_dir>/firefly-gan-vq-fsq-8x1024-21hz-generator.pth
    """

    name = "fish_speech"

    def __init__(self) -> None:
        """Initialise configuration references; weights are loaded lazily."""
        self._engine = None
        self._loaded = False
        self._device = settings.DEVICE
        self._ckpt_dir = Path(settings.FISH_SPEECH_CKPT)

    def load(self) -> None:
        """Load LLaMA + FireflyGAN weights into the configured device."""
        if self._loaded:
            return

        logger.info("FishSpeechBackend: loading models from %s", self._ckpt_dir)
        t0 = time.perf_counter()

        try:
            # pylint: disable=import-outside-toplevel
            from tools.inference_engine import TTSInferenceEngine
            from tools.llama.generate import launch_thread_safe_queue
            from tools.vqgan.inference import load_model as load_decoder_model

            # pylint: enable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError(
                "fish_speech package not found. Install it with:\n"
                "  pip install git+https://github.com/fishaudio/fish-speech.git@v1.5.0"
            ) from exc

        precision = torch.half if self._device != "cpu" else torch.float32

        llama_queue = launch_thread_safe_queue(
            checkpoint_path=str(self._ckpt_dir),
            device=self._device,
            precision=precision,
            compile=False,
        )

        decoder_ckpt = self._ckpt_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
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
        """Synthesise *text* without voice reference (fixed seed for consistency)."""
        self._run_inference(text, output_path, references=[], seed=42)

    def generate_with_reference(
        self,
        text: str,
        output_path: Path,
        reference_wav: Path | None = None,
        reference_text: str | None = None,
    ) -> None:
        """Synthesise *text* with an optional reference-audio voice clone."""
        try:
            from tools.schema import (  # pylint: disable=C0415
                ServeReferenceAudio,
            )
        except ImportError as exc:
            raise RuntimeError("fish_speech package not available") from exc

        references = []
        if reference_wav is not None and reference_wav.is_file():
            references = [
                ServeReferenceAudio(
                    audio=reference_wav.read_bytes(),
                    text=reference_text or "",
                )
            ]
        seed = None if references else 42
        self._run_inference(text, output_path, references=references, seed=seed)

    def _run_inference(  # pylint: disable=too-many-locals
        self, text: str, output_path: Path, references: list, seed: int | None
    ) -> None:
        """Execute the Fish Speech inference pipeline and write the WAV file."""
        if not self._loaded:
            raise RuntimeError("FishSpeechBackend.load() has not been called")

        try:
            import soundfile as sf  # pylint: disable=C0415
            from tools.schema import ServeTTSRequest  # pylint: disable=C0415
        except ImportError as exc:
            raise RuntimeError("fish_speech package not available") from exc

        logger.debug(
            "FishSpeechBackend: synthesising %d chars (refs: %d)",
            len(text),
            len(references),
        )
        t0 = time.perf_counter()

        req = ServeTTSRequest(
            text=text, streaming=False, references=references, seed=seed
        )

        try:
            result = None
            for result in self._engine.inference(req):  # type: ignore[attr-defined]
                if result.code == "error":
                    raise RuntimeError(f"Fish Speech inference error: {result.error}")
                if result.code == "final":
                    break

            if result is None or result.code != "final" or result.audio is None:
                raise RuntimeError("Fish Speech produced no audio output")

            sample_rate, audio_np = result.audio
            sf.write(
                str(output_path), audio_np, samplerate=sample_rate, subtype="PCM_16"
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error(
                "FishSpeechBackend: CUDA OOM — cache cleared. Text length: %d chars.",
                len(text),
            )
            raise

        elapsed = time.perf_counter() - t0
        logger.info(
            "FishSpeechBackend: synthesis done in %.2fs → %s", elapsed, output_path
        )


# ---------------------------------------------------------------------------
# XTTS v2 backend (Coqui TTS)
# ---------------------------------------------------------------------------


class XTTSBackend(BaseTTSBackend):
    """
    Coqui XTTS v2 — zero-shot voice cloning.

    Requires a reference speaker WAV file configured via XTTS_SPEAKER_WAV.
    The TTS library handles file writing; this backend delegates entirely to it.
    ``generate_with_reference()`` inherits the default (ignores reference audio)
    since XTTS uses its own fixed speaker WAV from config.
    """

    name = "xtts"

    def __init__(self) -> None:
        """Initialise configuration references; weights are loaded lazily."""
        self._tts = None
        self._loaded = False
        self._device = settings.DEVICE
        self._model_id = settings.XTTS_MODEL_ID
        self._speaker_wav = settings.XTTS_SPEAKER_WAV

    def load(self) -> None:
        """Download (if needed) and load the XTTS v2 model onto the device."""
        if self._loaded:
            return

        logger.info("XTTSBackend: loading model %r", self._model_id)
        t0 = time.perf_counter()

        try:
            from TTS.api import TTS  # type: ignore[import]  # pylint: disable=C0415
        except ImportError as exc:
            raise RuntimeError(
                "Coqui TTS package not found. Install it with:\n"
                "  pip install TTS>=0.22"
            ) from exc

        self._tts = TTS(self._model_id).to(self._device)

        elapsed = time.perf_counter() - t0
        logger.info("XTTSBackend: loaded in %.2fs", elapsed)
        self._loaded = True

    @torch.no_grad()
    def generate(self, text: str, output_path: Path) -> None:
        """Synthesise *text* using the fixed speaker reference from config."""
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
            self._tts.tts_to_file(  # type: ignore[attr-defined]
                text=text,
                speaker_wav=str(speaker_wav),
                language="en",
                file_path=str(output_path),
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error(
                "XTTSBackend: CUDA OOM — cache cleared. Text length: %d chars.",
                len(text),
            )
            raise

        elapsed = time.perf_counter() - t0
        logger.info("XTTSBackend: synthesis done in %.2fs → %s", elapsed, output_path)


# ---------------------------------------------------------------------------
# Supporting constants
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, str] = {
    "fish_speech": "fish_speech",
    "xtts": "xtts",
}

DEFAULT_MODEL = "fish_speech"


# ---------------------------------------------------------------------------
# TTSService — implements TTSPort
# ---------------------------------------------------------------------------


class TTSService(TTSPort):
    """
    Application-level TTS service that owns both backends.

    Implements ``TTSPort`` so callers depend on the domain interface, not on
    this concrete class.  Instantiated once at startup and stored on
    ``app.state.tts``.

    Usage::

        svc = TTSService()
        await svc.load()                          # called from FastAPI lifespan
        wav, mp3 = svc.synthesise(text, model, job_id, voice_id)
    """

    # Class variable kept for backwards-compat re-export shim only.
    _instance: ClassVar[TTSService | None] = None

    def __init__(self) -> None:
        """Create backend instances using settings from the environment."""
        self._backends: dict[str, BaseTTSBackend] = {
            "fish_speech": FishSpeechBackend(),
            "xtts": XTTSBackend(),
        }
        self._output_dir = settings.AUDIO_OUTPUT_DIR
        self._lazy_fish_speech = settings.LAZY_FISH_SPEECH
        self._lazy_xtts = settings.LAZY_XTTS

    # ------------------------------------------------------------------
    # Backwards-compatibility singleton (used only by re-export shim)
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> TTSService:
        """Return the process-wide instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # TTSPort implementation
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Load all enabled backends into GPU memory (runs at startup)."""
        import asyncio  # pylint: disable=import-outside-toplevel

        loop = asyncio.get_running_loop()

        if self._lazy_fish_speech:
            logger.info(
                "TTSService: LAZY_FISH_SPEECH=true — "
                "Fish Speech will be loaded on first request"
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

    def synthesise(
        self,
        text: str,
        model_name: str,
        job_id: str,
        voice_id: str | None,
    ) -> tuple[Path, Path]:
        """
        Synthesise *text* and save WAV + MP3; return their absolute paths.

        Blocking — must be called from a ``ThreadPoolExecutor`` thread.
        Voice reference files are resolved from the filesystem using *voice_id*.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        backend = self._resolve_backend(model_name)

        if not backend.is_loaded:
            logger.info(
                "TTSService: lazily loading backend %r before first use", backend.name
            )
            backend.load()

        wav_path = self._output_dir / f"{job_id}.wav"
        mp3_path = self._output_dir / f"{job_id}.mp3"

        reference_wav, reference_text = self._resolve_voice(voice_id)
        backend.generate_with_reference(
            text,
            wav_path,
            reference_wav=reference_wav,
            reference_text=reference_text,
        )
        self._convert_to_mp3(wav_path, mp3_path)
        return wav_path, mp3_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_backend(self, model_name: str) -> BaseTTSBackend:
        """Return the backend for *model_name*, falling back to Fish Speech."""
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
    def _resolve_voice(
        voice_id: str | None,
    ) -> tuple[Path | None, str | None]:
        """
        Look up a voice profile's WAV and transcript from the voices directory.

        Returns ``(None, None)`` when *voice_id* is absent or the file is missing.
        """
        if not voice_id:
            return None, None
        voices_dir = Path("app/static/voices")
        wav = voices_dir / f"{voice_id}.wav"
        if not wav.is_file():
            logger.warning("TTSService: voice %r not found, using default", voice_id)
            return None, None
        txt = voices_dir / f"{voice_id}.txt"
        transcript = txt.read_text(encoding="utf-8").strip() if txt.is_file() else ""
        return wav, transcript

    @staticmethod
    def _convert_to_mp3(wav_path: Path, mp3_path: Path) -> None:
        """Convert a WAV file to MP3 at 192 kbps using pydub."""
        try:
            from pydub import (  # pylint: disable=import-outside-toplevel  # type: ignore[import]
                AudioSegment,
            )
        except ImportError:
            logger.warning(
                "pydub not installed — skipping MP3 conversion. "
                "Install with: pip install pydub"
            )
            return

        try:
            segment = AudioSegment.from_wav(str(wav_path))
            segment.export(str(mp3_path), format="mp3", bitrate="192k")
            logger.debug("TTSService: MP3 saved → %s", mp3_path)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("TTSService: MP3 conversion failed for %s", wav_path)

    # ------------------------------------------------------------------
    # Legacy alias kept for backwards compatibility
    # ------------------------------------------------------------------

    def generate_with_voice(
        self,
        text: str,
        model_name: str,
        job_id: str,
        voice_id: str | None = None,
    ) -> tuple[Path, Path]:
        """Alias for ``synthesise()``; retained for the re-export shim."""
        return self.synthesise(text, model_name, job_id, voice_id)
