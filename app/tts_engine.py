"""
tts_engine.py — Re-export shim (backwards compatibility).

All TTS backend classes and TTSService have moved to:
    app.infrastructure.tts_engine
"""

from app.infrastructure.tts_engine import (  # noqa: F401
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    BaseTTSBackend,
    FishSpeechBackend,
    TTSService,
    XTTSBackend,
)

__all__ = [
    "BaseTTSBackend",
    "DEFAULT_MODEL",
    "FishSpeechBackend",
    "SUPPORTED_MODELS",
    "TTSService",
    "XTTSBackend",
]
