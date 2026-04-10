import io
from typing import Union

from faster_whisper import WhisperModel

# Module-level singleton — model loads once per process (~30–60s on first run)
_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """Lazy-load the Faster-Whisper base model.

    Apple Silicon note: CTranslate2 (used by faster-whisper) does not support
    Apple's Metal (MPS) backend. We use device="cpu" with int8 quantization,
    which is the correct and fastest option for M-series chips.
    """
    global _model
    if _model is None:
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model


def transcribe_audio(audio_input: Union[str, io.BytesIO]) -> str:
    """Transcribe audio to text using Faster-Whisper (local, offline).

    Args:
        audio_input: File path string OR BytesIO object containing raw audio bytes.

    Returns:
        Transcribed text as a single string.

    Raises:
        RuntimeError: If no speech is detected or transcription fails.
    """
    model = _get_model()

    # vad_filter=True silences non-speech segments before transcription
    segments, _info = model.transcribe(
        audio_input,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    # segments is a LAZY generator — iteration triggers the actual transcription
    text = " ".join(segment.text.strip() for segment in segments).strip()

    if not text:
        raise RuntimeError(
            "No speech detected. Ensure the audio contains clear speech and is not silent."
        )

    return text
