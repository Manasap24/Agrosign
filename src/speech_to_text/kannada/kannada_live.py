from functools import lru_cache

import numpy as np
import onnx_asr

SAMPLE_RATE = 16000


@lru_cache(maxsize=1)
def get_kannada_model():
    """Load the Kannada ASR model only once (cached across calls)."""
    print("Loading Kannada ASR model (IndicConformer)...")
    return onnx_asr.load_model("OpenVoiceOS/ai4bharat-indicconformer-kn-onnx")


def _result_to_text(result) -> str:
    """Same safety handling as kannada_recorded.py -- see comment there."""
    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    text = getattr(result, "text", None)

    if text is not None:
        return str(text).strip()

    return str(result).strip()


def transcribe_kannada_audio(audio: np.ndarray) -> str:
    """
    Transcribe a float32 PCM buffer (mono, 16kHz, range -1..1) directly.
    """
    model = get_kannada_model()

    result = model.recognize(audio, sample_rate=SAMPLE_RATE)

    return _result_to_text(result)


# ---------------------------------------------------------------------------
# Optional: standalone mic-based CLI for local testing.
# ---------------------------------------------------------------------------
def start_live_transcription(chunk_seconds: int = 3):
    import queue
    import sounddevice as sd

    audio_queue = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    print("🎤 Kannada Live Speech-to-Text")
    print("Speak in Kannada... Press Ctrl+C to stop.")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback
    ):
        buffer = np.empty((0, 1), dtype=np.float32)

        while True:
            data = audio_queue.get()
            buffer = np.concatenate((buffer, data), axis=0)

            if len(buffer) >= SAMPLE_RATE * chunk_seconds:
                text = transcribe_kannada_audio(buffer.flatten())

                if text:
                    print("Kannada:", text)

                buffer = np.empty((0, 1), dtype=np.float32)


if __name__ == "__main__":
    start_live_transcription(chunk_seconds=3)