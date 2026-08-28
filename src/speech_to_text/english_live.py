import queue
import tempfile
from functools import lru_cache

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000


@lru_cache(maxsize=1)
def get_whisper_model():
    """Load Whisper Tiny English model only once."""
    return WhisperModel(
        "tiny.en",
        device="cpu",
        compute_type="int8"
    )


def transcribe_chunk(audio: np.ndarray) -> str:
    """Transcribe one audio chunk and return English text."""
    model = get_whisper_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        write(f.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
        temp_path = f.name

    segments, _ = model.transcribe(
        temp_path,
        language="en",
        beam_size=1,
        vad_filter=True,
        temperature=0
    )

    return "".join(segment.text for segment in segments).strip()


def start_live_transcription(chunk_seconds: int = 3):
    """Continuously capture microphone audio and transcribe every few seconds."""

    audio_queue = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    print("🎤 Whisper Tiny English Live Speech-to-Text")
    print("Speak in English... Press Ctrl+C to stop.")

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
                text = transcribe_chunk(buffer)

                if text:
                    print("English:", text)

                buffer = np.empty((0, 1), dtype=np.float32)


if __name__ == "__main__":
    start_live_transcription(chunk_seconds=3)