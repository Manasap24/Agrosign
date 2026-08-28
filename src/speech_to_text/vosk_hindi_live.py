import json
from pathlib import Path
from functools import lru_cache

import sounddevice as sd
from vosk import Model, KaldiRecognizer

SAMPLE_RATE = 16000
BLOCK_SIZE = 8000


@lru_cache(maxsize=1)
def get_hindi_model():
    """Load the Hindi Vosk model only once."""
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "vosk-model-small-hi-0.22"
    return Model(str(model_path))


def create_recognizer():
    """Create a recognizer using the cached model."""
    return KaldiRecognizer(get_hindi_model(), SAMPLE_RATE)


def start_hindi_live_transcription():
    """Start real-time Hindi speech-to-text from the microphone."""

    recognizer = create_recognizer()

    print("🎤 Hindi Live Speech-to-Text")
    print("Speak in Hindi... Press Ctrl+C to stop.")

    def callback(indata, frames, time, status):
        if status:
            print(status)

        data = bytes(indata)

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
                print("Hindi:", text)
        else:
            partial = json.loads(recognizer.PartialResult())
            if partial.get("partial"):
                print("Listening:", partial["partial"])

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        while True:
            pass


if __name__ == "__main__":
    start_hindi_live_transcription()