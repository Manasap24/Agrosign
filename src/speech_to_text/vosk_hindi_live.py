
from pathlib import Path
from functools import lru_cache

from vosk import Model, KaldiRecognizer


SAMPLE_RATE = 16000
BLOCK_SIZE = 4096


@lru_cache(maxsize=1)
def get_hindi_model():
    base_dir = Path(__file__).resolve().parent

    model_path = (
        base_dir
        / "models"
        / "vosk-model-small-hi-0.22"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Hindi Vosk model not found at: {model_path}"
        )

    print("Loading Hindi Vosk model...")

    return Model(str(model_path))


def create_recognizer():
    return KaldiRecognizer(
        get_hindi_model(),
        SAMPLE_RATE
    )