import json
import wave
import subprocess
from pathlib import Path
from functools import lru_cache
from vosk import Model, KaldiRecognizer

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "vosk-model-small-hi-0.22"
WAV_FILE = BASE_DIR / "audio.wav"

# -------------------------------------------------
# Load model only once
# -------------------------------------------------
@lru_cache(maxsize=1)
def load_model():
    print("Loading Hindi Vosk model...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return Model(str(MODEL_PATH))

# -------------------------------------------------
# Convert audio/video to 16 kHz mono WAV
# -------------------------------------------------
def convert_to_wav(input_file: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-i", str(input_file),
            "-ar", "16000",
            "-ac", "1",
            "-y", str(WAV_FILE),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return WAV_FILE

# -------------------------------------------------
# Transcribe WAV file
# -------------------------------------------------
def transcribe_wav(wav_file: Path) -> str:
    model = load_model()

    wf = wave.open(str(wav_file), "rb")
    recognizer = KaldiRecognizer(model, wf.getframerate())

    text = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            if result.get("text"):
                text.append(result["text"])

    final = json.loads(recognizer.FinalResult())
    if final.get("text"):
        text.append(final["text"])

    return " ".join(text).strip()

# -------------------------------------------------
# Public function
# -------------------------------------------------
def transcribe_hindi(input_file: str) -> str:
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    wav_file = convert_to_wav(input_path)
    return transcribe_wav(wav_file)

# -------------------------------------------------
# Example usage
# -------------------------------------------------
if __name__ == "__main__":
    file_path = r"C:\Users\laksh\Downloads\WhatsApp Ptt 2026-08-05 at 7.02.41 PM.ogg"

    transcript = transcribe_hindi(file_path)

    print("Hindi Transcript:")
    print(transcript)