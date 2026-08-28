# import subprocess
# import shutil
# from pathlib import Path
# from functools import lru_cache
# from faster_whisper import WhisperModel

# BASE_DIR = Path(__file__).resolve().parent
# WAV_FILE = BASE_DIR / "audio.wav"


# def check_ffmpeg():
#     """Fail fast with a clear message if ffmpeg isn't on PATH."""
#     if shutil.which("ffmpeg") is None:
#         raise RuntimeError(
#             "ffmpeg not found on PATH. Install it and add it to PATH.\n"
#             "Easiest on Windows: 'winget install ffmpeg' or 'choco install ffmpeg', "
#             "then restart VS Code/terminal so PATH updates."
#         )


# @lru_cache(maxsize=1)
# def load_model():
#     print("Loading Faster-Whisper Small model...")
#     return WhisperModel(
#         "small",
#         device="cpu",
#         compute_type="int8"
#     )


# def convert_to_wav(input_file: Path) -> Path:
#     if not input_file.exists():
#         raise FileNotFoundError(f"Input file not found: {input_file}")

#     check_ffmpeg()

#     result = subprocess.run(
#         [
#             "ffmpeg",
#             "-i", str(input_file),
#             "-ar", "16000",
#             "-ac", "1",
#             "-y", str(WAV_FILE),
#         ],
#         capture_output=True,   # <-- capture instead of DEVNULL
#         text=True,
#     )

#     if result.returncode != 0:
#         raise RuntimeError(
#             f"ffmpeg failed (exit code {result.returncode}).\n"
#             f"--- ffmpeg stderr ---\n{result.stderr}"
#         )

#     if not WAV_FILE.exists() or WAV_FILE.stat().st_size == 0:
#         raise RuntimeError("ffmpeg ran but produced no/empty audio.wav")

#     return WAV_FILE


# def transcribe_english(input_file: str) -> str:
#     wav_file = convert_to_wav(Path(input_file))
#     model = load_model()

#     segments, info = model.transcribe(
#         str(wav_file),
#         language="en",
#         beam_size=1,
#         vad_filter=True,
#         temperature=0
#     )

#     segments = list(segments)  # materialize so we can inspect
#     if not segments:
#         print("WARNING: No segments returned — check audio.wav plays correctly "
#               "and actually contains speech.")

#     text = "".join(segment.text for segment in segments)
#     return text.strip()


# if __name__ == "__main__":
#     file_path = r"C:\Users\laksh\Downloads\WhatsApp Ptt 2026-08-05 at 7.31.30 PM.ogg"

#     try:
#         transcript = transcribe_english(file_path)
#         print("English Transcript:")
#         print(transcript)
#     except Exception as e:
#         print(f"ERROR: {e}")

import subprocess
from pathlib import Path
from functools import lru_cache
from faster_whisper import WhisperModel

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
WAV_FILE = BASE_DIR / "audio.wav"

# -------------------------------------------------
# Load Whisper Tiny model only once
# -------------------------------------------------
@lru_cache(maxsize=1)
def load_model():
    print("Loading Whisper Tiny English model...")
    return WhisperModel(
        "tiny",            # very fast on CPU
        device="cpu",
        compute_type="int8"
    )

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

    segments, info = model.transcribe(
        str(wav_file),
        language="en",      # force English
        beam_size=1,
        vad_filter=True,
        temperature=0,
    )

    text = "".join(segment.text for segment in segments)
    return text.strip()

# -------------------------------------------------
# Public function: transcribe any audio/video file
# -------------------------------------------------
def transcribe_english(input_file: str) -> str:
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    wav_file = convert_to_wav(input_path)
    return transcribe_wav(wav_file)

# -------------------------------------------------
# Example usage
# -------------------------------------------------
if __name__ == "__main__":
    file_path = r"C:\Users\laksh\Downloads\WhatsApp Ptt 2026-08-05 at 7.31.30 PM.ogg"
    transcript = transcribe_english(file_path)

    print("English Transcript:")
    print(transcript)