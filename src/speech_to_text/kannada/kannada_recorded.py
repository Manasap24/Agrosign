import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import onnx_asr


@lru_cache(maxsize=1)
def get_kannada_model():
    """Load the Kannada ASR model only once (cached across calls)."""
    print("Loading Kannada ASR model (IndicConformer)...")
    return onnx_asr.load_model("OpenVoiceOS/ai4bharat-indicconformer-kn-onnx")


def _result_to_text(result) -> str:
    """
    onnx_asr's recognize() can return either a plain string or a
    result object with a .text attribute, depending on model/version.
    """
    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    text = getattr(result, "text", None)

    if text is not None:
        return str(text).strip()

    return str(result).strip()


def _convert_to_wav_16k_mono(input_path: str) -> str:
    """
    Convert any audio/video file to 16kHz mono WAV using ffmpeg.

    librosa/soundfile can only reliably read WAV/FLAC/OGG-style files --
    they fail on .mp4, .m4a, .webm, and many other containers users
    might actually upload (the UI accepts "Audio or Video File").
    ffmpeg handles essentially any format, so we always normalize
    through it rather than relying on librosa's format support.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. Install it "
            "(e.g. 'winget install ffmpeg' on Windows, or download "
            "from https://ffmpeg.org/download.html) and make sure "
            "the 'ffmpeg' command works from a new terminal."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name

    command = [
        "ffmpeg",
        "-y",  # overwrite output without prompting
        "-i", input_path,
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",  # mono
        "-f", "wav",
        output_path,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg failed to convert audio: {result.stderr[-500:]}"
        )

    return output_path


def transcribe_kannada(file_path: str) -> str:
    """
    Transcribe an uploaded Kannada audio/video file and return the
    recognized Kannada text. Handles any format ffmpeg can read.
    """
    model = get_kannada_model()

    wav_path = _convert_to_wav_16k_mono(file_path)

    try:
        result = model.recognize(wav_path)
        return _result_to_text(result)

    finally:
        Path(wav_path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_path = "kannada_16k.wav"
    text = transcribe_kannada(test_path)
    print("Kannada:")
    print(text)