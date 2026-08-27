import whisper

model = whisper.load_model("base")


def speech_to_english(audio_path):
    result = model.transcribe(audio_path, task="translate")

    return result["text"].strip()
