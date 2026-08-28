import json
import sounddevice as sd

from vosk_hindi_live import create_recognizer, SAMPLE_RATE, BLOCK_SIZE
from translate_hindi_to_english import translate_hindi_to_english

def start_hindi_live_to_english():
    recognizer = create_recognizer()

    print("Hindi Live to English Translation")
    print("Speak in Hindi. Press Ctrl+C to stop.")

    def callback(indata, frames, time, status):
        if status:
            print(status)

        data = bytes(indata)

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            hindi_text = result.get("text", "").strip()

            if hindi_text:
                english_text = translate_hindi_to_english(hindi_text)
                print("Hindi   :", hindi_text)
                print("English :", english_text)

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
    start_hindi_live_to_english()