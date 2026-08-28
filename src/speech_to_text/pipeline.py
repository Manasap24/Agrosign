from vosk_hindi import transcribe_hindi
from translate_hindi_to_english import translate_hindi_to_english

def process_audio(audio_path: str):
    # Step 1: Hindi speech -> Hindi text
    hindi_text = transcribe_hindi(audio_path)

    # Step 2: Hindi text -> English text
    english_text = translate_hindi_to_english(hindi_text)

    return hindi_text, english_text

if __name__ == "__main__":
    audio_file = r"C:\Users\laksh\Downloads\WhatsApp Ptt 2026-08-05 at 7.02.41 PM.ogg"

    hindi, english = process_audio(audio_file)

    print("Hindi Transcript:")
    print(hindi)

    print("\\nEnglish Translation:")
    print(english)