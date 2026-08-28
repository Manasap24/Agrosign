from functools import lru_cache
from deep_translator import GoogleTranslator

# -------------------------------------------------
# Load translator only once
# -------------------------------------------------
@lru_cache(maxsize=1)
def load_translator():
    print("Loading Hindi -> English translator...")
    return GoogleTranslator(source="auto", target="en")

# -------------------------------------------------
# Translate Hindi text to English
# -------------------------------------------------
def translate_hindi_to_english(text: str) -> str:
    if not text or not text.strip():
        return ""

    translator = load_translator()
    return translator.translate(text)

# -------------------------------------------------
# Example usage
# -------------------------------------------------
if __name__ == "__main__":
    hindi_text = (
        "भारी वर्षा के बाद कपास के खेत में पोषक तत्वों की कमी के लक्षण दिखाई देने लगे। "
        "मिट्टी की जांच से नाइट्रोजन और पोटाश के निम्न स्तर का पता चला। "
        "किसान ने संतुलित उर्वरक का प्रयोग किया और खेत के चारों ओर जल निकासी की व्यवस्था में सुधार किया।"
    )

    english_text = translate_hindi_to_english(hindi_text)

    print("Hindi:")
    print(hindi_text)
    print("\\nEnglish:")
    print(english_text)