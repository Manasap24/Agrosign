import random
import time
from functools import lru_cache

from deep_translator import MyMemoryTranslator


# -------------------------------------------------
# Load translator only once
# -------------------------------------------------
@lru_cache(maxsize=1)
def load_translator():
    print("Loading MyMemory translator (kn-IN -> en-GB)...")
    return MyMemoryTranslator(source="kn-IN", target="en-GB")


# -------------------------------------------------
# Simple request throttle
# -------------------------------------------------
_last_request_time = 0.0
_MIN_INTERVAL = 0.4  # seconds between requests, tune if needed


def _throttle():
    global _last_request_time

    elapsed = time.time() - _last_request_time
    wait = _MIN_INTERVAL - elapsed

    if wait > 0:
        time.sleep(wait)

    _last_request_time = time.time()


def _looks_like_error_page(text: str) -> bool:
    if not text:
        return True

    lowered = text.lower()

    error_markers = [
        "<html",
        "error 500",
        "error 429",
        "that\u2019s an error",
        "that's an error",
        "that\u2019s all we know",
        "that's all we know",
    ]

    return any(marker in lowered for marker in error_markers)


def _try_translate(translator, text: str) -> str | None:
    try:
        result = translator.translate(text)

        if _looks_like_error_page(result):
            return None

        return result

    except Exception as e:
        print(f"Translator error (MyMemoryTranslator): {e}")
        return None


# -------------------------------------------------
# Translate Kannada text to English
# -------------------------------------------------
def translate_kannada_to_english(
    text: str,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> str:
    if not text or not text.strip():
        return ""

    translator = load_translator()

    for attempt in range(1, max_retries + 1):
        _throttle()

        result = _try_translate(translator, text)

        if result is not None:
            return result

        print(f"MyMemory translate attempt {attempt} failed, retrying...")

        delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
        time.sleep(delay)

    print("All translation attempts failed for this chunk.")
    return ""


# -------------------------------------------------
# Example usage
# -------------------------------------------------
if __name__ == "__main__":
    kannada_text = (
        "ಭಾರೀ ಮಳೆಯ ನಂತರ ಹತ್ತಿ ಹೊಲದಲ್ಲಿ ಪೋಷಕಾಂಶಗಳ ಕೊರತೆಯ ಲಕ್ಷಣಗಳು ಕಾಣಿಸಿಕೊಂಡವು. "
        "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯಿಂದ ಸಾರಜನಕ ಮತ್ತು ಪೊಟ್ಯಾಶ್‌ನ ಕಡಿಮೆ ಮಟ್ಟ ಪತ್ತೆಯಾಯಿತು. "
        "ರೈತನು ಸಮತೋಲಿತ ಗೊಬ್ಬರವನ್ನು ಬಳಸಿದನು ಮತ್ತು ಹೊಲದ ಸುತ್ತಲಿನ ಒಳಚರಂಡಿ ವ್ಯವಸ್ಥೆಯನ್ನು ಸುಧಾರಿಸಿದನು."
    )

    english_text = translate_kannada_to_english(kannada_text)

    print("Kannada:")
    print(kannada_text)
    print("\nEnglish:")
    print(english_text)