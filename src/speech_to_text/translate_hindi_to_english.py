# from functools import lru_cache
# from deep_translator import GoogleTranslator

# # -------------------------------------------------
# # Load translator only once
# # -------------------------------------------------
# @lru_cache(maxsize=1)
# def load_translator():
#     print("Loading Hindi -> English translator...")
#     return GoogleTranslator(source="auto", target="en")

# # -------------------------------------------------
# # Translate Hindi text to English
# # -------------------------------------------------
# def translate_hindi_to_english(text: str) -> str:
#     if not text or not text.strip():
#         return ""

#     translator = load_translator()
#     return translator.translate(text)

# # -------------------------------------------------
# # Example usage
# # -------------------------------------------------
# if __name__ == "__main__":
#     hindi_text = (
#         "भारी वर्षा के बाद कपास के खेत में पोषक तत्वों की कमी के लक्षण दिखाई देने लगे। "
#         "मिट्टी की जांच से नाइट्रोजन और पोटाश के निम्न स्तर का पता चला। "
#         "किसान ने संतुलित उर्वरक का प्रयोग किया और खेत के चारों ओर जल निकासी की व्यवस्था में सुधार किया।"
#     )

#     english_text = translate_hindi_to_english(hindi_text)

#     print("Hindi:")
#     print(hindi_text)
#     print("\\nEnglish:")
#     print(english_text)

import random
import time
from functools import lru_cache

from deep_translator import GoogleTranslator, MyMemoryTranslator


# -------------------------------------------------
# Load translators only once (primary + fallback)
# -------------------------------------------------
@lru_cache(maxsize=1)
def load_google_translator():
    print("Loading Google translator (primary)...")
    return GoogleTranslator(source="auto", target="en")


@lru_cache(maxsize=1)
def load_fallback_translator():
    print("Loading MyMemory translator (fallback)...")
    return MyMemoryTranslator(source="hi-IN", target="en-GB")


# -------------------------------------------------
# Simple request throttle — avoid firing translation
# calls too close together, which is what triggers
# Google's rate limiting in the first place.
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
    """
    Detect when a translation provider's scraping endpoint returned an
    HTML error page instead of a real translation (rate limiting / block).
    """
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
    """Attempt one translation call; return None on failure or bad response."""
    try:
        result = translator.translate(text)

        if _looks_like_error_page(result):
            return None

        return result

    except Exception as e:
        print(f"Translator error ({translator.__class__.__name__}): {e}")
        return None


# -------------------------------------------------
# Translate Hindi text to English
# -------------------------------------------------
def translate_hindi_to_english(
    text: str,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> str:
    if not text or not text.strip():
        return ""

    google = load_google_translator()

    # --- Try Google first, with throttled retries + backoff ---
    for attempt in range(1, max_retries + 1):
        _throttle()

        result = _try_translate(google, text)

        if result is not None:
            return result

        print(f"Google translate attempt {attempt} failed, retrying...")

        # Exponential backoff with a little jitter so retries don't
        # all line up at exactly the same interval.
        delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
        time.sleep(delay)

    # --- Google exhausted its retries — fall back to MyMemory ---
    print("Google translate failed after retries, trying fallback provider...")

    try:
        fallback = load_fallback_translator()
        _throttle()
        result = _try_translate(fallback, text)

        if result is not None:
            return result

    except Exception as e:
        print(f"Fallback translator failed: {e}")

    # --- Both providers failed — fail safe, don't return garbage ---
    print("All translation attempts failed for this chunk.")
    return ""


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
    print("\nEnglish:")
    print(english_text)