
import re


STOPWORDS = {
    "is", "are", "the", "in", "on", "at", "of",
    "to", "and", "a", "an", "for", "with",
    "was", "were", "be", "been", "being",
    "used", "using", "use", "make", "making",
    "do", "doing", "by", "from", "into",
    "over", "under", "between",
    "his", "her", "their", "its",
    "this", "that", "these", "those",
    "get", "getting", "has", "have", "had",
    "will", "would", "could", "should", "may", "might",
}


# SIMPLE STEMMER
# ---------------------------------------------------
def simple_stem(word):
    suffixes = ["ing", "tion", "ion", "ation", "ed", "er", "s", "es"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


# TEXT PREPROCESSING
# ---------------------------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]


def get_variants(word):
    stemmed = simple_stem(word)
    variants = [word]
    if stemmed != word:
        variants.append(stemmed)
    return variants
