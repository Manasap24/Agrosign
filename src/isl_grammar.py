"""
isl_grammar.py
Rule-based English -> ISL (Indian Sign Language) gloss ordering.

Pipeline position:
    NLP Preprocessing -> isl_grammar.transform() -> Embedding / Process Matching

Rules applied:
    1. Remove articles, auxiliary/linking verbs, prepositions
    2. Reduce words to root form (lemma)
    3. Reorder to  TIME + SUBJECT/OBJECT + VERB  (SOV)
    4. Put negation, then question word, at the end

No external dependencies. Extend DOMAIN_VERBS for your agriculture dataset.
"""

import re

# ---------- word lists ----------
ARTICLES = {"a", "an", "the"}
AUXILIARIES = {
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "will", "shall", "would", "should",
    "can", "could", "may", "might", "must", "have", "has", "had",
}
PREPOSITIONS = {
    "in", "on", "at", "to", "of", "for", "with", "by", "from",
    "into", "onto", "over", "under", "about", "during", "using",
}
FILLERS = {"please", "kindly", "so", "then", "also", "just", "very"}

TIME_WORDS = {
    "morning", "afternoon", "evening", "night", "today", "tomorrow",
    "yesterday", "now", "daily", "weekly", "monthly", "tonight",
}
QUESTION_WORDS = {"how", "what", "when", "where", "why", "who", "which"}
NEGATIONS = {"not", "no", "never", "cannot"}

# Domain verbs (lemma form). Add the action_lemmas from process_metadata.csv.
DOMAIN_VERBS = {
    "apply", "spray", "sow", "plant", "water", "irrigate", "harvest",
    "plough", "plow", "weed", "fertilize", "prune", "cut", "dig",
    "store", "dry", "sell", "buy", "protect", "treat", "grow", "pick",
    "mix", "pour", "spread", "transplant", "thresh", "clean", "check",
    "use", "make", "add", "remove", "prepare", "cultivate", "feed",
}

CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "can't": "cannot", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "shouldn't": "should not",
    "couldn't": "could not", "wouldn't": "would not",
}

IRREGULAR = {
    "ran": "run", "grew": "grow", "grown": "grow", "sold": "sell",
    "bought": "buy", "dug": "dig", "made": "make", "fed": "feed",
    "children": "child", "men": "man", "women": "woman",
    "leaves": "leaf", "knives": "knife",
}


# ---------- helpers ----------
def lemmatize(word: str) -> str:
    """Very small rule-based lemmatizer (swap for NLTK/spaCy if you prefer)."""
    w = word.lower()
    if w in IRREGULAR:
        return IRREGULAR[w]
    if w in DOMAIN_VERBS or len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        stem = w[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:      # planning -> plan
            stem = stem[:-1]
        return stem if stem in DOMAIN_VERBS else (stem + "e" if stem + "e" in DOMAIN_VERBS else stem)
    if w.endswith("ed") and len(w) > 4:
        stem = w[:-2]
        if stem in DOMAIN_VERBS:
            return stem
        if stem + "e" in DOMAIN_VERBS:                   # irrigated -> irrigate
            return stem + "e"
        if w.endswith("ied"):
            return w[:-3] + "y"
        return stem
    if w.endswith("es") and len(w) > 4 and w[:-2] in DOMAIN_VERBS | {"crop"}:
        return w[:-2]
    if w.endswith("es") and w[:-2].endswith(("s", "x", "ch", "sh")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    text = text.lower().strip()
    for c, full in CONTRACTIONS.items():
        text = text.replace(c, full)
    return re.findall(r"[a-z]+", text)


# ---------- main transformation ----------
def transform(sentence: str, tense_marker: bool = False) -> list[str]:
    """
    Convert an English sentence into an ordered list of ISL gloss tokens.

    >>> transform("How to apply fertilizer in the morning?")
    ['MORNING', 'FERTILIZER', 'APPLY', 'HOW']
    """
    tokens = tokenize(sentence)

    past = any(t in {"was", "were", "did", "had"} or (t.endswith("ed") and len(t) > 4)
               for t in tokens)
    future = "will" in tokens or "shall" in tokens

    time_part, content, verbs, negation, question = [], [], [], [], []

    for tok in tokens:
        if tok in ARTICLES or tok in AUXILIARIES or tok in PREPOSITIONS or tok in FILLERS:
            continue
        if tok in NEGATIONS:
            negation.append("NOT")
            continue
        if tok in QUESTION_WORDS:
            question.append(tok.upper())
            continue
        if tok in TIME_WORDS:
            time_part.append(tok.upper())
            continue

        lemma = lemmatize(tok)
        if lemma in DOMAIN_VERBS:
            verbs.append(lemma.upper())
        else:
            content.append(lemma.upper())

    # ISL shows tense with time signs instead of verb endings
    if tense_marker and not time_part:
        if past:
            time_part.append("BEFORE")
        elif future:
            time_part.append("FUTURE")

    # remove duplicates while keeping order
    def dedupe(items):
        seen, out = set(), []
        for i in items:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    negation = dedupe(negation)
    return time_part + content + verbs + negation + question


def to_sign_string(sentence: str, sep: str = " | ") -> str:
    """Return tokens joined like your sign_sequence column: 'sign1 | sign2'."""
    return sep.join(t.lower() for t in transform(sentence))


# ---------- demo / quick tests ----------
if __name__ == "__main__":
    tests = {
        "How to apply fertilizer in the morning?": "MORNING FERTILIZER APPLY HOW",
        "Farmer sprays pesticide": "FARMER PESTICIDE SPRAY",
        "Do not water the crop": "CROP WATER NOT",
        "Sow seeds in the morning": "MORNING SEED SOW",
        "When should I harvest wheat?": "I WHEAT HARVEST WHEN",
        "Don't spray pesticide today": "TODAY PESTICIDE SPRAY NOT",
    }
    ok = 0
    for sentence, expected in tests.items():
        got = " ".join(transform(sentence))
        flag = "OK " if got == expected else "FAIL"
        ok += got == expected
        print(f"[{flag}] {sentence!r:45} -> {got}")
    print(f"\n{ok}/{len(tests)} passed")

    print("\nsign_sequence format:", to_sign_string("How to apply fertilizer in the morning?"))