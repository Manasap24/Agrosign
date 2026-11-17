"""
isl_sequence.py
Turns detector events into an ISL-ordered keyword list for video lookup.

Order per event:
    TIME + AGENT + PATIENT + PROCESS/VARIANT signs + NOT/CESSATION/HYPOTHETICAL + QUESTION

Every word goes through the same RESOLVER chain, so missing words are handled
in one place:
    1. exact keyword / synonym / simple plural      (agro_terms.csv)
    2. fallback table: unknown word -> broader keyword (barley -> crops)
    3. fuzzy spelling match (flagged as approximate)
    4. not found -> reported in result["missing"] (and optionally logged)
"""

import re
import difflib

from isl_grammar import DOMAIN_VERBS, NEGATIONS, transform as transform_isl_gloss

TIME_WORDS = {"morning", "afternoon", "evening", "night", "today", "tomorrow",
              "yesterday", "now", "daily", "weekly", "monthly"}
QUESTION_WORDS = {"how", "what", "when", "where", "why", "who", "which"}
END_MARKERS = {"NEGATION", "CESSATION", "HYPOTHETICAL"}   # signed after the process
# Keywords/synonyms that are also everyday English words. They are NOT picked up when
# scanning running text ("well-drained field" must not sign a water well). They still
# work when they are the detected agent/object or a curated concept. Edit freely.
SCAN_EXCLUDE = {"well", "gram", "ram", "iron", "natural", "growing", "collecting",
                "separating", "grading", "cold", "heat", "spray"}
# Keywords the sign generator uses for NEGATION / CESSATION / HYPOTHETICAL. They are added
# from the event's polarity/status (at the end), never picked up from running text.
MODIFIER_KEYWORDS = {"not", "stop", "may"}
MODIFIER_CONCEPTS = {"negation", "cessation", "hypothetical"}
SKIP_WORDS = {"the", "a", "an", "it", "them", "this", "that",
              "i", "we", "you", "he", "she", "they", "one"}   # no sign for these

# unknown word -> broader keyword that exists in agro_terms.csv.
# Used only if the target keyword exists. Extend via term_fallbacks.csv.
DEFAULT_FALLBACKS = {
    **{w: "crops" for w in ["barley", "oats", "pulses", "lentil", "chickpea", "soybean",
                            "groundnut", "mustard", "sunflower", "jute", "tea", "coffee"]},
    "bajra": "millet", "jowar": "millet", "sorghum": "millet",
    **{w: "vegetable" for w in ["potato", "tomato", "onion", "brinjal", "cabbage", "carrot"]},
    **{w: "fruit" for w in ["mango", "banana", "apple", "orange", "grapes", "papaya"]},
    **{w: "fertilizer" for w in ["urea", "dap", "npk"]},
    **{w: "pesticide" for w in ["fungicide", "herbicide", "insecticide"]},
    **{w: "cattle" for w in ["bull", "ox", "calf", "sheep", "livestock"]},
    "land": "soil",
}


def _compact(token: str) -> str:
    """'Drip_Irrigation' / 'drip irrigation' -> 'dripirrigation'."""
    return re.sub(r"[^a-z0-9]", "", str(token).lower())


def build_vocabulary(csv_path) -> dict[str, str]:
    """agro_terms.csv -> {compact keyword or synonym: canonical keyword}."""
    import pandas as pd
    df = pd.read_csv(csv_path).fillna("")
    vocab: dict[str, str] = {}
    for _, row in df.iterrows():
        keyword = str(row["keyword"]).strip().lower()
        for syn in str(row["synonyms"]).split("|"):
            if syn.strip():
                vocab.setdefault(_compact(syn), keyword)
    for _, row in df.iterrows():
        keyword = str(row["keyword"]).strip().lower()
        vocab[_compact(keyword)] = keyword          # keywords beat synonyms
    return vocab


def load_fallbacks(csv_path) -> dict[str, str]:
    """Optional dataset/term_fallbacks.csv with columns: word,keyword"""
    import pandas as pd
    df = pd.read_csv(csv_path).fillna("")
    return {str(r["word"]).strip().lower(): str(r["keyword"]).strip().lower()
            for _, r in df.iterrows()}


class Resolver:
    def __init__(self, vocab: dict[str, str], fallbacks: dict[str, str] | None = None,
                 fuzzy: bool = True):
        self.vocab = vocab
        self.keywords = set(vocab.values())
        merged = {**DEFAULT_FALLBACKS, **(fallbacks or {})}
        self.fallbacks = {_compact(k): v for k, v in merged.items()}
        self.fuzzy = fuzzy
        self.approximate: list[tuple[str, str, str]] = []   # (word, keyword, how)

    @staticmethod
    def _forms(t: str, spot: bool = False):
        forms = [t, t[:-1] if t.endswith("s") else "", t[:-2] if t.endswith("es") else ""]
        if not spot:                                  # "crop" -> "crops" (only for direct lookups)
            forms += [t + "s", t + "es"]
        return [c for c in forms if c]

    def resolve(self, token: str, spot: bool = False) -> str | None:
        """spot=True: strict mode for scanning running text (no spelling guesses)."""
        t = _compact(token)
        for c in self._forms(t, spot):                            # 1. exact / synonym
            if c in self.vocab:
                return self.vocab[c]
        for c in self._forms(t, spot):                            # 2. broader keyword
            kw = self.fallbacks.get(c)
            if kw and kw in self.keywords:
                self.approximate.append((token, kw, "fallback"))
                return kw
        if self.fuzzy and not spot and len(t) >= 6:               # 3. spelling variant
            close = difflib.get_close_matches(t, list(self.vocab), n=1, cutoff=0.9)
            if close:
                kw = self.vocab[close[0]]
                self.approximate.append((token, kw, "fuzzy"))
                return kw
        return None                                               # 4. missing


def _as_resolver(vocabulary) -> "Resolver | None":
    if vocabulary is None or isinstance(vocabulary, Resolver):
        return vocabulary
    if isinstance(vocabulary, dict):
        return Resolver(vocabulary)
    return Resolver({_compact(v): v for v in vocabulary})         # plain set of keywords


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text).lower())


def map_words(words: list[str], resolver: "Resolver", missing: list | None = None,
              spot: bool = False, skip: set[str] = SKIP_WORDS) -> list[str]:
    """
    Words -> keywords, longest phrase first ("crop rotation" before "crop").
    spot=True : scan running text; words without a sign are ignored silently.
    spot=False: words without a sign are added to `missing`.
    """
    out, i = [], 0
    while i < len(words):
        matched = False
        for n in (3, 2):                                          # multi-word keywords
            if i + n <= len(words):
                phrase = _compact("".join(words[i:i + n]))
                if phrase in resolver.vocab:
                    out.append(resolver.vocab[phrase])
                    i += n
                    matched = True
                    break
        if matched:
            continue
        w = words[i]
        i += 1
        if w in skip or (spot and w in SCAN_EXCLUDE):
            continue
        m = resolver.resolve(w, spot=spot)
        if spot and m in MODIFIER_KEYWORDS:
            continue
        if m:
            out.append(m)
        elif not spot and missing is not None and w not in missing:
            missing.append(w)
    return out


def _resolve_concepts(concepts: list[str], resolver: "Resolver", missing: list | None,
                      keep_unresolved: bool) -> list[str]:
    """Curated concepts: exact keyword/synonym only (no fallback or spelling guesses)."""
    out = []
    for c in concepts:
        m = resolver.vocab.get(_compact(c))
        if m:
            out.append(m)
            continue
        if missing is not None and c not in missing:
            missing.append(c)
        if keep_unresolved:
            out.append(c)
    return out


def _keep_last(tokens: list[str]) -> list[str]:
    """Remove repeats, keeping the LAST occurrence (process signs end up after their objects)."""
    ordered = []
    for t in reversed(tokens):
        if t not in ordered:
            ordered.append(t)
    return ordered[::-1]


def _keep_first(tokens: list[str]) -> list[str]:
    """Deduplicate context signs without moving them later in the gloss."""
    seen = set()
    ordered = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def event_to_isl(event: dict, vocabulary=None, missing: list | None = None,
                 keep_unresolved_concepts: bool = False, omit: set[str] | None = None) -> list[str]:
    """
    Order: TIME + AGENT + OBJECTS + PROCESS signs + NOT/CESSATION/HYPOTHETICAL + QUESTION

    Every word in the clause that has a sign (wheat, rice, crop rotation ...) is
    picked up by keyword scanning, so it does not depend on the detector's
    grammar roles (subject/object/conjunction).
    """
    resolver = _as_resolver(vocabulary)
    evidence_words = _words(event.get("evidence", ""))

    # keep_unresolved_concepts=True : a curated metadata concept with no keyword/video is
    #   kept in the output (so the caller can report it as a missing sign).
    # omit : canonical keywords to leave out (e.g. the agent when it has not changed).

    time_tokens = [w for w in evidence_words if w in TIME_WORDS]
    question = [w for w in evidence_words[:2] if w in QUESTION_WORDS]
    agent = [w for w in _words(event.get("agent", "")) if w not in SKIP_WORDS]
    patient = [w for w in _words(event.get("patient", "")) if w not in SKIP_WORDS]

    concepts = [c.lower() for c in event.get("sign_concepts", [])]
    front = [c for c in concepts if c.upper() not in END_MARKERS]
    back = [c for c in concepts if c.upper() in END_MARKERS]      # NEGATION etc. go last

    if resolver is None:
        return _keep_last(time_tokens + agent + patient + front + back + question)

    # time/question words: signed only if a sign exists (no "missing" noise)
    time_signs = [m for m in (resolver.resolve(t, spot=True) for t in time_tokens) if m]
    q_signs = [m for m in (resolver.resolve(t, spot=True) for t in question) if m]

    agent_signs = map_words(agent, resolver, missing)
    patient_signs = map_words(patient, resolver, missing)
    scanned = map_words(evidence_words, resolver, spot=True)      # all other signed words
    concept_signs = _resolve_concepts(front, resolver, missing, keep_unresolved_concepts)
    end_signs = _resolve_concepts(back, resolver, missing, keep_unresolved_concepts)

    # words already signed as the agent are not repeated
    scanned = [t for t in scanned if t not in agent_signs]
    ordered = _keep_last(time_signs + agent_signs + patient_signs + scanned
                         + concept_signs + end_signs + q_signs)
    if omit:
        ordered = [t for t in ordered if t not in omit]
    return ordered


def event_context_signs(event: dict, vocabulary, missing: list | None = None,
                        omit: set[str] | None = None) -> dict:
    """
    For callers that keep the detector's process ids and let sign_sequence_generator
    expand them. Returns the signs that belong BEFORE and AFTER the process signs:
        {"pre":  TIME + AGENT + OBJECTS (keywords, lowercase),
         "post": QUESTION word signs}
    omit: keywords to leave out (repeated agent, signs already inside the process).
    """
    resolver = _as_resolver(vocabulary)
    evidence = str(event.get("evidence", ""))
    # A purpose phrase names a possible/future action, not an action asserted
    # by this event. Do not let "prepare the field for sowing" add PLANTING to
    # the preparation event; the sowing event, when present, supplies it.
    evidence_for_context = re.sub(
        r"\b(?:for|to)\s+(?:the\s+)?(?:sowing|sow|planting|plant)\b"
        r"(?:\s+(?:of\s+)?(?:the\s+)?(?:seeds?|crops?|field))?",
        " ", evidence, flags=re.I,
    )
    evidence_words = _words(evidence_for_context)
    time_tokens = [w for w in evidence_words if w in TIME_WORDS]
    question = [w for w in evidence_words[:2] if w in QUESTION_WORDS]
    agent = [w for w in _words(event.get("agent", "")) if w not in SKIP_WORDS]
    patient = [w for w in _words(event.get("patient", "")) if w not in SKIP_WORDS]

    time_signs = [m for m in (resolver.resolve(t, spot=True) for t in time_tokens) if m]
    q_signs = [m for m in (resolver.resolve(t, spot=True) for t in question) if m]
    agent_signs = map_words(agent, resolver, missing)
    patient_signs = map_words(patient, resolver, missing)
    # Keep action words out of the context/topic portion. The detected process
    # concepts below represent the action and belong after the objects.
    action_terms = set(DOMAIN_VERBS)
    action = str(event.get("action", "")).strip().lower()
    if action:
        action_terms.add(action)
    action_signs = {sign for word in action_terms
                    if (sign := resolver.resolve(word, spot=True))}
    scanned = [
        t for t in map_words(evidence_words, resolver, spot=True)
        if t not in agent_signs
        and t not in action_signs
        and t not in DOMAIN_VERBS
        and t not in TIME_WORDS
        and t not in QUESTION_WORDS
        and t not in NEGATIONS
        and t not in MODIFIER_KEYWORDS
        and t not in MODIFIER_CONCEPTS
    ]

    # Keep first occurrence: keep-last could move an object to the end of the
    # clause when it is repeated, breaking TIME + AGENT + OBJECT + PROCESS.
    pre = _keep_first(time_signs + agent_signs + patient_signs + scanned)
    if omit:
        pre = [t for t in pre if t not in omit]
        q_signs = [t for t in q_signs if t not in omit]
    return {"pre": pre, "post": q_signs}


def events_to_isl(events: list[dict], vocabulary=None, missing: list | None = None) -> list[str]:
    out = []
    for event in sorted(events, key=lambda e: e.get("sequence_index", 0)):
        out.extend(event_to_isl(event, vocabulary, missing))
    return out


def log_missing(words: list[str], path: str = "missing_terms.csv") -> None:
    """Append unresolved words; review this file and add rows to agro_terms.csv."""
    if not words:
        return
    with open(path, "a", encoding="utf-8") as f:
        for w in words:
            f.write(f"{w}\n")


def sentence_to_isl(text: str, detect_fn, vocabulary=None, log_path: str | None = None) -> dict:
    """
    Returns {"tokens", "source": "events"|"rules", "missing", "approximate"}
    approximate = words matched via fallback/fuzzy, worth a quick human check.
    """
    resolver = _as_resolver(vocabulary)
    if resolver is not None:
        resolver.approximate = []
    missing: list[str] = []
    events = detect_fn(text).get("events", [])

    if events:
        tokens, source = events_to_isl(events, resolver, missing), "events"
    else:
        tokens, source = [], "rules"
        light = {"use", "make", "do", "have", "get", "give", "take", "put"}   # no sign needed
        # `text_preprocessor.py` is not part of this project. Use the existing
        # rule-based grammar so direct vocabulary inputs (including one word)
        # still reach the same sign resolver.
        gloss = transform_isl_gloss(text)
        if resolver is None:
            tokens.extend(gloss)
        else:
            tokens.extend(map_words(gloss, resolver, missing, skip=SKIP_WORDS | light))
        tokens = _keep_last(tokens)

    if log_path:
        log_missing(missing, log_path)
    return {"tokens": tokens, "source": source, "missing": missing,
            "approximate": list(resolver.approximate) if resolver else []}


if __name__ == "__main__":
    vocab = {_compact(k): k for k in ["farmer", "crops", "soil", "plough", "wheat",
                                      "harvesting", "millet", "fertilizer", "farm"]}
    vocab[_compact("fertiliser")] = "fertilizer"
    r = Resolver(vocab)
    for word in ["wheat", "crop", "barley", "jowar", "urea", "fertilisers", "tractorz", "xyz"]:
        print(f"{word:12} ->", r.resolve(word))
    print("approximate:", r.approximate)
