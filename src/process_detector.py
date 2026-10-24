"""Agricultural event extraction and process recognition.

The event catalogue is deliberately small and explicit. Sentence embeddings
rank its descriptions/examples; action, object, method, polarity and temporal
cues provide the context needed to avoid word-only matches.
"""

from __future__ import annotations

import re
import logging
from functools import lru_cache

import pandas as pd
from sentence_transformers import util

from embeddings import BASE_DIR, load_model, load_processes
from event_model import AgriculturalEvent
from preprocessing import nlp as _shared_nlp
from bart_verifier import classify_processes

LOGGER = logging.getLogger(__name__)


METADATA_PATH = BASE_DIR / "dataset" / "agro_process_metadata.csv"
SCORING_CONFIG = {
    # Initial transparent heuristic weights; calibrate against a labeled set.
    "action": 0.35,
    "object": 0.20,
    "phrase": 0.20,
    "semantic": 0.25,
    "high": 0.78,
    "medium": 0.55,
    "accept": 0.48,
    "ambiguity_margin": 0.06,
    # BART is invoked only for low first-stage scores or close leading process
    # candidates. These are calibration settings, not empirically proven cutoffs.
    "bart_top_k": 3,
    "bart_trigger_score": 0.55,
    "bart_min_entailment": 0.50,
    # The curated catalogue is intentionally small; retain enough ranked
    # candidates for specific agricultural actions to beat broad neighbours.
    "top_k": 12,
    "legacy_semantic_min": 0.72,
}

ACTION_WORDS = {
    "prepare", "prepares", "prepared", "preparing", "plough", "ploughs",
    "ploughed", "plowing", "plow", "plowed", "till", "tilled", "level",
    "select", "selects", "selected", "choose", "chooses", "chose",
    "clear", "clears", "cleared", "remove", "removes", "removed",
    "add", "adds", "added", "mix", "mixes", "mixed", "incorporate",
    "incorporates", "incorporated", "raise", "raises", "raised", "shape",
    "shapes", "shaped", "form", "forms", "formed", "build", "builds",
    "built", "maintain", "maintains", "maintained", "conserve", "conserves",
    "conserved", "retain", "retains", "retained", "preserve", "preserves",
    "preserved", "transplant", "transplants", "transplanted", "move", "moves",
    "moved", "inspect", "inspects", "inspected", "check", "checks", "checked",
    "measure", "measures", "measured", "clean", "cleans", "cleaned", "drain",
    "drains", "drained", "fill", "fills", "filled", "cover", "covers", "covered",
    "prune", "prunes", "pruned", "stake", "stakes", "staked", "weed", "weeds",
    "weeded", "dig", "digs", "dug", "break", "breaks", "broke", "loosen", "loosens",
    "loosened",
    "sow", "sows", "sowed", "sowing", "plant", "plants", "planted",
    "treat", "treated", "coat", "coated", "irrigate", "irrigates",
    "irrigated", "irrigating", "water", "waters", "watered", "watering",
    "fertilize", "fertilizes", "fertilized", "fertilise", "fertilised",
    "manure", "apply", "applies", "applied", "use", "uses", "used", "using", "spray", "sprays", "sprayed",
    "sprinkle", "sprinkled", "control", "controls", "controlled", "manage",
    "managed", "harvest", "harvests", "harvested", "reap", "reaped", "collect",
    "collected", "spread",
    # Verb vocabulary used by the full process catalogue (assessment,
    # monitoring, planning, and post-harvest activities).
    "assess", "assesses", "assessed", "evaluate", "evaluates", "evaluated",
    "analyze", "analyse", "analyzes", "analyses", "analyzed", "analysed",
    "plan", "plans", "planned", "schedule", "schedules", "scheduled",
    "grow", "grows", "grew", "grown", "rear", "rears", "reared",
    "cultivate", "cultivates", "cultivated", "decompose", "decomposes", "decomposed",
    "support", "supports", "supported", "distribute", "distributes", "distributed",
    "classify", "classifies", "classified", "reduce", "reduces", "reduced",
    "minimize", "minimizes", "minimized", "estimate", "estimates", "estimated",
    "calculate", "calculates", "calculated", "sample", "samples", "sampled",
    "map", "maps", "mapped", "allocate", "allocates", "allocated",
    "organize", "organizes", "organized", "organise", "organises", "organised",
    "adapt", "adapts", "adapted", "regulate", "regulates", "regulated",
    "monitor", "monitors", "monitored", "observe", "observes", "observed",
    "scout", "scouts", "scouted", "detect", "detects", "detected",
    "identify", "identifies", "identified", "analyze", "analyse", "test", "tests", "tested",
    "cool", "cools", "cooled", "package", "packages", "packaged",
    "sell", "sells", "sold", "process", "processes", "processed",
    "thresh", "threshes", "threshed", "sieve", "sieves", "sieved",
    "tie", "ties", "tied", "guide", "guides", "guided", "pump", "pumps", "pumped",
    "conserve", "conserves", "conserved", "prevent", "prevents", "prevented",
    "handle", "handles", "handled", "repair", "repairs", "repaired",
    "diversify", "diversifies", "diversified", "intercrop", "intercrops", "intercropped",
    "rotate", "rotates", "rotated", "pollinate", "pollinates", "pollinated",
    "shade", "shades", "shaded", "warn", "warns", "warned", "record", "records", "recorded",
    "carry", "carries", "carried", "feed", "feeds", "fed", "care", "cares", "cared",
}
IRREGULAR = {
    "waters": "water", "watered": "water", "watering": "water",
    "irrigates": "irrigate", "irrigated": "irrigate", "irrigating": "irrigate",
    "sows": "sow", "sowed": "sow", "sown": "sow", "sowing": "sow",
    "plants": "plant", "planted": "plant", "planting": "plant",
    "fertilizes": "fertilize", "fertilized": "fertilize", "fertilizing": "fertilize",
    "fertilises": "fertilise", "fertilised": "fertilise", "fertilising": "fertilise",
    "ploughed": "plough", "ploughs": "plough", "ploughing": "plough",
    "plowed": "plow", "plowing": "plow", "tilled": "till", "tilling": "till",
    "treated": "treat", "treating": "treat", "sprayed": "spray", "spraying": "spray",
    "sprinkled": "sprinkle", "sprinkling": "sprinkle", "harvested": "harvest",
    "harvesting": "harvest", "reaped": "reap", "collected": "collect",
    "applies": "apply", "applied": "apply", "uses": "use", "used": "use",
    "drinks": "drink", "drank": "drink",
}
STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on", "for", "with", "by", "from", "and", "or", "that", "this", "these", "those"}


def _parts(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [part.strip().lower() for part in value.split("|") if part.strip()]


def normalize_word(word: str) -> str:
    word = word.lower().strip()
    if word in IRREGULAR:
        return IRREGULAR[word]
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [normalize_word(w) for w in re.findall(r"[a-z]+", str(text).lower()) if w not in STOP_WORDS and len(w) > 1]


@lru_cache(maxsize=1)
def _nlp():
    return _shared_nlp


def initialize_process_detector() -> None:
    """Load parser and model, then build static process vectors once."""
    _nlp()
    _catalogue()


@lru_cache(maxsize=1)
def _catalogue() -> dict:
    process_df = load_processes()
    metadata_df = pd.read_csv(METADATA_PATH).fillna("")
    try:
        model = load_model()
    except Exception as exc:
        # Keep the API usable in an offline installation without cached weights.
        # Curated action/object/phrase evidence still works; semantic-only
        # fallback candidates are disabled in this degraded mode.
        LOGGER.warning("Sentence embedding model unavailable; using lexical context scoring: %s", exc)
        model = None
    rows_by_name = {str(r["process_name"]).strip().lower(): r for _, r in process_df.iterrows()}

    profiles = []
    all_profiles = []
    row_index_by_name = {}
    for row_index, row in process_df.iterrows():
        name = str(row["process_name"]).strip()
        examples = [x.strip() for x in str(row.get("context_examples", "")).split(";") if x.strip()]
        profile_text = " ".join([name, str(row.get("description", "")), *examples])
        all_profiles.append({"process_id": str(row.get("process_id", "")),
                             "process_name": name,
                             "category": str(row.get("category", "")),
                             "description": str(row.get("description", "")),
                             "examples": examples,
                             "sign_concepts": _parts(row.get("sign_sequence", "")),
                             "profile_text": profile_text,
                             "row": row})
        row_index_by_name[name.lower()] = len(all_profiles) - 1
    all_vectors = (model.encode([p["profile_text"] for p in all_profiles], convert_to_tensor=True, normalize_embeddings=True)
                   if model is not None else None)
    for _, meta in metadata_df.iterrows():
        canonical = str(meta["process_name"]).strip()
        # Match the compact event IDs to their existing, video-mapped process rows.
        legacy_name = str(meta.get("legacy_process_name", "")).strip().lower()
        row = rows_by_name.get(legacy_name or canonical.lower())
        if row is None:
            continue
        examples = [x.strip() for x in str(row.get("context_examples", "")).split(";") if x.strip()]
        aliases = _parts(meta.get("aliases", ""))
        methods = _parts(meta.get("method_concepts", ""))
        profile_text = " ".join([canonical, str(row.get("description", "")), *examples, *aliases])
        profiles.append({
            "process_id": str(meta["process_id"]).strip(),
            "process_name": canonical,
            "parent_id": str(meta.get("parent_id", "")).strip(),
            "actions": set(_parts(meta.get("action_lemmas", ""))),
            "objects": set(_parts(meta.get("object_concepts", ""))),
            "methods": methods,
            "aliases": aliases,
            "sign_concepts": _parts(meta.get("sign_concepts", "")),
            "category": str(row.get("category", "")),
            "description": str(row.get("description", "")),
            "examples": examples,
            "sign_sequence": str(row.get("sign_sequence", "")),
            "profile_text": profile_text,
            "row": row,
            "embedding_index": row_index_by_name[canonical.lower()],
        })
    return {"profiles": profiles, "all_profiles": all_profiles,
            "vectors": all_vectors, "model": model}


def _phrase_present(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    normalized_text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    normalized_phrase = re.sub(r"[^a-z0-9 ]+", " ", phrase.lower())
    return bool(re.search(r"(?<!\w)" + re.escape(normalized_phrase.strip()) + r"(?!\w)", normalized_text))


def _phrase_is_negated(text: str, phrase: str) -> bool:
    """Do not treat a negated alternative as positive process evidence."""
    normalized = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    target = re.sub(r"[^a-z0-9 ]+", " ", phrase.lower()).strip()
    if not target:
        return False
    for match in re.finditer(r"(?<!\w)" + re.escape(target) + r"(?!\w)", normalized):
        if re.search(r"(?:\bnot|\bnever|\bno|\bneither)\s*$", normalized[:match.start()].rstrip()):
            return True
    return False


def _clause_spans(text: str) -> list[tuple[str, str]]:
    """Split coordinated events while preserving fronted temporal clauses."""
    clean = text.strip()
    if not clean:
        return []
    # Fronted temporal clauses: "After sowing..., the farmer irrigated...".
    m = re.match(r"(?i)\s*(after|before)\s+(.+?),\s*(.+)$", clean)
    if m:
        relation, subclause, main = m.groups()
        main_clauses = _clause_spans(main)
        if relation.lower() == "after":
            if main_clauses:
                first_clause, first_relation = main_clauses[0]
                main_clauses[0] = (first_clause, first_relation or "after")
            return [(subclause, "after_reference"), *main_clauses]
        if main_clauses:
            first_clause, first_relation = main_clauses[0]
            main_clauses[0] = (first_clause, first_relation or "before")
        return [*main_clauses, (subclause, "before_reference")]

    actions = sorted(ACTION_WORDS, key=len, reverse=True)
    action_lookahead = r"(?=" + "|".join(re.escape(a) for a in actions) + r"\b)"
    # Split explicit discourse sequencing and coordinated verb events, not every "and".
    pattern = re.compile(r"\s*(?:(first|then|next|finally|subsequently|afterwards|but)\b\s*|(?:,\s*)?\band\b\s*)", re.I)
    pieces = []
    pos = 0
    previous_relation = ""
    for match in pattern.finditer(clean):
        tail = clean[match.end():]
        marker = (match.group(1) or "").lower()
        # Conjunctions split only when followed by another supported action.
        first_word = re.match(r"[a-z]+", tail.lstrip(" ,"), flags=re.I)
        action_led = bool(first_word and normalize_word(first_word.group(0)) in ACTION_WORDS)
        if not marker and not (re.match(action_lookahead, tail.lstrip(" ,"), flags=re.I) or action_led):
            continue
        piece = clean[pos:match.start()].strip(" ,.")
        piece = re.sub(r"(?i)\s+and$", "", piece)
        if piece:
            pieces.append((piece, previous_relation))
        previous_relation = marker
        pos = match.end()
    last = clean[pos:].strip(" ,.")
    if last:
        pieces.append((last, previous_relation))
    if not pieces:
        return [(clean, "")]
    # Carry a prior subject into coordinated fragments when the fragment is verb-led.
    subject = ""
    whole_doc = _nlp()(clean)
    whole_root = next((t for t in whole_doc if t.dep_ == "ROOT"), None)
    whole_subject = next((t for t in whole_doc
                          if t.dep_ == "nsubj" and (whole_root is None or t.head == whole_root)), None)
    if whole_subject is not None:
        subject = whole_subject.text
    result = []
    for piece, relation in pieces:
        if not subject:
            doc = _nlp()(piece)
            root = next((t for t in doc if t.dep_ == "ROOT"), None)
            subject_token = next((t for t in doc
                                  if t.dep_ == "nsubj"
                                  and (root is None or t.head == root or t.head.dep_ == "ROOT")), None)
            if subject_token:
                root = subject_token
                while root.head != root and root.head.pos_ not in {"VERB", "AUX"}:
                    root = root.head
                subject = piece[subject_token.idx:subject_token.idx + len(subject_token.text)]
        starts_with_subject = bool(subject and re.match(
            r"(?i)^(?:(?:the|a|an)\s+)?" + re.escape(subject) + r"\b", piece
        ))
        if subject and not starts_with_subject and any(_phrase_present(piece, a) for a in actions):
            piece = f"{subject} {piece}"
        result.append((piece, relation))
    return result


def split_action_clauses(sentence: str) -> list[str]:
    """Compatibility helper retained for callers of the former detector."""
    return [text for text, _ in _clause_spans(sentence)]


def _event_features(clause: str) -> dict:
    doc = _nlp()(clause)
    tokens = [t for t in doc if not t.is_space and not t.is_punct]
    lemmas = {normalize_word(t.lemma_.lower()) for t in tokens}
    words = {normalize_word(t.text.lower()) for t in tokens}
    verb_tokens = [t for t in tokens if t.pos_ == "VERB"]
    domain_verbs = [t for t in verb_tokens if normalize_word(t.lemma_.lower()) in ACTION_WORDS]
    # Prefer a specific agricultural action over a generic "use" construction.
    first_lexical = next((t for t in tokens if t.pos_ not in {"DET", "PUNCT", "CCONJ"}), None)
    # In imperative instructions the parser can mistake an adjective/participle
    # inside the object (“raised beds”) for the clause root. The leading verb is
    # the instruction's action and should take precedence in that construction.
    first_position = tokens.index(first_lexical) if first_lexical in tokens else -1
    first_following = tokens[first_position + 1] if first_position >= 0 and first_position + 1 < len(tokens) else None
    first_is_copular_predicate = bool(first_following and first_following.lower_ in {
        "is", "are", "was", "were", "be", "been", "being"
    })
    leading_instruction = (first_lexical if first_lexical is not None
                           and normalize_word(first_lexical.text.lower()) in ACTION_WORDS
                           and first_lexical.pos_ in {"VERB", "NOUN", "ADJ"}
                           and not first_is_copular_predicate
                           else None)
    action = leading_instruction or next((t for t in domain_verbs
                   if t.dep_ == "ROOT" and normalize_word(t.lemma_.lower()) != "use"), None)
    if action is None:
        action = next((t for t in domain_verbs if normalize_word(t.lemma_.lower()) != "use"), None)
    if action is None:
        action = domain_verbs[0] if domain_verbs else (verb_tokens[0] if verb_tokens else None)
    # The small English parser often labels imperative verbs as nouns in
    # instructions ("Sow the seeds", "Transplant seedlings"). Recover an
    # action-led clause from its first lexical token rather than dropping it.
    if action is None:
        first = first_lexical
        following = tokens[tokens.index(first) + 1] if first in tokens and tokens.index(first) + 1 < len(tokens) else None
        copular_predicate = bool(following and following.lower_ in {"is", "are", "was", "were", "be", "been", "being"})
        if (first is not None and normalize_word(first.text.lower()) in ACTION_WORDS
                and not copular_predicate):
            action = first
    # spaCy dependency roles are preferred; noun chunks provide a useful fallback.
    # Resolve grammatical roles relative to the selected agricultural verb.
    # Taking the first subject in a whole clause can incorrectly assign a
    # subordinate subject (e.g. "if rainfall is low, he irrigates ...").
    action_subtree = list(action.subtree) if action is not None else tokens
    governors = [action] + list(action.ancestors) if action is not None else []
    subject_candidates = [(governors.index(t.head), t) for t in action_subtree
                          if t.dep_ in {"nsubj", "nsubjpass"} and t.head in governors]
    subject = min(subject_candidates, key=lambda item: item[0])[1] if subject_candidates else None
    agent = subject.text if subject is not None and subject.dep_ != "nsubjpass" else ""
    objects = [t for t in action_subtree
               if t.dep_ in {"dobj", "obj", "attr", "pobj", "oprd"}
               and t.pos_ in {"NOUN", "PROPN"}
               and (not governors or t.head in governors)]
    # Direct objects are the event patient. Prepositional objects such as
    # "nursery" in "sow seeds in a nursery" are context/method, not patient.
    direct_objects = [t for t in objects if t.dep_ in {"dobj", "obj", "attr", "oprd"}]
    patient = " ".join(t.text for t in (direct_objects or objects)[:3])
    if action is not None and action.lemma_.lower() in {
        "apply", "spread", "spray", "add", "supply", "provide", "deliver"
    }:
        recipient_prepositions = {"to", "onto", "on", "around", "over"}
        recipients = [t for t in action_subtree
                      if t.dep_ == "pobj" and t.head.dep_ == "prep"
                      and t.head.lower_ in recipient_prepositions]
        if recipients:
            patient = recipients[0].text
    if action is not None and action == tokens[0]:
        # In parser-misread imperatives, recover the first noun phrase after
        # the action and stop at a preposition (e.g. "Sow tomato seeds in...").
        tail = []
        for token in tokens[1:]:
            if token.pos_ in {"ADP", "CCONJ", "VERB"}:
                break
            if token.pos_ in {"NOUN", "PROPN"}:
                tail.append(token.text)
        if tail:
            patient = " ".join(tail[:3])
    if not patient and subject is not None and subject.dep_ == "nsubjpass":
        patient = subject.text
    if not patient:
        noun_chunks = list(doc.noun_chunks)
        patient = next((c.text for c in noun_chunks if c.root.dep_ not in {"nsubj", "nsubjpass"}), "")

    lowered = clause.lower().replace("’", "'").replace("‘", "'")
    has_explicit_subject = subject is not None
    # Scope negation to the selected action. Trailing contrast such as
    # "applies fertilizer, not pesticide" negates a noun, not the action.
    negation = None
    if action is not None:
        neg_token = next((token for token in action.children
                          if token.idx < action.idx
                          and (token.dep_ == "neg" or token.lower_ in {"not", "n't", "never"})), None)
        if neg_token is not None:
            negation = neg_token.text
        else:
            prefix = re.sub(r"[^a-z']+", " ", clause[:action.idx].lower().replace("’", "'").replace("‘", "'")).strip()
            match = re.search(
                r"(?:\b(?:do|does|did|should|must|will|can|could|would|has|have|had)\s+)?"
                r"(not|n't|never)\s*$", prefix
            )
            if match:
                negation = match.group(1)
    if negation:
        polarity = "negative"
        status = "negated" if has_explicit_subject else "prohibited"
        if re.search(r"\b(?:should|must)\s+not\b|\b(?:shouldn't|mustn't)\b", lowered) or (
            not has_explicit_subject
            and re.search(r"\b(?:do|does|did)\s+not\b|\b(?:don't|doesn't|didn't)\b", lowered)
        ):
            status = "prohibited"
        polarity_evidence = negation
    elif re.search(r"\b(stopped|stop|stops|ceased|cease|ceases|no longer)\b", lowered):
        polarity, status, polarity_evidence = "positive", "ceased", ""
    elif re.search(r"\b(may|might|could|would|possibly|perhaps)\b", lowered):
        polarity, status, polarity_evidence = "positive", "hypothetical", ""
    else:
        polarity, status, polarity_evidence = "positive", "asserted", ""

    tense = "past" if any(t.tag_ in {"VBD", "VBN"} for t in tokens) else "present"
    # Action-specific object cues prevent "drinks water" becoming irrigation.
    has_agricultural_object = any(w in words for w in {"crop", "field", "soil", "plant", "seed", "farm", "land"})
    method_text = ""
    return {"doc": doc, "lemmas": lemmas, "words": words, "action_token": action,
            "action": normalize_word(action.lemma_.lower()) if action else "",
            "agent": agent, "patient": patient, "polarity": polarity,
            "polarity_evidence": polarity_evidence,
            "status": status, "tense": tense, "has_agricultural_object": has_agricultural_object,
            "method_text": method_text}


def _score_profiles(clause: str, features: dict, top_k: int = 12, query_vector=None,
                    context_text: str = "") -> list[dict]:
    catalogue = _catalogue()
    profiles = catalogue["profiles"]
    if not profiles:
        return []
    retrieval_text = context_text or clause
    if catalogue["model"] is not None and query_vector is None:
        query_vector = catalogue["model"].encode(retrieval_text, convert_to_tensor=True, normalize_embeddings=True)
    semantic = (util.cos_sim(query_vector, catalogue["vectors"])[0]
                if catalogue["model"] is not None else None)
    profiles_by_id = {profile["process_id"]: profile for profile in profiles}
    scored = []
    for profile in profiles:
        sem = semantic[profile["embedding_index"]].item() if semantic is not None else 0.0
        # Specific method phrases are independent of the parent process label.
        method = next((m for m in profile["methods"] + profile["aliases"]
                       if _phrase_present(retrieval_text, m)
                       and not _phrase_is_negated(retrieval_text, m)), "")
        action_fit = float(bool(
            (features["action"] and features["action"] in profile["actions"])
            or any(" " in action and _phrase_present(clause, action) for action in profile["actions"])
            or (features["action"] == "use" and bool(method))
        ))
        # A subtype must be explicitly present; semantic resemblance alone must
        # not turn ordinary irrigation into sprinkler/drip irrigation.
        if profile["parent_id"] in {"IRRIGATION", "FERTILIZATION"} and not method:
            continue
        parent = profiles_by_id.get(profile["parent_id"])
        if parent is not None:
            explicit_name = _phrase_present(retrieval_text, profile["process_name"])
            explicit_alias = any(_phrase_present(retrieval_text, alias) for alias in profile["aliases"])
            positive_method = any(_phrase_present(retrieval_text, candidate)
                                  and not _phrase_is_negated(retrieval_text, candidate)
                                  for candidate in profile["methods"])
            if profile["methods"] and not (explicit_name or explicit_alias or positive_method):
                continue
            distinct_method = bool(method and method in profile["methods"]
                                   and method not in parent["methods"])
            distinct_action = features["action"] in (profile["actions"] - parent["actions"])
            patient_tokens = set(tokenize(features.get("patient", "")))
            child_object_match = bool(patient_tokens & {normalize_word(x) for x in profile["objects"]})
            parent_object_match = bool(patient_tokens & {normalize_word(x) for x in parent["objects"]})
            if not (explicit_name or explicit_alias or distinct_method or distinct_action
                    or (child_object_match and not parent_object_match)):
                continue
        phrase_fit = float(bool(method or any(_phrase_present(retrieval_text, a) for a in profile["aliases"])))
        object_words = set(tokenize(features.get("patient", "")))
        if normalize_word(features.get("patient", "")) in {"it", "them", "this", "that"}:
            # Coordinated directions often say "add compost ... and mix it";
            # use same-sentence antecedent context only for pronoun patients.
            object_words.update(tokenize(retrieval_text))
        if not profile["objects"]:
            object_fit = 0.5
        else:
            object_fit = float(any(normalize_word(o) in object_words
                                   or _phrase_present(features.get("patient", ""), o)
                                   for o in profile["objects"]))
        if profile["objects"] and object_fit == 0 and not method and not phrase_fit:
            # Do not call a bed a field preparation or a crop a seed just because
            # the action verb overlaps; the selected object must fit the process.
            action_fit = 0.0
        if features["action"] == "apply" and not object_fit and not method and not phrase_fit:
            action_fit = 0.0
        semantic_fit = max(0.0, min(1.0, (float(sem) + 1.0) / 2.0)) if semantic is not None else 0.0

        # Exclude lexical lookalikes where the verb's sense is clearly non-agricultural.
        if features["action"] == "drink" and profile["process_id"] in {"IRRIGATION", "DRIP_IRRIGATION", "SPRINKLER_IRRIGATION"}:
            continue
        if profile["process_id"] in {"IRRIGATION", "DRIP_IRRIGATION", "SPRINKLER_IRRIGATION"} and features["action"] == "water" and not features["has_agricultural_object"] and not method:
            continue
        if not action_fit and not phrase_fit:
            # Semantic-only retrieval remains available for paraphrases, but require
            # an agricultural object/context to avoid generic word association.
            if not features["has_agricultural_object"]:
                continue
        score = (SCORING_CONFIG["action"] * action_fit
                 + SCORING_CONFIG["object"] * object_fit
                 + SCORING_CONFIG["phrase"] * phrase_fit
                 + SCORING_CONFIG["semantic"] * semantic_fit)
        scored.append({"profile": profile, "score": score, "semantic": float(sem),
                       "action_fit": action_fit, "object_fit": object_fit,
                       "phrase_fit": phrase_fit, "method": method})
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


def _detect_general_clause(clause: str, relation: str, index: int, query_vector=None,
                           context_text: str = "") -> AgriculturalEvent | None:
    """Fallback over legacy process rows not yet given curated event metadata."""
    features = _event_features(clause)
    if not features["action"] or features["action"] in {"drink", "eat", "repair", "walk", "sleep"}:
        return None
    if not features["has_agricultural_object"] and not re.search(r"\b(farmer|farmers|agricultur|crop|field|soil|seed|plant)\b", clause, re.I):
        return None
    catalogue = _catalogue()
    if catalogue["model"] is None:
        return None
    retrieval_text = context_text or clause
    query = query_vector if query_vector is not None else catalogue["model"].encode(retrieval_text, convert_to_tensor=True, normalize_embeddings=True)
    similarities = util.cos_sim(query, catalogue["vectors"])[0]
    candidates = []
    for profile_index, profile in enumerate(catalogue["all_profiles"]):
        profile_tokens = set(tokenize(profile["profile_text"]))
        name_tokens = set(tokenize(profile["process_name"]))
        # A term anywhere in a long description is weak evidence for a verb.
        # Require an action compatible with the process name or explicit
        # process phrase; the curated profiles handle synonym-rich actions.
        action_fit = float(features["action"] in name_tokens)
        phrase_fit = float(_phrase_present(clause, profile["process_name"]) or features["action"] in name_tokens)
        patient_tokens = set(tokenize(features["patient"]))
        if patient_tokens & {"it", "them", "this", "that"}:
            patient_tokens = set(tokenize(retrieval_text))
        object_fit = float(bool(patient_tokens & profile_tokens))
        semantic = float(similarities[profile_index].item())
        if not (action_fit or phrase_fit or (semantic >= 0.62 and object_fit)):
            continue
        score = 0.35 * action_fit + 0.20 * object_fit + 0.15 * phrase_fit + 0.30 * max(0.0, min(1.0, (semantic + 1.0) / 2.0))
        candidates.append((score, semantic, profile, action_fit, object_fit))
    candidates.sort(key=lambda x: x[0], reverse=True)
    if not candidates:
        return None
    score, semantic, profile, action_fit, object_fit = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else 0.0
    exact_process_action = _phrase_present(clause, profile["process_name"]) or features["action"] in set(tokenize(profile["process_name"]))
    if (score < 0.60 or not action_fit or not object_fit
            or (not exact_process_action and semantic < SCORING_CONFIG["legacy_semantic_min"])
            or (score - second_score < SCORING_CONFIG["ambiguity_margin"] and not exact_process_action)):
        return None
    process_id = re.sub(r"[^A-Z0-9]+", "_", profile["process_name"].upper()).strip("_")
    sign_concepts = [key.upper() for key in profile["sign_concepts"]]
    links = []
    if relation in {"after", "before", "then", "next", "finally", "subsequently", "afterwards"}:
        links.append({"relation": relation, "to_sequence_index": index + 1 if relation == "before" else max(1, index - 1)})
    return AgriculturalEvent(
        process_id=process_id, process_name=profile["process_name"],
        action=features["action"], agent=features["agent"], patient=features["patient"],
        polarity=features["polarity"], polarity_evidence=features["polarity_evidence"],
        status=features["status"], tense=features["tense"],
        sequence_index=index, temporal_links=links, evidence=clause.strip(),
        confidence=round(score, 4), confidence_level=_confidence_level(score),
        context_verified=True, category=profile["category"], sign_concepts=sign_concepts,
        semantic_confidence=round(semantic, 4), sequence_relation=relation,
    )


def _confidence_level(score: float) -> str:
    if score >= SCORING_CONFIG["high"]:
        return "high"
    if score >= SCORING_CONFIG["medium"]:
        return "medium"
    return "low"


def _bart_verify_candidates(clause: str, ranked: list[dict]) -> tuple[dict | None, bool | None, float | None]:
    """Use BART as a gated second-stage verifier, never as process selector.

    Strong first-stage matches bypass BART. For uncertain clauses, BART sees
    only action/object-compatible process candidates. It reports whether the
    first-stage choice is supported, but never replaces or deletes that choice.
    If its local checkpoint is unavailable, preserve the first-stage decision.
    """
    if not ranked:
        return None, None, None
    best = ranked[0]
    gap = best["score"] - ranked[1]["score"] if len(ranked) > 1 else 1.0
    needs_verification = (
        best["score"] < SCORING_CONFIG["bart_trigger_score"]
        or (len(ranked) > 1 and gap < SCORING_CONFIG["ambiguity_margin"])
    )
    if not needs_verification:
        return best, None, None

    candidates = ranked[:SCORING_CONFIG["bart_top_k"]]
    names = [candidate["profile"]["process_name"] for candidate in candidates]
    scores = classify_processes(clause, names)
    if scores is None:
        return best, None, None
    if not scores:
        return best, None, None

    selected_name = best["profile"]["process_name"]
    selected_score = scores.get(selected_name, 0.0)
    highest_name = max(scores, key=scores.get)
    bart_verified = (
        selected_score >= SCORING_CONFIG["bart_min_entailment"]
        and highest_name == selected_name
    )
    # BART is evidence shown to downstream callers/reviewers. It is not allowed
    # to overrule the action/object detector: MNLI is general-domain and its
    # agricultural label scores have not been calibrated as a decision policy.
    return best, bart_verified, round(selected_score, 4)


def _detect_clause(clause: str, relation: str, index: int, query_vector=None,
                   context_text: str = "") -> AgriculturalEvent | None:
    features = _event_features(clause)
    profiles = _catalogue()["profiles"]
    ranked = _score_profiles(clause, features, max(SCORING_CONFIG["top_k"], len(profiles)),
                              query_vector, context_text)
    # Do not let a semantically similar but action-incompatible profile block
    # a lower-ranked process whose action and object fit the clause.
    ranked = [candidate for candidate in ranked
              if candidate["action_fit"] > 0 and candidate["score"] >= SCORING_CONFIG["accept"]]
    if not ranked:
        return None
    best, bart_verified, bart_score = _bart_verify_candidates(clause, ranked)
    if best is None:
        return None
    p = best["profile"]
    score = best["score"]
    # Do not guess when there is no process action/phrase or candidates are tied.
    # A bare gerund inside "before planting" is a temporal reference, not by
    # itself an asserted event with an actor or object. Leave it uncertain until
    # the passage provides event context.
    if features["action_token"] is not None and not features["agent"] and not features["patient"]:
        if features["action_token"].tag_ == "VBG" and len(tokenize(clause)) <= 2:
            return None

    variant_id = p["process_id"] if p["process_id"] != "IRRIGATION" and p["parent_id"] == "IRRIGATION" else ""
    if best["method"] in {"drip irrigation", "drip system", "micro irrigation"}:
        variant_id = "DRIP_IRRIGATION"
    elif best["method"] in {"sprinkler", "sprinkler system", "sprinkler irrigation"}:
        variant_id = "SPRINKLER_IRRIGATION"
    elif best["method"] in {"organic fertilizer", "organic manure", "compost"}:
        variant_id = "ORGANIC_FERTILIZATION"
    process_id = p["parent_id"] if p["process_id"] in {"DRIP_IRRIGATION", "SPRINKLER_IRRIGATION", "ORGANIC_FERTILIZATION"} else p["process_id"]
    # Water + agriculture maps to irrigation even when the lexical action is water.
    if p["process_id"] in {"DRIP_IRRIGATION", "SPRINKLER_IRRIGATION"}:
        process_id, variant_id = "IRRIGATION", p["process_id"]
    process_name = p["process_name"]
    if variant_id and process_id in {"IRRIGATION", "FERTILIZATION"}:
        process_name = next((profile["process_name"] for profile in _catalogue()["profiles"] if profile["process_id"] == process_id), process_name)
    patient_terms = {normalize_word(term) for term in p["objects"]}
    patient_matches = [t.text for t in features["doc"]
                       if t.dep_ in {"dobj", "obj", "attr", "pobj", "oprd"}
                       and t.pos_ in {"NOUN", "PROPN"}
                       and normalize_word(t.lemma_.lower()) in patient_terms]
    patient = " ".join(dict.fromkeys(patient_matches)) or features["patient"]
    sign_concepts = [process_id]
    if variant_id:
        sign_concepts.append(variant_id)
    if best["method"] in {"drip irrigation", "drip system", "micro irrigation"} and "DRIP_IRRIGATION" not in sign_concepts:
        sign_concepts.append("DRIP_IRRIGATION")
    if best["method"] in {"sprinkler", "sprinkler system", "sprinkler irrigation"} and "SPRINKLER_IRRIGATION" not in sign_concepts:
        sign_concepts.append("SPRINKLER_IRRIGATION")
    if features["polarity"] == "negative":
        sign_concepts.insert(0, "NEGATION")
    elif features["status"] == "ceased":
        sign_concepts.insert(0, "CESSATION")
    elif features["status"] == "hypothetical":
        sign_concepts.insert(0, "HYPOTHETICAL")

    links = []
    if relation in {"after", "before", "then", "next", "finally", "subsequently", "afterwards"}:
        target_index = index + 1 if relation == "before" else max(1, index - 1)
        links.append({"relation": relation, "to_sequence_index": target_index})
    return AgriculturalEvent(
        process_id=process_id, process_name=process_name,
        action=features["action"], agent=features["agent"], patient=patient,
        method=best["method"], polarity=features["polarity"],
        polarity_evidence=features["polarity_evidence"], status=features["status"],
        tense=features["tense"], sequence_index=index, temporal_links=links,
        evidence=clause.strip(), confidence=round(score, 4),
        confidence_level=_confidence_level(score), process_variant_id=variant_id,
        context_verified=bool(best["action_fit"] and (best["object_fit"] or best["phrase_fit"])),
        category=p["category"], sign_concepts=sign_concepts,
        semantic_confidence=round(best["semantic"], 4),
        bart_verified=bart_verified, bart_score=bart_score, sequence_relation=relation,
    )


def detect_process_many(sentences: list[str]) -> list[dict]:
    """Detect a document's sentences with one batched embedding call."""
    clauses_by_sentence = [_clause_spans(sentence) for sentence in sentences]
    flat_clauses = [f"{sentence} {clause}" for sentence, groups in zip(sentences, clauses_by_sentence)
                    for clause, _ in groups]
    model = _catalogue()["model"]
    vectors = (model.encode(flat_clauses, batch_size=32, convert_to_tensor=True, normalize_embeddings=True)
               if model is not None and flat_clauses else None)
    results = []
    vector_index = 0
    for clauses in clauses_by_sentence:
        events = []
        sentence_context = sentences[len(results)]
        for clause, relation in clauses:
            query_vector = vectors[vector_index] if vectors is not None else None
            vector_index += 1
            event = _detect_clause(clause, relation, len(events) + 1, query_vector, sentence_context)
            if event:
                # Coordinated actions that resolve to one process (for example,
                # "plough and level the field") describe one event unless the
                # text explicitly marks a later occurrence.
                existing = next((item for item in events
                                 if item.process_id == event.process_id
                                 and item.polarity == event.polarity
                                 and item.status == event.status
                                 and not event.sequence_relation), None)
                if existing is not None:
                    existing.evidence = f"{existing.evidence}; {event.evidence}"
                    existing.sign_concepts = list(dict.fromkeys(existing.sign_concepts + event.sign_concepts))
                    existing.confidence = max(existing.confidence, event.confidence)
                    existing.confidence_level = _confidence_level(existing.confidence)
                else:
                    events.append(event)
        for index, event in enumerate(events, start=1):
            event.sequence_index = index
            if index > 1 and not event.sequence_relation:
                event.temporal_links = [{"relation": "then", "to_sequence_index": index - 1}]
        process_rows = [{
            "process_id": event.process_id,
            "process_name": event.process_name,
            "category": event.category,
            "sign_sequence": "|".join(event.sign_concepts),
            "sign_concepts": event.sign_concepts,
            "semantic_confidence": event.semantic_confidence,
            "bart_verified": event.bart_verified,
            "bart_score": event.bart_score,
            "combined_score": event.confidence,
            "context_verified": event.context_verified,
            "event": event.to_dict(),
        } for event in events]
        results.append({"processes": process_rows, "events": [event.to_dict() for event in events]})
    return results


def detect_process(sentence: str) -> dict:
    """Return one sentence's backward-compatible rows and structured events."""
    return detect_process_many([sentence])[0]


if __name__ == "__main__":
    examples = [
        "The farmer waters the crops.",
        "The farmer irrigates the field.",
        "The farmer drinks water.",
        "The farmer uses drip irrigation to water the crops.",
        "First prepare the soil, then sow the seeds and finally irrigate the field.",
        "The farmer does not irrigate the crops.",
        "Do not water the crop.",
        "After sowing the seeds, the farmer irrigates the field.",
        "The farmer applies fertilizer to the crop.",
        "The farmer uses a sprinkler system to water the crops.",
    ]
    initialize_process_detector()
    for example in examples:
        print(example, detect_process(example)["events"])
