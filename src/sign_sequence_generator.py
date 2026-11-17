"""Map ideal event concepts to real videos without changing NLP decisions."""

from functools import lru_cache

from pathlib import Path
import re

import pandas as pd

from config import BASE_DIR, VIDEO_DIR

METADATA_PATH = BASE_DIR / "dataset" / "agro_process_metadata.csv"

@lru_cache(maxsize=1)

def load_sign_videos() -> dict[str, str]:

    terms_path = BASE_DIR / "dataset" / "agro_terms.csv"

    df = pd.read_csv(terms_path).fillna("")

    required = {"keyword", "video_path"}

    missing = required.difference(df.columns)

    if missing:

        raise ValueError(f"Sign mapping CSV is missing required columns: {', '.join(sorted(missing))}")

    return {

        str(row["keyword"]).strip().lower(): str(row["video_path"]).strip()

        for _, row in df.iterrows()

        if str(row["keyword"]).strip() and str(row["video_path"]).strip()

    }

@lru_cache(maxsize=1)

@lru_cache(maxsize=1)
def load_agro_term_rows() -> list[dict]:
    """Load canonical keywords and synonyms for sentence-level matching."""
    terms_path = BASE_DIR / "dataset" / "agro_terms.csv"
    df = pd.read_csv(terms_path).fillna("")
    required = {"keyword", "synonyms", "video_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"agro_terms.csv is missing columns: {', '.join(sorted(missing))}"
        )

    rows = []
    for _, row in df.iterrows():
        keyword = str(row["keyword"]).strip().lower()
        video_path = str(row["video_path"]).strip()
        if not keyword or not video_path:
            continue

        variants = [keyword.replace("_", " ")]
        variants.extend(
            synonym.strip().lower().replace("_", " ")
            for synonym in str(row["synonyms"]).split("|")
            if synonym.strip()
        )
        rows.append({
            "keyword": keyword,
            "video_path": video_path,
            "variants": list(dict.fromkeys(variants)),
        })
    return rows


def extract_sentence_term_keys(sentence: str) -> list[str]:
    """Return canonical agro_terms keywords explicitly mentioned in a sentence."""
    if not sentence:
        return []

    text = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
    matches = []
    for row in load_agro_term_rows():
        earliest_position = None
        for variant in row["variants"]:
            phrase = re.sub(r"[^a-z0-9]+", " ", variant).strip()
            if not phrase:
                continue
            pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
            match = re.search(pattern, text)
            if match and (earliest_position is None or match.start() < earliest_position):
                earliest_position = match.start()
        if earliest_position is not None:
            matches.append((earliest_position, row["keyword"]))

    matches.sort(key=lambda item: item[0])
    return list(dict.fromkeys(keyword for _, keyword in matches))


def load_concept_metadata() -> dict[str, dict]:

    df = pd.read_csv(METADATA_PATH).fillna("")

    required = {"process_id", "process_name", "parent_id", "sign_concepts"}

    missing = required.difference(df.columns)

    if missing:

        raise ValueError(f"Process metadata CSV is missing required columns: {', '.join(sorted(missing))}")

    return {

        str(row["process_id"]).strip().upper(): {

            "name": str(row["process_name"]).strip(),

            "parent_id": str(row["parent_id"]).strip().upper(),

            "sign_concepts": [x.strip().lower() for x in str(row["sign_concepts"]).split("|") if x.strip()],

        }

        for _, row in df.iterrows()

    }

def _video_for_key(key: str) -> str | None:

    key = str(key).strip().lower()
    relative = load_sign_videos().get(key)

    # Sign keywords may be referred to by a synonym in process metadata
    # (for example, NEGATION expands to "not", while the CSV keyword is
    # "negation"). Resolve aliases to their canonical keyword before looking
    # up the clip path.
    if not relative:
        normalized = re.sub(r"[^a-z0-9]+", " ", key).strip()
        for row in load_agro_term_rows():
            if normalized in row["variants"]:
                relative = row["video_path"]
                break

    if not relative:

        return None

    path = (BASE_DIR / relative).resolve()

    # Do not advertise a mapping whose file is absent from the actual library.

    try:

        path.relative_to(VIDEO_DIR.resolve())

    except ValueError:

        return None

    return str(path) if path.is_file() else None

def _expand_concept(concept_id: str) -> tuple[list[str], str, str]:

    """Return mapped video keys, type, and a human-readable concept name."""

    metadata = load_concept_metadata()

    key = concept_id.upper()

    record = metadata.get(key)

    if record:

        return record["sign_concepts"], "process" if key not in {"DRIP_IRRIGATION", "SPRINKLER_IRRIGATION", "ORGANIC_FERTILIZATION"} else "method", record["name"]

    if key == "NEGATION":

        return ["not"], "modifier", "Negation"

    if key == "CESSATION":

        return ["stop"], "modifier", "Cessation"

    if key == "HYPOTHETICAL":

        return ["may"], "modifier", "Hypothetical modality"

    return [key.lower()], "concept", key.replace("_", " ").title()

def _resolve_ideal_concept(concept_id: str, event: dict, example: str) -> tuple[dict, list[dict], list[dict]]:

    keys, concept_type, concept_name = _expand_concept(concept_id)

    direct_paths = [(key, _video_for_key(key)) for key in keys]

    available = [(key, path) for key, path in direct_paths if path]

    missing = []

    renderable = []

    fallback_records = []

    if available:

        for key, path in available:

            renderable.append({

                "video_concept": key,

                "path": path,

                "requested_concept_id": concept_id,

                "fallback": False,

                "event_polarity": event.get("polarity", "positive"),

                "event_status": event.get("status", "asserted"),

            })

        for key, _ in direct_paths:

            if not _video_for_key(key):

                missing.append({

                    "concept": key.replace("_", " ").title(),

                    "concept_id": key.upper(),

                    "type": "sign concept",

                    "part_of_concept_id": concept_id,

                    "reason": "Required component sign has no mapped video.",

                    "required_for_sequence": True,

                    "video_available": False,

                    "fallback": None,

                    "example_sentence": example,

                    "priority": "Medium",

                    "event_polarity": event.get("polarity", "positive"),

                })

    else:

        fallback_concept_id = ""

        fallback_clips = []

        record = load_concept_metadata().get(concept_id.upper())

        parent = record["parent_id"] if record else ""

        visited = set()

        while parent and parent not in visited:

            visited.add(parent)

            parent_record = load_concept_metadata().get(parent)

            if not parent_record:

                break

            parent_clips = [(key, _video_for_key(key)) for key in parent_record["sign_concepts"]]

            fallback_clips = [(key, path) for key, path in parent_clips if path]

            if fallback_clips:

                fallback_concept_id = parent

                break

            parent = parent_record["parent_id"]

        for fallback_key, fallback_path in fallback_clips:

            renderable.append({

                "video_concept": fallback_key,

                "path": fallback_path,

                "requested_concept_id": concept_id,

                "fallback": True,

            })

        if fallback_clips:

            fallback_records.append({

                "requested_concept_id": concept_id,

                "requested_concept": concept_name,

                "rendered_concept_id": fallback_concept_id,

                "rendered_concept": load_concept_metadata()[fallback_concept_id]["name"],

                "video_concepts": [key.upper() for key, _ in fallback_clips],

                "video_paths": [path for _, path in fallback_clips],

                "reason": "Specific concept video unavailable; mapped parent concept used.",

            })

        missing.append({

            "concept": concept_name,

            "concept_id": concept_id,

            "type": concept_type,

            "reason": "Important detected concept has no directly mapped video.",

            "required_for_sequence": True,

            "video_available": False,

            "fallback": fallback_concept_id or None,

            "example_sentence": example,

            "priority": "High" if concept_id in {

                "NEGATION", "DRIP_IRRIGATION", "SOIL_PREPARATION", "SEED_SOWING",

                "IRRIGATION", "FIELD_SELECTION", "LAND_CLEARING", "SOIL_AMENDMENT",

                "BED_PREPARATION", "TRANSPLANTING", "SOIL_MOISTURE_CONSERVATION",

            } else "Medium",

            "event_polarity": event.get("polarity", "positive"),

        })

    ideal = {

        "concept": concept_name,

        "concept_id": concept_id,

        "type": concept_type,

        "event_index": event.get("sequence_index"),

        "event_polarity": event.get("polarity", "positive"),

        "event_status": event.get("status", "asserted"),

        "video_available": bool(available) and len(available) == len(direct_paths),

        "required_for_sequence": True,

    }

    return ideal, renderable, missing + fallback_records

def generate_video_sequence(
    process_result: dict,
    sentence: str | None = None,
) -> dict:
    """Resolve process concepts plus relevant terms explicitly present in the sentence."""
    event = process_result.get("event", process_result)
    concepts = list(event.get("sign_concepts") or process_result.get("sign_concepts") or [])
    source_sentence = sentence or event.get("evidence", "")

    # Supplement process-level signs with terms independently matched from the sentence.
    # Manual translation has already chosen the event's context terms and
    # ordered them for ISL. Rescanning the full clause here can re-add words
    # from another event or a temporal phrase (and create playlist-only extras).
    additional_terms = ([] if process_result.get("context_signs_resolved")
                        else extract_sentence_term_keys(source_sentence))
    seen_concepts = {str(concept).strip().upper() for concept in concepts}
    for keyword in additional_terms:
        if keyword.upper() not in seen_concepts:
            concepts.append(keyword.upper())
            seen_concepts.add(keyword.upper())

    ideal_sequence = []
    ideal_gloss = []
    renderable_sequence = []
    missing_sign_concepts = []
    fallbacks = []
    seen_video_paths = set()
    seen_ideal_keys = set()

    for concept_id in concepts:
        keys, _concept_type, _concept_name = _expand_concept(str(concept_id))
        for key in keys:
            normalized_key = str(key).strip().upper()
            if normalized_key and normalized_key not in seen_ideal_keys:
                ideal_gloss.append(normalized_key)
                seen_ideal_keys.add(normalized_key)
        ideal, renderable, records = _resolve_ideal_concept(
            str(concept_id), event, source_sentence
        )
        ideal_sequence.append(ideal)
        for clip in renderable:
            if clip["path"] not in seen_video_paths:
                seen_video_paths.add(clip["path"])
                renderable_sequence.append(clip)
        for record in records:
            if "example_sentence" in record:
                missing_sign_concepts.append(record)
            else:
                fallbacks.append(record)

    polarity = str(event.get("polarity", "positive")).lower()
    polarity_rendered = polarity != "negative" or bool(_video_for_key("not"))
    if not polarity_rendered:
        renderable_sequence = []
        diagnostics = [{
            "code": "UNREPRESENTED_NEGATION",
            "event_index": event.get("sequence_index"),
            "polarity": polarity,
            "status": event.get("status", ""),
            "message": (
                "The event is negative, but the sign-video library has no mapped "
                "negation sign; process clips were withheld to avoid presenting "
                "a positive instruction."
            ),
        }]
    else:
        diagnostics = []

    return {
        "process_id": event.get("process_id", ""),
        "process_name": process_result.get("process_name", event.get("process_name", "")),
        "category": process_result.get("category", event.get("category", "")),
        "polarity": polarity,
        "status": event.get("status", "asserted"),
        "polarity_rendered": polarity_rendered,
        "translation_diagnostics": diagnostics,
        "semantic_confidence": process_result.get(
            "semantic_confidence", event.get("semantic_confidence", 0.0)
        ),
        "bart_verified": process_result.get("bart_verified"),
        "bart_score": process_result.get("bart_score"),
        "video_sequence": [clip["path"] for clip in renderable_sequence],
        # The gloss is built from the exact component signs requested by the
        # same resolver that selects videos. Missing clips remain visible in
        # the gloss and are reported separately instead of inventing a video.
        "ideal_gloss": ideal_gloss,
        "ideal_sign_sequence": ideal_sequence,
        "renderable_sign_sequence": renderable_sequence,
        "missing_sign_concepts": missing_sign_concepts,
        "fallbacks": fallbacks,
        "sentence_term_keys": additional_terms,
    }
