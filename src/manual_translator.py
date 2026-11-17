"""Translate document text into ordered events and sign-video sequences.

ISL additions (switch off with USE_ISL_ORDER = False):
  * Each event keeps its process ids (IRRIGATION, CROP_ROTATION ...), which
    sign_sequence_generator expands into real signs and parent fallbacks.
    Around them the signs are put in ISL order:
        TIME + AGENT + OBJECTS + PROCESS + NOT/STOP/MAY + QUESTION
  * Words in the clause that have a sign (wheat, rice, crop rotation ...) are added
    even when the detector did not extract them as the object.
  * A sentence with no detected event falls back to rule-based ISL ordering, and is
    rendered through the same generator (so missing signs and negation are handled alike).
"""

import logging
import re
from functools import lru_cache

from config import BASE_DIR
from preprocessing import split_sentences
from process_detector import detect_process_many
from sign_sequence_generator import generate_video_sequence, load_concept_metadata
from isl_sequence import (END_MARKERS, Resolver, build_vocabulary, load_fallbacks,
                          event_context_signs, map_words, sentence_to_isl)

LOGGER = logging.getLogger(__name__)

USE_ISL_ORDER = True            # False = original detector order, no fallback
SIGN_REPEATED_AGENT = False     # False = sign FARMER only when the agent changes
FALLBACK_INTO_SEQUENCE = True   # include rule-based fallback clips in the video sequence
TERMS_PATH = BASE_DIR / "dataset" / "agro_terms.csv"
FALLBACKS_PATH = BASE_DIR / "dataset" / "term_fallbacks.csv"     # optional: word,keyword
PRONOUNS = {"he", "she", "they", "it", "we", "i"}
NEGATION_RE = re.compile(r"\b(not|no|never|cannot)\b|n['’]t", re.I)


@lru_cache(maxsize=1)
def _resolver():
    """Sign vocabulary (agro_terms.csv), loaded once. None disables the ISL step."""
    try:
        fallbacks = load_fallbacks(FALLBACKS_PATH) if FALLBACKS_PATH.exists() else None
        return Resolver(build_vocabulary(TERMS_PATH), fallbacks=fallbacks)
    except Exception as exc:
        LOGGER.warning("ISL ordering disabled (could not load %s): %s", TERMS_PATH, exc)
        return None


def _empty_result() -> dict:
    return {
        "total_sentences": 0,
        "meaning": "",
        "isl_gloss": [],
        "translations": [],
        "complete_video_sequence": [],
        "process_sequence": [],
        "ideal_event_sequence": [],
        "ideal_sign_sequence": [],
        "renderable_sign_sequence": [],
        "missing_sign_concepts": [],
        "sign_fallbacks": [],
        "translation_diagnostics": [],
        "fallback_sign_sequence": [],
        "isl_missing_words": [],
    }


def translate_manual(text: str) -> dict:
    if not str(text or "").strip():
        return _empty_result()
    sentences = split_sentences(text)
    sentence_detections = detect_process_many(sentences)
    resolver = _resolver() if USE_ISL_ORDER else None

    translation_results = []
    complete_video_sequence = []
    ideal_sign_sequence = []
    renderable_sign_sequence = []
    missing_by_id = {}
    fallbacks = []
    diagnostics = []
    all_processes = []
    ideal_event_sequence = []
    fallback_tokens_all = []
    isl_missing_words = []          # words with no sign (informational)
    global_index = 0
    last_named_agent = ""
    last_signed_agent = ""
    # Keep paragraph topics/participants once. Process/action concepts remain
    # attached to each event, while repeated FARMER/CROPS/SOIL context does not
    # bloat a multi-sentence playlist.
    paragraph_context_signs = set()

    def collect_missing(video_result, sentence):
        for missing in video_result["missing_sign_concepts"]:
            item = missing_by_id.get(missing["concept_id"])
            if item is None:
                item = dict(missing)
                missing_by_id[missing["concept_id"]] = item
            else:
                previous = [x.strip() for x in item.get("example_sentence", "").split(" | ")]
                normalize = lambda value: value.strip().rstrip(".!?").lower()
                if normalize(sentence) not in {normalize(value) for value in previous}:
                    item["example_sentence"] = (item.get("example_sentence", "") + " | " + sentence).strip(" |")

    for sentence_index, sentence in enumerate(sentences, start=1):
        detection = sentence_detections[sentence_index - 1]
        sentence_events = []
        sequence_items = []
        for process in detection["processes"]:
            event = dict(process["event"])
            local_index = event.get("sequence_index", 1)
            sequence_offset = global_index
            global_index += 1
            event["sequence_index"] = global_index
            ideal_event_sequence.append(event)
            agent = str(event.get("agent", "")).strip()
            if agent.lower() in PRONOUNS and last_named_agent:
                event["agent"] = last_named_agent
            elif agent and agent.lower() not in PRONOUNS:
                last_named_agent = agent
            for link in event.get("temporal_links", []):
                # Detector links are sentence-local; translate their target to
                # the page-wide event index.
                local_target = int(link.get("to_sequence_index", local_index))
                if local_target == local_index:
                    if sequence_offset:
                        link["to_sequence_index"] = sequence_offset
                    else:
                        link["to_sequence_index"] = 0
                else:
                    link["to_sequence_index"] = local_target + sequence_offset
            event["temporal_links"] = [link for link in event.get("temporal_links", [])
                                        if link["to_sequence_index"] > 0
                                        and link["to_sequence_index"] != global_index]
            if global_index > 1 and not event.get("temporal_links"):
                event["temporal_links"] = [{"relation": "then", "to_sequence_index": global_index - 1}]

            # ---- ISL ordering: signs before / after the detected process --------------
            process_for_video = process
            if resolver is not None:
                try:
                    metadata = load_concept_metadata()
                    concepts = [str(c) for c in (event.get("sign_concepts")
                                                 or process.get("sign_concepts") or [])]
                    markers = [c for c in concepts if c.upper() in END_MARKERS]
                    process_part = [c for c in concepts if c.upper() not in END_MARKERS]

                    # Do not scan for signs the process already supplies, or for other
                    # process names: those come from the detected process itself.
                    omit = {pid.lower() for pid in metadata}
                    # Also omit the resolved component signs of this event's
                    # process while gathering its topic/object context. For
                    # example, COMPOSTING supplies COMPOST itself; putting a
                    # scanned COMPOST before ADD would reverse the action.
                    for concept in concepts:
                        record = metadata.get(concept.upper())
                        if record:
                            omit.update(sign.lower() for sign in record["sign_concepts"])
                    omit.update(paragraph_context_signs)
                    agent_keys = map_words(str(event.get("agent", "")).lower().split(),
                                           resolver, spot=True)
                    agent_key = agent_keys[0] if agent_keys else ""
                    if agent_key:
                        if agent_key == last_signed_agent and not SIGN_REPEATED_AGENT:
                            omit.add(agent_key)          # same topic as before: do not sign again
                        else:
                            last_signed_agent = agent_key

                    context = event_context_signs(event, resolver, isl_missing_words, omit=omit)
                    paragraph_context_signs.update(context["pre"])
                    ordered = ([t.upper() for t in context["pre"]] + process_part + markers
                               + [t.upper() for t in context["post"]])
                    if ordered:
                        event["sign_concepts_detected"] = concepts
                        event["sign_concepts"] = ordered
                        process_for_video = {**process, "sign_concepts": ordered,
                                             "sign_sequence": "|".join(ordered),
                                             "context_signs_resolved": True}
                except Exception as exc:                 # never break translation because of ISL
                    LOGGER.warning("ISL ordering skipped for %r: %s", sentence, exc)
            # ------------------------------------------------------------------------------

            video_result = generate_video_sequence({**process_for_video, "event": event})
            sentence_events.append({
                "process_id": event["process_id"],
                "process_name": event["process_name"],
                "category": event["category"],
                "polarity": event.get("polarity", "positive"),
                "status": event.get("status", "asserted"),
                "semantic_confidence": event["semantic_confidence"],
                "bart_verified": event["bart_verified"],
                "bart_score": event["bart_score"],
                "combined_score": event["confidence"],
                "context_verified": process.get("context_verified", True),
                "event": event,
                "video_sequence": video_result["video_sequence"],
                "ideal_sign_sequence": video_result["ideal_sign_sequence"],
                "renderable_sign_sequence": video_result["renderable_sign_sequence"],
                "missing_sign_concepts": video_result["missing_sign_concepts"],
                "fallbacks": video_result["fallbacks"],
                "polarity_rendered": video_result["polarity_rendered"],
                "translation_diagnostics": video_result["translation_diagnostics"],
            })
            evidence = str(event.get("evidence", ""))
            sequence_items.append({
                "position": event.get("clause_index", event.get("sequence_index", 0)),
                "gloss": video_result["ideal_gloss"],
                "video": video_result["video_sequence"],
                "ideal": video_result["ideal_sign_sequence"],
                "renderable": video_result["renderable_sign_sequence"],
            })
            paragraph_context_signs.update(
                str(sign).lower() for sign in video_result.get("ideal_gloss", [])
            )
            all_processes.append(event["process_name"])
            fallbacks.extend(video_result["fallbacks"])
            diagnostics.extend(video_result["translation_diagnostics"])
            collect_missing(video_result, sentence)

        sentence_missing = [c for item in sentence_events for c in item["missing_sign_concepts"]]

        # ---- rule-based ISL fallback when no event was detected ---------------------------
        fallback_entry = None
        fallback_entries = []
        if resolver is not None:
            # Translate only clauses that did not produce a process event.
            # Previously the fallback ran only when the whole sentence had no
            # detected process, dropping meaningful state clauses in mixed
            # sentences and rescanning other events' words.
            clauses_to_fallback = detection.get("unmatched_clauses", [])
            if not detection["processes"] and not clauses_to_fallback:
                clauses_to_fallback = [{"text": sentence, "relation": "", "clause_index": 0}]
            for unmatched in clauses_to_fallback:
                clause = str(unmatched.get("text", "")).strip()
                if not clause:
                    continue
                try:
                    fb = sentence_to_isl(clause, lambda _text: {"events": []}, resolver)
                    # Keep negation scoped to this unmatched clause.
                    negative = bool(NEGATION_RE.search(clause))
                    tokens = [t for t in fb["tokens"]
                              if str(t).strip().lower() not in {"not", "no", "never", "cannot", "negation"}]
                    tokens = [t for t in tokens if str(t).lower() not in paragraph_context_signs]
                    paragraph_context_signs.update(str(t).lower() for t in tokens)
                    concepts = [t.upper() for t in tokens] + (["NEGATION"] if negative else [])
                    current_fallback = {"sentence": clause, "sentence_index": sentence_index,
                                        "tokens": tokens, "negative": negative,
                                        "missing_words": fb["missing"], "approximate": fb["approximate"]}
                    isl_missing_words.extend(w for w in fb["missing"] if w not in isl_missing_words)
                    if concepts and FALLBACK_INTO_SEQUENCE:
                        fb_event = {"process_id": "", "process_name": "ISL rule-based fallback",
                                    "category": "", "polarity": "negative" if negative else "positive",
                                    "status": "asserted", "evidence": clause,
                                    "sequence_index": None, "semantic_confidence": 0.0,
                                    "sign_concepts": concepts}
                        video_result = generate_video_sequence({"event": fb_event})
                        sequence_items.append({
                            "position": unmatched.get("clause_index", len(sequence_items)),
                            "gloss": video_result["ideal_gloss"],
                            "video": video_result["video_sequence"],
                            "ideal": video_result["ideal_sign_sequence"],
                            "renderable": video_result["renderable_sign_sequence"],
                        })
                        sentence_missing.extend(video_result["missing_sign_concepts"])
                        fallbacks.extend(video_result["fallbacks"])
                        diagnostics.extend(video_result["translation_diagnostics"])
                        collect_missing(video_result, clause)
                        fallback_tokens_all.extend(tokens)
                        current_fallback["concepts"] = concepts
                        current_fallback["video_sequence"] = video_result["video_sequence"]
                    fallback_entries.append(current_fallback)
                    if fallback_entry is None:
                        fallback_entry = current_fallback
                except Exception as exc:
                    LOGGER.warning("ISL fallback skipped for %r: %s", clause, exc)
        # ------------------------------------------------------------------------------------

        sequence_items.sort(key=lambda item: item["position"])
        sentence_videos = [clip for item in sequence_items for clip in item["video"]]
        sentence_ideal = [clip for item in sequence_items for clip in item["ideal"]]
        sentence_renderable = [clip for item in sequence_items for clip in item["renderable"]]
        sentence_gloss = [concept for item in sequence_items for concept in item["gloss"]]
        complete_video_sequence.extend(sentence_videos)
        ideal_sign_sequence.extend(sentence_ideal)
        renderable_sign_sequence.extend(sentence_renderable)
        translation_results.append({
            "sentence": sentence,
            # English source text is the meaning shown beside the ISL gloss.
            "meaning": sentence.strip(),
            "isl_gloss": sentence_gloss,
            "isl_gloss_text": " ".join(sentence_gloss),
            "sentence_index": sentence_index,
            "processes": sentence_events,
            "events": [item["event"] for item in sentence_events],
            "video_sequence": sentence_videos,
            "ideal_sign_sequence": sentence_ideal,
            "renderable_sign_sequence": sentence_renderable,
            "missing_sign_concepts": sentence_missing,
            "fallback": fallback_entry,
            "fallbacks": fallback_entries,
        })

    return {
        "total_sentences": len(sentences),
        "meaning": " ".join(item["meaning"] for item in translation_results),
        "isl_gloss": [token for item in translation_results for token in item["isl_gloss"]],
        "translations": translation_results,
        "complete_video_sequence": complete_video_sequence,
        "process_sequence": all_processes,
        "ideal_event_sequence": ideal_event_sequence,
        "ideal_sign_sequence": ideal_sign_sequence,
        "renderable_sign_sequence": renderable_sign_sequence,
        "missing_sign_concepts": list(missing_by_id.values()),
        "sign_fallbacks": fallbacks,
        "translation_diagnostics": diagnostics,
        "fallback_sign_sequence": fallback_tokens_all,
        "isl_missing_words": isl_missing_words,
    }
