"""Translate document text into ordered events and sign-video sequences."""

import re

from preprocessing import split_sentences
from process_detector import detect_process_many, normalize_word
from sign_sequence_generator import generate_video_sequence, load_sign_term_entries


def _matches_temporal_reference(reference: dict, prior_event: dict) -> bool:
    """Require process and action agreement before resolving an event reference."""
    if reference.get("process_id") != prior_event.get("process_id"):
        return False
    reference_action = str(reference.get("action", "")).strip().lower()
    prior_action = str(prior_event.get("action", "")).strip().lower()
    if not reference_action or reference_action != prior_action:
        return False

    reference_patient = set(re.findall(r"[a-z0-9]+", str(reference.get("patient", "")).lower()))
    prior_patient = set(re.findall(r"[a-z0-9]+", str(prior_event.get("patient", "")).lower()))
    return not reference_patient or not prior_patient or bool(reference_patient & prior_patient)


def _term_signs(text: str, events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Match mapped lexical terms to signs without creating process events."""
    if any(str(event.get("polarity", "positive")).lower() == "negative" for event in events):
        return [], []
    searchable = str(text).lower()
    searchable = re.sub(r"\b(before|after)\b.*$", " ", searchable)
    for event in events:
        condition = str(event.get("condition_text", "")).strip().lower()
        if condition:
            searchable = re.sub(r"(?<!\w)" + re.escape(condition) + r"(?!\w)", " ", searchable)
    raw_words = re.findall(r"[a-z]+", searchable)
    words = [normalize_word(word) for word in raw_words]
    ideals, clips = [], []
    seen_paths = set()
    for entry in load_sign_term_entries():
        aliases = [entry["keyword"], *entry["synonyms"].split("|")]
        matched = False
        for alias in aliases:
            alias_words = [normalize_word(word) for word in re.findall(r"[a-z]+", alias)]
            if not alias_words:
                continue
            width = len(alias_words)
            raw_alias_words = re.findall(r"[a-z]+", alias)
            for index in range(len(words) - width + 1):
                raw_match = raw_words[index:index + width] == raw_alias_words
                normalized_match = words[index:index + width] == alias_words
                # Do not collapse gerunds into noun aliases (e.g. "planting"
                # must not match the crops mapping's synonym "plants").
                has_unmatched_gerund = any(
                    word.endswith("ing") and word != raw_alias_words[offset]
                    for offset, word in enumerate(raw_words[index:index + width])
                )
                if raw_match or (normalized_match and not has_unmatched_gerund):
                    matched = True
                    break
            if matched:
                break
        if not matched or entry["path"] in seen_paths:
            continue
        event_material = " ".join(
            str(event.get(field, ""))
            for event in events
            for field in ("action", "patient", "method_text")
        ).lower()
        event_words = [normalize_word(word) for word in re.findall(r"[a-z]+", event_material)]
        canonical_words = [normalize_word(word) for word in re.findall(r"[a-z]+", entry["keyword"])]
        if canonical_words and any(
            event_words[index:index + len(canonical_words)] == canonical_words
            for index in range(len(event_words) - len(canonical_words) + 1)
        ):
            # Event arguments/actions are already handled by the process-level
            # sign mapping; term augmentation is for additional sentence terms.
            continue
        seen_paths.add(entry["path"])
        ideals.append({
            "concept": entry["keyword"].replace("_", " ").title(),
            "concept_id": entry["keyword"].upper(),
            "type": "term",
            "event_index": None,
            "event_polarity": "unscoped",
            "event_status": "asserted",
            "video_available": True,
            "required_for_sequence": True,
        })
        clips.append({
            "video_concept": entry["keyword"],
            "path": entry["path"],
            "requested_concept_id": entry["keyword"].upper(),
            "fallback": False,
            "event_polarity": "unscoped",
            "event_status": "asserted",
        })
    return ideals, clips


def translate_manual(text: str) -> dict:
    if not str(text or "").strip():
        return {
            "total_sentences": 0,
            "translations": [],
            "complete_video_sequence": [],
            "process_sequence": [],
            "ideal_event_sequence": [],
            "ideal_sign_sequence": [],
            "renderable_sign_sequence": [],
            "missing_sign_concepts": [],
            "sign_fallbacks": [],
            "translation_diagnostics": [],
        }
    sentences = split_sentences(text)
    sentence_detections = detect_process_many(sentences)
    translation_results = []
    complete_video_sequence = []
    ideal_sign_sequence = []
    renderable_sign_sequence = []
    missing_by_id = {}
    fallbacks = []
    diagnostics = []
    all_processes = []
    ideal_event_sequence = []
    global_index = 0
    last_named_agent = ""
    prior_events = []

    for sentence_index, sentence in enumerate(sentences, start=1):
        detection = sentence_detections[sentence_index - 1]
        sentence_events = []
        sentence_videos = []
        sentence_ideal_terms = []
        sentence_term_clips = []
        local_to_global = {}
        for process in detection["processes"]:
            event = dict(process["event"])
            local_index = event.get("sequence_index", 1)
            if event.get("sequence_relation") == "after_reference":
                reference_target = next((prior for prior in reversed(prior_events)
                                         if _matches_temporal_reference(event, prior)), None)
                if reference_target is not None:
                    # The phrase refers back to an established event; it is
                    # not a second occurrence to render in the sign sequence.
                    local_to_global[local_index] = reference_target["sequence_index"]
                    continue

            sequence_offset = global_index
            global_index += 1
            event["sequence_index"] = global_index
            ideal_event_sequence.append(event)
            agent = str(event.get("agent", "")).strip()
            if agent.lower() in {"he", "she", "they", "it", "we", "i"} and last_named_agent:
                event["agent"] = last_named_agent
            elif agent and agent.lower() not in {"he", "she", "they", "it", "we", "i"}:
                last_named_agent = agent
            for link in event.get("temporal_links", []):
                # Detector links are sentence-local; translate their target to
                # the page-wide event index.
                local_target = int(link.get("to_sequence_index", local_index))
                if local_target in local_to_global:
                    link["to_sequence_index"] = local_to_global[local_target]
                elif local_target == local_index:
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
            local_to_global[local_index] = global_index
            prior_events.append(event)
            video_result = generate_video_sequence({**process, "event": event})
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
            all_processes.append(event["process_name"])
            sentence_videos.extend(video_result["video_sequence"])
            ideal_sign_sequence.extend(video_result["ideal_sign_sequence"])
            renderable_sign_sequence.extend(video_result["renderable_sign_sequence"])
            fallbacks.extend(video_result["fallbacks"])
            diagnostics.extend(video_result["translation_diagnostics"])
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

        term_ideals, term_clips = _term_signs(sentence, [item["event"] for item in sentence_events])
        seen_paths = {clip["path"] for item in sentence_events
                      for clip in item["renderable_sign_sequence"]}
        for ideal, clip in zip(term_ideals, term_clips):
            if clip["path"] not in seen_paths:
                sentence_ideal_terms.append(ideal)
                sentence_term_clips.append(clip)
                seen_paths.add(clip["path"])
        if not sentence_events and not sentence_term_clips:
            diagnostics.append({
                "code": "UNMAPPED_TERM",
                "sentence": sentence,
                "message": "No keyword or synonym in the agricultural sign mapping matched this input.",
            })
        sentence_videos.extend(clip["path"] for clip in sentence_term_clips)
        ideal_sign_sequence.extend(sentence_ideal_terms)
        renderable_sign_sequence.extend(sentence_term_clips)
        complete_video_sequence.extend(sentence_videos)
        translation_results.append({
            "sentence": sentence,
            "sentence_index": sentence_index,
            "processes": sentence_events,
            "events": [item["event"] for item in sentence_events],
            "video_sequence": sentence_videos,
            "ideal_sign_sequence": [concept for item in sentence_events for concept in item["ideal_sign_sequence"]] + sentence_ideal_terms,
            "renderable_sign_sequence": [clip for item in sentence_events for clip in item["renderable_sign_sequence"]] + sentence_term_clips,
            "missing_sign_concepts": [concept for item in sentence_events for concept in item["missing_sign_concepts"]],
        })

    return {
        "total_sentences": len(sentences),
        "translations": translation_results,
        "complete_video_sequence": complete_video_sequence,
        "process_sequence": all_processes,
        "ideal_event_sequence": ideal_event_sequence,
        "ideal_sign_sequence": ideal_sign_sequence,
        "renderable_sign_sequence": renderable_sign_sequence,
        "missing_sign_concepts": list(missing_by_id.values()),
        "sign_fallbacks": fallbacks,
        "translation_diagnostics": diagnostics,
    }
