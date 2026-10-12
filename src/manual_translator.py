"""Translate document text into ordered events and sign-video sequences."""

from preprocessing import split_sentences
from process_detector import detect_process_many
from sign_sequence_generator import generate_video_sequence


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

    for sentence_index, sentence in enumerate(sentences, start=1):
        detection = sentence_detections[sentence_index - 1]
        sentence_events = []
        sentence_videos = []
        for process in detection["processes"]:
            event = dict(process["event"])
            local_index = event.get("sequence_index", 1)
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

        complete_video_sequence.extend(sentence_videos)
        translation_results.append({
            "sentence": sentence,
            "sentence_index": sentence_index,
            "processes": sentence_events,
            "events": [item["event"] for item in sentence_events],
            "video_sequence": sentence_videos,
            "ideal_sign_sequence": [concept for item in sentence_events for concept in item["ideal_sign_sequence"]],
            "renderable_sign_sequence": [clip for item in sentence_events for clip in item["renderable_sign_sequence"]],
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
