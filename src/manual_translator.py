from .preprocessing import split_sentences
from .process_detector import detect_process
from .sign_sequence_generator import generate_video_sequence


def translate_manual(text):
    """
    Translate an agricultural manual into a continuous
    sequence of sign videos.
    """

    sentences = split_sentences(text)

    translation_results = []
    complete_video_sequence = []

    for sentence in sentences:

        process_result = detect_process(sentence)

        sentence_processes = []
        sentence_videos = []

        for process in process_result["processes"]:

            video_result = generate_video_sequence(process)

            sentence_processes.append(
                {
                    "process_name": video_result["process_name"],
                    "category": video_result["category"],
                    "semantic_confidence": video_result["semantic_confidence"],
                    "bart_verified": video_result["bart_verified"],
                    "bart_score": video_result["bart_score"],
                    "video_sequence": video_result["video_sequence"],
                }
            )

            sentence_videos.extend(video_result["video_sequence"])

        translation_results.append(
            {
                "sentence": sentence,
                "processes": sentence_processes,
                "video_sequence": sentence_videos,
            }
        )

        complete_video_sequence.extend(sentence_videos)

    return {
        "total_sentences": len(sentences),
        "translations": translation_results,
        "complete_video_sequence": complete_video_sequence,
    }


if __name__ == "__main__":

    sample_text = """
    Prepare the soil by ploughing and leveling.
    Apply organic manure before sowing.
    Irrigate the field regularly.
    Harvest the mature crop carefully.
    """

    result = translate_manual(sample_text)

    from pprint import pprint

    pprint(result)
