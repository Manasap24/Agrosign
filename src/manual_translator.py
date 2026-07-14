from preprocessing import split_sentences
from process_detector import detect_process
from sign_sequence_generator import generate_video_sequence


def translate_manual(text):
    """
    Translate an agricultural manual (sentence, paragraph,
    or entire page) into a continuous sequence of sign videos.
    """

    # Step 1: Split input into sentences
    sentences = split_sentences(text)

    translation_results = []
    complete_video_sequence = []

    # Step 2: Process every sentence independently
    for sentence in sentences:

        # Detect agricultural process
        process_result = detect_process(sentence)

        # Generate sign video sequence
        video_result = generate_video_sequence(process_result)

        # Store complete result for this sentence
        translation_results.append(
            {
                "sentence": sentence,
                "process_name": video_result["process_name"],
                "category": video_result["category"],
                "semantic_confidence": video_result["semantic_confidence"],
                "bart_verified": video_result["bart_verified"],
                "bart_score": video_result["bart_score"],
                "video_sequence": video_result["video_sequence"],
            }
        )

        # Merge all videos into one continuous playlist
        complete_video_sequence.extend(video_result["video_sequence"])

    # Step 3: Return final translation
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
