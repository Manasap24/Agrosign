from .preprocessing import split_sentences
from .process_detector import detect_process
from .sign_sequence_generator import generate_video_sequence


def translate_manual(text):

    sentences = split_sentences(text)

    translation_results = []
    complete_video_sequence = []
    process_sequence = []

    for sentence in sentences:

        process_result = detect_process(sentence)

        sentence_processes = []
        sentence_videos = []

        for process in process_result.get("processes", []):

            video_result = generate_video_sequence(process)

            process_name = process["process_name"]

            sentence_processes.append(process_name)

            process_sequence.append(process_name)

            videos = video_result.get("video_sequence", [])

            sentence_videos.extend(videos)

            complete_video_sequence.extend(videos)

        translation_results.append(
            {
                "sentence": sentence,
                "process_name": sentence_processes,
                "processes": sentence_processes,
                "video_sequence": sentence_videos,
            }
        )

    return {
        "total_sentences": len(sentences),
        "translations": translation_results,
        "process_sequence": process_sequence,
        "complete_video_sequence": complete_video_sequence,
    }


if __name__ == "__main__":

    sample_text = """
    The farmer prepares the field by ploughing the soil
    and leveling it properly.

    The farmer then applies organic manure
    and sows healthy seeds at the appropriate depth.

    The farmer uses drip irrigation to water the crops.
    """

    result = translate_manual(sample_text)

    from pprint import pprint

    pprint(result)
