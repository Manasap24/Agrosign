from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent.parent / "sign_videos"


def generate_video_sequence(process_result):
    """
    Generate the ordered list of sign video paths
    for the detected agricultural process.
    """

    sign_sequence = process_result["sign_sequence"]
    signs = sign_sequence.split("|")

    video_sequence = []

    for sign in signs:
        video_path = VIDEO_DIR / f"{sign}.mp4"

        video_sequence.append(str(video_path))

    return {
        "process_name": process_result["process_name"],
        "category": process_result["category"],
        "video_sequence": video_sequence,
        "semantic_confidence": process_result["semantic_confidence"],
        "bart_verified": process_result["bart_verified"],
        "bart_score": process_result["bart_score"],
    }


if __name__ == "__main__":

    sample_result = {
        "process_name": "Irrigation",
        "category": "Crop Management",
        "sign_sequence": "irrigation|water",
        "semantic_confidence": 0.67,
        "bart_verified": True,
        "bart_score": 0.75,
    }

    result = generate_video_sequence(sample_result)

    print(result)
