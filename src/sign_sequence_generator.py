from functools import lru_cache

from .database import db

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "sign_videos"


@lru_cache(maxsize=1)
def load_sign_videos():
    records = list(db["agro_terms"].find({}, {"_id": 0, "keyword": 1, "video_path": 1}))

    return {
        record["keyword"].strip().lower(): record["video_path"].strip()
        for record in records
    }


def generate_video_sequence(process_result):
    """
    Generate the ordered list of sign video paths
    for the detected agricultural process.
    """

    sign_sequence = process_result["sign_sequence"]
    signs = sign_sequence.split("|")

    video_mapping = load_sign_videos()

    video_sequence = []

    for sign in signs:
        sign = sign.strip().lower()

        video_path = video_mapping.get(sign)

        if video_path:
            video_sequence.append(str(BASE_DIR / video_path))

    return {
        "process_name": process_result["process_name"],
        "category": process_result["category"],
        "video_sequence": video_sequence,
        "semantic_confidence": process_result["semantic_confidence"],
        "bart_verified": process_result["bart_verified"],
        "bart_score": process_result["bart_score"],
    }
