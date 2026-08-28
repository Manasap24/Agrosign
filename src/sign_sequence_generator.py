from functools import lru_cache
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "sign_videos"


@lru_cache(maxsize=1)
def load_sign_videos():
    terms_path = BASE_DIR / "dataset" / "agro_terms.csv"

    df = pd.read_csv(terms_path)

    return {
        row["keyword"].strip().lower(): row["video_path"].strip()
        for _, row in df.iterrows()
        if row["keyword"] and row["video_path"]
    }


def generate_video_sequence(process_result):

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
