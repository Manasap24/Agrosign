import pandas as pd
from .database import db


def load_dataset():
    records = list(db["agro_terms"].find({}, {"_id": 0}))

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No agricultural terms found in MongoDB.")

    df["keyword"] = df["keyword"].fillna("").str.lower().str.strip()
    df["synonyms"] = df["synonyms"].fillna("").str.lower().str.strip()
    df["video_path"] = df["video_path"].fillna("").str.strip()

    return df
