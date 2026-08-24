from functools import lru_cache

import pandas as pd
from sentence_transformers import SentenceTransformer

from .database import db


@lru_cache(maxsize=1)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def load_processes():
    records = list(db["agro_processes"].find({}, {"_id": 0}))

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No agricultural processes found in MongoDB.")

    return df


def build_process_embeddings(model, df):

    process_texts = []

    for _, row in df.iterrows():

        combined_text = (
            f"{row['process_name']}. "
            f"{row['description']}. "
            f"{row['context_examples']}"
        )

        process_texts.append(combined_text)

    embeddings = model.encode(process_texts, convert_to_tensor=True)

    return df, embeddings
