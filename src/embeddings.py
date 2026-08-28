from functools import lru_cache
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def load_processes():

    processes_path = BASE_DIR / "dataset" / "agro_processes.csv"

    df = pd.read_csv(processes_path)

    if df.empty:
        raise ValueError("No agricultural processes found in CSV.")

    return df


def build_process_embeddings(model, df):

    process_embeddings = []

    for _, row in df.iterrows():

        text = f"{row['process_name']}. " f"{row['description']}"

        embedding = model.encode(text, convert_to_tensor=True)

        process_embeddings.append(embedding)

    return (df, process_embeddings)
