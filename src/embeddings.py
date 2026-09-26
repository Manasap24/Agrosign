# from sentence_transformers import SentenceTransformer
# from functools import lru_cache
# from pathlib import Path

# @lru_cache(maxsize=1)
# def load_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")


# import pandas as pd


# BASE_DIR = Path(__file__).resolve().parent.parent

# def load_processes():
#     """
#     Load the agricultural process dataset.
#     """
#     csv_path = BASE_DIR / "dataset" / "agro_processes.csv"
#     return pd.read_csv(csv_path)


# def build_process_embeddings(model, df):

#     process_texts = []

#     for _, row in df.iterrows():

#         combined_text = (
#             f"{row['process_name']}. "
#             f"{row['description']}. "
#             f"{row['context_examples']}"
#         )

#         process_texts.append(combined_text)
#     embeddings = model.encode(process_texts, convert_to_tensor=True)

#     return df, embeddings


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