# import pandas as pd
# import streamlit as st
# from config import BASE_DIR


# @st.cache_data(ttl=0)
# def load_dataset():
#     try:
#         dataset_path = BASE_DIR / "dataset" / "agro_terms.csv"
#         df = pd.read_csv(dataset_path)
#         df["keyword"] = df["keyword"].fillna("").str.lower().str.strip()
#         df["synonyms"] = df["synonyms"].fillna("").str.lower().str.strip()
#         df["video_path"] = df["video_path"].fillna("").str.strip()
#         return df
#     except Exception as e:
#         st.error(f"Dataset loading error: {e}")
#         st.stop()
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def load_dataset():
    terms_path = BASE_DIR / "dataset" / "agro_terms.csv"

    df = pd.read_csv(terms_path)

    if df.empty:
        raise ValueError("No agricultural terms found in CSV.")

    df["keyword"] = df["keyword"].fillna("").str.lower().str.strip()
    df["synonyms"] = df["synonyms"].fillna("").str.lower().str.strip()
    df["video_path"] = df["video_path"].fillna("").str.strip()

    return df