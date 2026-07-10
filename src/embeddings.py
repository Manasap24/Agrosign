from sentence_transformers import SentenceTransformer
import streamlit as st


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


import pandas as pd


def load_processes():
    """
    Load the agricultural process dataset.
    """
    return pd.read_csv("datasets/agro_processes.csv")


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
