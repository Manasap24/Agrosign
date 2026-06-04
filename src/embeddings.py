import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")





# ---------------------------------------------------
# CREATE BERT EMBEDDINGS (keyword + synonyms)
# ---------------------------------------------------
@st.cache_resource
def build_embeddings(_model, _df):
    search_texts = []
    for _, row in _df.iterrows():
        combined = str(row["keyword"]) + " " + str(row["synonyms"]).replace("|", " ")
        search_texts.append(combined)
    return _model.encode(search_texts)



