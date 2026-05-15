import streamlit as st
import pandas as pd
import re
from difflib import get_close_matches

# Synonym dictionary
synonyms = {
    "plants": "crop",
    "plant": "crop",
    "harvest": "crop",
    "harvesting": "crop",
    "irrigation": "water",
    "watering": "water",
    "farms": "farm",
    "farmers": "farmer"
}

# Load dataset
df = pd.read_csv("dataset/agro_terms.csv")

keywords = df["keyword"].tolist()

st.title("🌾 AgroSign AI")

text = st.text_input("Enter agriculture text:")

if text:
   

    # Remove punctuation and split properly
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = clean_text.split()

    stopwords = ["is", "the", "in", "are", "and", "of", "to"]

    words = [word for word in words if word not in stopwords]

    st.subheader("Detected Signs:")

    for word in words:

        # Basic normalization
        if word.endswith("ing"):
            word = word[:-3]

        if word.endswith("s"):
            word = word[:-1]

        # Synonym mapping
        if word in synonyms:
            word = synonyms[word]

        match = df[df["keyword"] == word]

        if not match.empty:
            video_path = match.iloc[0]["video_path"]
            st.write(f"👉 {word}")
            st.video(video_path)

        else:
            # 🔥 Smart suggestion part
            suggestion = get_close_matches(word, keywords, n=1, cutoff=0.6)

            if suggestion:
                st.write(f"❓ Did you mean: **{suggestion[0]}**?")