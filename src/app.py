import streamlit as st
import pandas as pd

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

st.title("🌾 AgroSign AI")

text = st.text_input("Enter agriculture text:")

if text:
    words = text.lower().split()

    st.subheader("Detected Signs:")

    for word in words:   # ✅ NOW INSIDE

        # Basic normalization
        if word.endswith("ing"):
            word = word[:-3]

        if word.endswith("s"):
            word = word[:-1]

        # Apply synonym mapping
        if word in synonyms:
            word = synonyms[word]

        match = df[df["keyword"] == word]

        if not match.empty:
            video_path = match.iloc[0]["video_path"]
            st.write(f"👉 {word}")
            st.video(video_path)