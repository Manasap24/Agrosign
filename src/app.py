import streamlit as st
import pandas as pd
import re
from difflib import get_close_matches

# ---------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------
st.set_page_config(
    page_title="AgroSign AI",
    page_icon="🌾",
    layout="centered"
)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
try:
    df = pd.read_csv("dataset/agro_terms.csv")
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()


keywords = df["keyword"].tolist()

# ---------------------------------------------------
# CLEAN DATASET
# ---------------------------------------------------
df["keyword"] = df["keyword"].fillna("").str.lower().str.strip()

df["synonyms"] = (
    df["synonyms"]
    .fillna("")
    .str.lower()
    .str.strip()
)

df["video_path"] = df["video_path"].fillna("")

# ---------------------------------------------------
# TEXT CLEANING FUNCTION
# ---------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

# ---------------------------------------------------
# FIND MATCH FUNCTION
# ---------------------------------------------------
def find_match(word):

    # Exact match
    keyword_match = df[df["keyword"] == word]
    if not keyword_match.empty:
        return keyword_match.iloc[0]

    # Synonym match
    synonym_match = df[
        df["synonyms"].apply(
            lambda x: word in [s.strip() for s in x.split("|")]
        )
    ]

    if not synonym_match.empty:
        return synonym_match.iloc[0]

    return None

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("🌾 AgroSign AI")
st.markdown("### Agriculture Text to Sign Language System")

text = st.text_input("Enter agriculture-related sentence:")

# ---------------------------------------------------
# PROCESS INPUT
# ---------------------------------------------------
if text:

    cleaned_text = clean_text(text)
    words = cleaned_text.split()

    detected = []
    unknown_words = []

    st.subheader("Detected Signs")

    stopwords = [
        "is", "are", "the", "in", "on", "at", "of",
        "to", "and", "a", "an", "for", "with"
    ]

    for word in words:

        # 🚫 Skip stopwords
        if word in stopwords:
            continue

        result = find_match(word)

        if result is not None:
            keyword = result["keyword"]
            video_path = result["video_path"]

            if keyword not in detected:
                detected.append(keyword)
                st.markdown(f"### 👉 {keyword}")
                try:
                    st.video(video_path)
                except:
                    st.warning(f"Video not found: {video_path}")

        else:
            suggestion = get_close_matches(word, keywords, n=1, cutoff=0.6)

            if suggestion:
                    st.warning(f"No sign for '{word}'. Did you mean: {suggestion[0]}?")
            else:
                unknown_words.append(word)

    # ---------------------------------------------------
    # UNKNOWN WORDS
    # ---------------------------------------------------
    if unknown_words:
        st.subheader("Unknown Words")
        for word in unknown_words:
            st.warning(f"No sign found for: {word}")

    # ---------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------
    st.subheader("Summary")

    if detected:
        st.success("Detected Keywords:")
        st.write(", ".join(detected))
    else:
        st.error("No matching agriculture signs found.")