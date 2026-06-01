import streamlit as st
import pandas as pd
import re
from difflib import get_close_matches

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
@st.cache_data
def load_dataset():
    try:
        from pathlib import Path

        BASE_DIR = Path(__file__).resolve().parent.parent
        DATASET_PATH = BASE_DIR / "dataset" / "agro_terms.csv"

        df = pd.read_csv(DATASET_PATH)
        df["keyword"] = df["keyword"].fillna("").str.lower().str.strip()
        df["synonyms"] = df["synonyms"].fillna("").str.lower().str.strip()
        df["video_path"] = df["video_path"].fillna("")
        return df
    except Exception as e:
        st.error(f"Dataset loading error: {e}")
        st.stop()

df = load_dataset()
keywords = df["keyword"].tolist()

# ---------------------------------------------------
# LOAD SENTENCE-BERT MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------------------------------
# CREATE BERT EMBEDDINGS (keyword + synonyms)
# ---------------------------------------------------
@st.cache_resource
def build_embeddings(_model, _df):
    search_texts = []
    for _, row in _df.iterrows():
        combined = (
            str(row["keyword"]) + " " +
            str(row["synonyms"]).replace("|", " ")
        )
        search_texts.append(combined)
    return _model.encode(search_texts)

keyword_embeddings = build_embeddings(model, df)

# ---------------------------------------------------
# SIMPLE STEMMER (no library needed)
# ---------------------------------------------------
def simple_stem(word):
    """Strip common suffixes so irrigating->irrigat, irrigation->irrigat"""
    suffixes = ["ing", "tion", "ion", "ation", "ed", "er", "s", "es"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word

# ---------------------------------------------------
# STOPWORDS (no NLTK needed)
# ---------------------------------------------------
STOPWORDS = {
    "is", "are", "the", "in", "on", "at", "of",
    "to", "and", "a", "an", "for", "with",
    "was", "were", "be", "been", "being",
    "used", "using", "use", "make", "making",
    "do", "doing", "by", "from", "into",
    "over", "under", "between",
    "his", "her", "their", "its",
    "this", "that", "these", "those",
    "get", "getting", "has", "have", "had",
    "will", "would", "could", "should", "may", "might"
}

# ---------------------------------------------------
# TEXT PREPROCESSING
# ---------------------------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    filtered = [t for t in tokens if t not in STOPWORDS]
    return filtered

def get_variants(word):
    """Return original word + stemmed version for matching."""
    stemmed = simple_stem(word)
    variants = [word]
    if stemmed != word:
        variants.append(stemmed)
    return variants

# ---------------------------------------------------
# EXACT + SYNONYM MATCH
# ---------------------------------------------------
def find_match(word):
    for variant in get_variants(word):
        match = df[df["keyword"] == variant]
        if not match.empty:
            return match.iloc[0], "exact"

        syn_match = df[
            df["synonyms"].apply(
                lambda x: variant in [s.strip() for s in x.split("|")]
            )
        ]
        if not syn_match.empty:
            return syn_match.iloc[0], "synonym"

    return None, None

# ---------------------------------------------------
# BERT SEMANTIC MATCH
# ---------------------------------------------------
def semantic_match(word, threshold=0.45):
    """
    Uses Sentence-BERT (all-MiniLM-L6-v2) to find
    the closest keyword via cosine similarity.
    """
    word_embedding = model.encode([word])
    scores = cosine_similarity(word_embedding, keyword_embeddings)[0]
    best_index = scores.argmax()
    best_score = scores[best_index]

    if best_score >= threshold:
        return df.iloc[best_index], best_score

    return None, 0

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("🌾 AgroSign AI")
st.markdown("### Agriculture Text → Sign Language System")

st.markdown(
    """
    <div style='background:#f0f7f0;padding:10px 16px;border-radius:8px;
    border-left:4px solid #2e7d32;margin-bottom:16px'>
    🧠 <b>Powered by Sentence-BERT</b> (all-MiniLM-L6-v2) + Semantic Matching
    </div>
    """,
    unsafe_allow_html=True,
)

text = st.text_input("Enter an agriculture-related sentence:")

# ---------------------------------------------------
# PROCESS INPUT
# ---------------------------------------------------
if text:
    tokens = preprocess(text)

    if not tokens:
        st.warning("No meaningful words found after preprocessing.")
    else:
        st.markdown(
            f"**🔤 Preprocessed tokens:** `{' → '.join(tokens)}`"
        )
        st.subheader("🖐️ Detected Signs")

        detected = []
        unknown_words = []
        semantic_count = 0

        for word in tokens:
            result, match_type = find_match(word)
            score = None

            # Fall back to BERT if no exact/synonym match
            if result is None:
              result, score = semantic_match(word)

            if result is not None:
             match_type = "bert"
             semantic_count += 1

            if result is not None:
                keyword = result["keyword"]
                video_path = result["video_path"]

                if keyword not in detected:
                    detected.append(keyword)

                    # Fix path since app.py runs from src/
                    if video_path and not video_path.startswith("../"):
                        video_path = "../" + video_path

                    if match_type == "exact":
                        badge = "🟢 Exact Match"
                        badge_color = "#2e7d32"
                    elif match_type == "synonym":
                        badge = "🔵 Synonym Match"
                        badge_color = "#1565c0"
                    else:
                       badge = "🧠 Semantic Match"
                    badge_color = "#6a1b9a"

                    st.info(
                        "AI-assisted semantic matching enabled for agriculture terminology."
                      )

                    try:
                        st.video(video_path)
                    except Exception:
                        st.warning(f"Video not found: {video_path}")

            else:
                suggestion = get_close_matches(
                    word, keywords, n=1, cutoff=0.6
                )
                if suggestion:
                    st.warning(
                        f"No sign for **'{word}'**. "
                        f"Did you mean: **{suggestion[0]}**?"
                    )
                else:
                    unknown_words.append(word)

        # ---------------------------------------------------
        # UNKNOWN WORDS
        # ---------------------------------------------------
        if unknown_words:
            st.subheader("❓ Unknown Words")
            for word in unknown_words:
                st.error(f"No sign found for: **{word}**")

        # ---------------------------------------------------
        # PIPELINE SUMMARY
        # ---------------------------------------------------
        st.subheader("📊 Pipeline Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Signs Detected", len(detected))
        col2.metric("Semantic Matches", semantic_count)
        col3.metric("Unknown Words", len(unknown_words))

        if detected:
            st.success(f"✅ Detected Keywords: **{', '.join(detected)}**")

        with st.expander("🔬 How the pipeline works"):
            st.markdown(
                """
**Step 1 — Preprocessing**
- Lowercasing → Punctuation removal → Tokenization → Stopword removal

**Step 2 — Exact / Synonym Match**
- Checks keyword column directly, then synonym column (pipe-separated)

**Step 3 — Semantic Understanding Layer**
- Uses Sentence-BERT embeddings
- Understands related agriculture concepts
- Finds closest keyword using cosine similarity

Examples:
- irrigating → water
- cultivation → farming

**Step 4 — Fuzzy Fallback**
- `difflib.get_close_matches` for typo-tolerant suggestions
                """
            )