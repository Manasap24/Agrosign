import streamlit as st
import pandas as pd
import re
import streamlit.components.v1 as components
from difflib import get_close_matches
from pathlib import Path

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
# PROJECT PATHS
# ---------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "sign_videos"

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
@st.cache_data(ttl=0)
def load_dataset():
    try:
        dataset_path = BASE_DIR / "dataset" / "agro_terms.csv"
        df = pd.read_csv(dataset_path)
        df["keyword"]    = df["keyword"].fillna("").str.lower().str.strip()
        df["synonyms"]   = df["synonyms"].fillna("").str.lower().str.strip()
        df["video_path"] = df["video_path"].fillna("").str.strip()
        return df
    except Exception as e:
        st.error(f"Dataset loading error: {e}")
        st.stop()

df       = load_dataset()
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
# SIMPLE STEMMER
# ---------------------------------------------------
def simple_stem(word):
    suffixes = ["ing", "tion", "ion", "ation", "ed", "er", "s", "es"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word

# ---------------------------------------------------
# STOPWORDS
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
    "will", "would", "could", "should", "may", "might",
}

# ---------------------------------------------------
# TEXT PREPROCESSING
# ---------------------------------------------------
def preprocess(text):
    text   = text.lower()
    text   = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]

def get_variants(word):
    stemmed  = simple_stem(word)
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
    word_embedding = model.encode([word])
    scores         = cosine_similarity(word_embedding, keyword_embeddings)[0]
    best_index     = scores.argmax()
    best_score     = scores[best_index]
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

# Debug expander
with st.expander("🔧 Debug Info"):
    st.write(f"**BASE_DIR:** `{BASE_DIR}`")
    st.write(f"**VIDEO_DIR:** `{VIDEO_DIR}`")
    st.write(f"**VIDEO_DIR exists:** `{VIDEO_DIR.exists()}`")
    st.dataframe(df[["keyword", "video_path"]].head(5))

text = st.text_input("Enter an agriculture-related sentence:")

# ---------------------------------------------------
# PROCESS INPUT
# ---------------------------------------------------
if text:
    tokens = preprocess(text)

    if not tokens:
        st.warning("No meaningful words found after preprocessing.")
    else:
        st.markdown(f"**🔤 Preprocessed tokens:** `{' → '.join(tokens)}`")

        detected       = []
        unknown_words  = []
        semantic_count = 0
        match_info     = {}

        # FIRST: collect silently
        for word in tokens:
            result, match_type = find_match(word)
            score = None

            if result is None:
                result, score = semantic_match(word)
                if result is not None:
                    match_type     = "bert"
                    semantic_count += 1

            if result is not None:
                keyword        = result["keyword"]
                raw_video_path = result["video_path"].strip()
                video_path     = (BASE_DIR / raw_video_path).resolve()

                if keyword not in detected:
                    detected.append(keyword)
                    match_info[keyword] = {
                        "match_type": match_type,
                        "video_path": video_path
                    }
            else:
                suggestion = get_close_matches(word, keywords, n=1, cutoff=0.6)
                if suggestion:
                    st.warning(f"No sign for **'{word}'**. Did you mean: **{suggestion[0]}**?")
                else:
                    unknown_words.append(word)

        # SECOND: playlist at top
        if detected:
            st.subheader("🎬 Full Sign Sequence (All Words Combined)")
            import base64

            def video_to_base64(path):
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode()

            playlist        = []
            playlist_labels = []
            for kw in detected:
                row = df[df["keyword"] == kw]
                if not row.empty:
                    raw = row.iloc[0]["video_path"].strip()
                    vp  = (BASE_DIR / raw).resolve()
                    if vp.exists():
                        playlist.append(str(vp))
                        playlist_labels.append(kw)

            if playlist:
                sources_js = "[" + ",".join([f'"{video_to_base64(p)}"' for p in playlist]) + "]"
                labels_js  = "[" + ",".join([f'"{playlist_labels[i]}"' for i in range(len(playlist))]) + "]"

                html = f"""
                <div style="text-align:center;">
                    <div style="margin-bottom:8px;font-weight:bold;font-size:1.1em">
                        Now Signing: <span id="label" style="color:#2e7d32"></span>
                    </div>
                    <video id="player" width="480" autoplay playsinline
                        style="border-radius:12px;border:2px solid #2e7d32">
                    </video>
                    <div style="margin-top:8px;color:gray;font-size:0.9em">
                        Word <span id="cur">1</span> of <span id="tot"></span>
                    </div>
                </div>
                <script>
                    const sources = {sources_js};
                    const labels  = {labels_js};
                    const video   = document.getElementById("player");
                    const label   = document.getElementById("label");
                    const cur     = document.getElementById("cur");
                    const tot     = document.getElementById("tot");

                    let index = 0;
                    tot.textContent = sources.length;

                    function playNext() {{
                        if (index >= sources.length) index = 0;
                        label.textContent = labels[index];
                        cur.textContent   = index + 1;
                        video.src = "data:video/mp4;base64," + sources[index];
                        video.load();
                        video.play();
                        index++;
                    }}

                    video.addEventListener("ended", playNext);
                    playNext();
                </script>
                """
                st.components.v1.html(html, height=420)
            else:
                st.warning("No valid videos found for playlist.")

        # THIRD: individual signs below
        st.subheader("🖐️ Detected Signs")
        for kw in detected:
            info       = match_info[kw]
            match_type = info["match_type"]
            video_path = info["video_path"]

            if match_type == "exact":
                badge       = "🟢 Exact Match"
                badge_color = "#2e7d32"
            elif match_type == "synonym":
                badge       = "🔵 Synonym Match"
                badge_color = "#1565c0"
            else:
                badge       = "🧠 Semantic Match"
                badge_color = "#6a1b9a"

            st.markdown(
                f"<span style='background:{badge_color};color:white;"
                f"padding:3px 10px;border-radius:12px;font-size:0.85em'>"
                f"{badge}</span>  **{kw}**",
                unsafe_allow_html=True,
            )

            if video_path.exists():
                st.video(str(video_path))
            else:
                st.warning(f"Video not found: `{video_path}`")

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
        col1.metric("Signs Detected",   len(detected))
        col2.metric("Semantic Matches", semantic_count)
        col3.metric("Unknown Words",    len(unknown_words))

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