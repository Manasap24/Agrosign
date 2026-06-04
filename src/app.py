import streamlit as st
from dataset import load_dataset
import streamlit.components.v1 as components
from difflib import get_close_matches
from config import BASE_DIR, VIDEO_DIR
from preprocessing import preprocess, get_variants
from embeddings import load_model, build_embeddings
from matching import find_match, semantic_match




# ---------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------
st.set_page_config(
    page_title="AgroSign AI",
    page_icon="🌾",
    layout="centered"
)


df = load_dataset()
keywords = df["keyword"].tolist()


model = load_model()
keyword_embeddings = build_embeddings(model, df)


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
            result, match_type = find_match(word, df)
            score = None

            if result is None:
                result, score = semantic_match(word, model, keyword_embeddings, df)
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
