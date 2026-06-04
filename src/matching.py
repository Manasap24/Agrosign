from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import get_variants


# EXACT + SYNONYM MATCH
# ---------------------------------------------------
def find_match(word, df):
    for variant in get_variants(word):
        match = df[df["keyword"] == variant]
        if not match.empty:
            return match.iloc[0], "exact"

        syn_match = df[
            df["synonyms"].apply(lambda x: variant in [s.strip() for s in x.split("|")])
        ]
        if not syn_match.empty:
            return syn_match.iloc[0], "synonym"

    return None, None


# BERT SEMANTIC MATCH
# ---------------------------------------------------
def semantic_match(word, model, keyword_embeddings, df, threshold=0.45):
    word_embedding = model.encode([word])
    scores = cosine_similarity(word_embedding, keyword_embeddings)[0]
    best_index = scores.argmax()
    best_score = scores[best_index]
    if best_score >= threshold:
        return df.iloc[best_index], best_score
    return None, 0
