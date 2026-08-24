from sentence_transformers import util

from .bart_verifier import verify_process
from .preprocessing import preprocess
from .embeddings import (
    load_model,
    load_processes,
    build_process_embeddings,
)

model = load_model()

process_df = load_processes()

process_df, process_embeddings = build_process_embeddings(model, process_df)


def generate_query_embedding(sentence):
    processed = preprocess(sentence)
    processed_text = " ".join(processed[0])

    query_embedding = model.encode(processed_text, convert_to_tensor=True)

    return query_embedding


def find_best_process(query_embedding):
    similarity_scores = util.cos_sim(query_embedding, process_embeddings)

    best_index = similarity_scores.argmax().item()

    best_process = process_df.iloc[best_index]

    confidence = similarity_scores[0][best_index].item()

    return best_process, confidence


def detect_process(sentence):

    query_embedding = generate_query_embedding(sentence)

    best_process, confidence = find_best_process(query_embedding)

    verified, bart_score = verify_process(sentence, best_process["process_name"])

    return {
        "process_name": best_process["process_name"],
        "category": best_process["category"],
        "sign_sequence": best_process["sign_sequence"],
        "semantic_confidence": confidence,
        "bart_verified": verified,
        "bart_score": bart_score,
    }


if __name__ == "__main__":

    result = detect_process("The farmer supplied water through drip irrigation.")

    print(result)
