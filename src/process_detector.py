from sentence_transformers import util

from preprocessing import preprocess

from embeddings import (
    load_model,
    load_processes,
    build_process_embeddings,
)

# Load model and process database
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

    return {
        "process_name": best_process["process_name"],
        "category": best_process["category"],
        "sign_sequence": best_process["sign_sequence"],
        "confidence": confidence,
    }


result = detect_process("The farmer harvested mature crops.")

print(result)
