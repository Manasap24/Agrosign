import re
import math
import torch

from sentence_transformers import util

from .bart_verifier import verify_process
from .embeddings import load_model, load_processes

model = load_model()
process_df = load_processes()


STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "by",
    "from",
    "and",
    "or",
    "that",
    "this",
    "these",
    "those",
    "farmer",
    "farmers",
    "field",
    "crop",
    "crops",
    "soil",
    "plant",
    "plants",
}


def normalize_word(word):

    word = word.lower().strip()

    irregular = {
        "watering": "water",
        "watered": "water",
        "waters": "water",
        "irrigated": "irrigate",
        "irrigating": "irrigate",
        "irrigates": "irrigate",
        "irrigation": "irrigate",
        "sowing": "sow",
        "sowed": "sow",
        "sown": "sow",
        "sows": "sow",
        "seeds": "seed",
        "ploughing": "plough",
        "ploughed": "plough",
        "ploughs": "plough",
        "plowing": "plough",
        "plowed": "plough",
        "leveling": "level",
        "levelling": "level",
        "leveled": "level",
        "levelled": "level",
        "levels": "level",
        "manuring": "manure",
        "manured": "manure",
        "manures": "manure",
        "fertilizers": "fertilizer",
        "fertilising": "fertilize",
        "fertilizing": "fertilize",
        "fertilized": "fertilize",
    }

    if word in irregular:
        return irregular[word]

    if len(word) > 6 and word.endswith("ing"):
        return word[:-3]

    if len(word) > 5 and word.endswith("ed"):
        return word[:-2]

    if len(word) > 5 and word.endswith("es"):
        return word[:-2]

    if len(word) > 4 and word.endswith("s"):
        return word[:-1]

    return word


def tokenize(text):

    words = re.findall(r"[a-z]+", str(text).lower())

    return [
        normalize_word(word)
        for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]


def split_action_clauses(sentence):

    sentence = sentence.strip()

    parts = re.split(
        r"\s+(?:and|then|while|before|after|but)\s+", sentence, flags=re.IGNORECASE
    )

    parts = [part.strip(" ,.") for part in parts if part.strip(" ,.")]

    if len(parts) <= 1:
        return parts

    first = parts[0]

    match = re.match(
        r"^(.*?\b(?:farmer|worker|person|they|he|she)\b)", first, flags=re.IGNORECASE
    )

    subject = match.group(1) if match else "The farmer"

    clauses = [first]

    for part in parts[1:]:
        clauses.append(f"{subject} {part}")

    return clauses


def get_examples(row):

    value = row.get("context_examples", "")

    if not isinstance(value, str):
        return []

    return [x.strip() for x in value.split(";") if x.strip()]


def build_profiles():

    profiles = []

    for _, row in process_df.iterrows():

        examples = get_examples(row)

        profiles.append(
            {
                "name": str(row["process_name"]).strip(),
                "description": str(row.get("description", "")).strip(),
                "examples": examples,
            }
        )

    return profiles


profiles = build_profiles()


def build_embeddings():

    name_embeddings = []
    description_embeddings = []
    example_embeddings = []

    for profile in profiles:

        name_embeddings.append(model.encode(profile["name"], convert_to_tensor=True))

        description_embeddings.append(
            model.encode(profile["description"], convert_to_tensor=True)
        )

        if profile["examples"]:

            embeddings = model.encode(profile["examples"], convert_to_tensor=True)

        else:

            embeddings = torch.empty((0, 384))

        example_embeddings.append(embeddings)

    return (
        torch.stack(name_embeddings),
        torch.stack(description_embeddings),
        example_embeddings,
    )


name_embeddings, description_embeddings, example_embeddings = build_embeddings()


def build_document_frequency():

    document_frequency = {}

    for profile in profiles:

        text = " ".join(
            [profile["name"], profile["description"], " ".join(profile["examples"])]
        )

        words = set(tokenize(text))

        for word in words:

            document_frequency[word] = document_frequency.get(word, 0) + 1

    return document_frequency


document_frequency = build_document_frequency()

TOTAL_PROCESSES = len(profiles)


def idf(word):

    frequency = document_frequency.get(word, 0)

    return math.log((TOTAL_PROCESSES + 1) / (frequency + 1)) + 1


def weighted_overlap(sentence, text):

    sentence_words = tokenize(sentence)

    text_words = set(tokenize(text))

    if not sentence_words:
        return 0.0

    score = 0.0
    total = 0.0

    for word in sentence_words:

        weight = idf(word)

        total += weight

        if word in text_words:
            score += weight

    return score / total if total else 0.0


def best_example_lexical_score(clause, profile):

    if not profile["examples"]:
        return 0.0

    scores = [weighted_overlap(clause, example) for example in profile["examples"]]

    return max(scores)


def profile_lexical_score(clause, profile):

    evidence = " ".join(
        [profile["name"], profile["description"], " ".join(profile["examples"])]
    )

    return weighted_overlap(clause, evidence)


def get_best_example_score(clause, index):

    examples = profiles[index]["examples"]

    embeddings = example_embeddings[index]

    if not examples:
        return 0.0

    query_embedding = model.encode(clause, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, embeddings)[0]

    return scores.max().item()


def retrieve_candidates(clause, top_k=15):

    query_embedding = model.encode(clause, convert_to_tensor=True)

    name_scores = util.cos_sim(query_embedding, name_embeddings)[0]

    description_scores = util.cos_sim(query_embedding, description_embeddings)[0]

    candidates = []

    for index, profile in enumerate(profiles):

        semantic_name = name_scores[index].item()

        semantic_description = description_scores[index].item()

        example_semantic = get_best_example_score(clause, index)

        example_lexical = best_example_lexical_score(clause, profile)

        profile_lexical = profile_lexical_score(clause, profile)

        retrieval_score = (
            0.30 * example_lexical
            + 0.25 * profile_lexical
            + 0.25 * example_semantic
            + 0.10 * semantic_name
            + 0.10 * semantic_description
        )

        candidates.append(
            {
                "index": index,
                "process": process_df.iloc[index],
                "example_semantic": example_semantic,
                "example_lexical": example_lexical,
                "profile_lexical": profile_lexical,
                "name_semantic": semantic_name,
                "description_semantic": semantic_description,
                "retrieval_score": retrieval_score,
            }
        )

    candidates.sort(key=lambda x: x["retrieval_score"], reverse=True)

    return candidates[:top_k]


def verify_candidates(clause, candidates):

    if not candidates:
        return []

    results = []

    for candidate in candidates:

        process = candidate["process"]

        verified, bart_score = verify_process(
            clause,
            process["process_name"],
            str(process.get("description", "")),
            str(process.get("context_examples", "")),
        )

        final_score = 0.95 * candidate["retrieval_score"] + 0.05 * bart_score

        result = dict(candidate)

        result["bart_score"] = bart_score

        result["verified"] = verified

        result["final_score"] = final_score

        results.append(result)

    results.sort(key=lambda x: x["final_score"], reverse=True)

    return results


def select_process(clause):

    candidates = retrieve_candidates(clause)

    if not candidates:
        return None

    ranked = verify_candidates(clause, candidates)

    if not ranked:
        return None

    best = ranked[0]

    if best["retrieval_score"] < 0.30:
        return None

    return best


def detect_process(sentence):

    clauses = split_action_clauses(sentence)

    selected = []

    for clause in clauses:

        result = select_process(clause)

        if result is not None:
            selected.append(result)

    unique = {}

    for result in selected:

        name = result["process"]["process_name"]

        if name not in unique or result["final_score"] > unique[name]["final_score"]:

            unique[name] = result

    final = list(unique.values())

    final.sort(key=lambda x: x["final_score"], reverse=True)

    processes = []

    for result in final:

        process = result["process"]

        processes.append(
            {
                "process_name": process["process_name"],
                "category": process["category"],
                "sign_sequence": process["sign_sequence"],
                "semantic_confidence": result["example_semantic"],
                "bart_verified": result["verified"],
                "bart_score": result["bart_score"],
                "combined_score": result["final_score"],
            }
        )

    return {"processes": processes}


if __name__ == "__main__":

    tests = [
        "The farmer is watering the plants regularly.",
        "The farmer prepares the field by ploughing the soil and leveling it properly.",
        "The farmer then applies organic manure and sows healthy seeds at the appropriate depth.",
        "The farmer uses drip irrigation to water the crops.",
        "The farmer irrigates different parts of the field according to crop needs.",
    ]

    for sentence in tests:

        print("\n" + "=" * 70)

        print(sentence)

        print("=" * 70)

        result = detect_process(sentence)

        print("\nFINAL PROCESSES:")

        for process in result["processes"]:

            print(
                process["process_name"],
                "| semantic:",
                round(process["semantic_confidence"], 4),
                "| BART:",
                round(process["bart_score"], 4),
                "| verified:",
                process["bart_verified"],
            )
