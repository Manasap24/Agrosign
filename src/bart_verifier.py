from functools import lru_cache
from transformers import pipeline


@lru_cache(maxsize=1)
def load_bart_model():
    return pipeline("text-classification", model="facebook/bart-large-mnli")


def verify_process(sentence, process_name):
    hypothesis = f"This sentence describes the agricultural process " f"{process_name}."
    classifier = load_bart_model()

    result = classifier({"text": sentence, "text_pair": hypothesis})

    label = result[0]["label"]
    score = result[0]["score"]

    if label == "ENTAILMENT":
        return True, score

    return False, score
