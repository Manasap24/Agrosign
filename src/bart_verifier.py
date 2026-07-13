from functools import lru_cache
from transformers import pipeline


@lru_cache(maxsize=1)
def load_bart_model():
    return pipeline("text-classification", model="facebook/bart-large-mnli")


classifier = load_bart_model()


def verify_process(sentence, process_name):
    hypothesis = f"This sentence describes the agricultural process " f"{process_name}."

    result = classifier({"text": sentence, "text_pair": hypothesis})

    label = result["label"]
    score = result["score"]

    if label.lower() == "entailment":
        return True, score

    return False, score
