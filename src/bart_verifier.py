# from functools import lru_cache
# from transformers import pipeline


# @lru_cache(maxsize=1)
# def load_bart_model():
#     return pipeline("text-classification", model="facebook/bart-large-mnli")


# classifier = load_bart_model()


# def verify_process(sentence, process_name):
#     hypothesis = f"This sentence describes the agricultural process " f"{process_name}."

#     result = classifier({"text": sentence, "text_pair": hypothesis})

#     label = result["label"]
#     score = result["score"]

#     if label.lower() == "entailment":
#         return True, score

#     return False, score


from functools import lru_cache

from transformers import pipeline


@lru_cache(maxsize=1)
def load_bart_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


classifier = load_bart_model()


def classify_processes(sentence, process_names):

    if not process_names:
        return {}

    result = classifier(
        sentence,
        candidate_labels=process_names,
        multi_label=False,
        hypothesis_template=("The agricultural process described is {}."),
    )

    return dict(zip(result["labels"], result["scores"]))


def verify_process(sentence, process_name, description="", context_examples=""):

    result = classify_processes(sentence, [process_name])

    score = result.get(process_name, 0.0)

    return score >= 0.50, score