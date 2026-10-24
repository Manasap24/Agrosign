"""Lazy, local-only BART-MNLI verification for shortlisted process candidates."""

from __future__ import annotations

import logging
from functools import lru_cache

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

LOGGER = logging.getLogger(__name__)
MODEL_ID = "facebook/bart-large-mnli"


@lru_cache(maxsize=1)
def load_bart_model():
    """Load BART only on demand and never start a network download at runtime."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, local_files_only=True
        )
        return pipeline(
            "zero-shot-classification", model=model, tokenizer=tokenizer
        )
    except Exception as exc:
        # BART is an optional second-stage verifier. A missing local checkpoint
        # must not take down text translation or cause repeated load attempts.
        LOGGER.warning("BART verifier unavailable; retaining first-stage result: %s", exc)
        return None


def classify_processes(sentence: str, process_names: list[str]) -> dict[str, float] | None:
    """Return independent entailment scores for a shortlist of process labels.

    `multi_label=True` is important here: candidate labels are scored against
    entailment/contradiction independently instead of forcing one winner among
    the candidates. The premise is the complete action clause, so a negated
    instruction can still be recognized as concerning the underlying process.
    """
    labels = list(dict.fromkeys(str(name).strip() for name in process_names if str(name).strip()))
    if not labels:
        return {}

    classifier = load_bart_model()
    if classifier is None:
        return None

    result = classifier(
        sentence,
        candidate_labels=labels,
        multi_label=True,
        hypothesis_template="This sentence concerns the agricultural process of {}.",
    )
    return {str(label): float(score) for label, score in zip(result["labels"], result["scores"])}


def verify_process(sentence: str, process_name: str, description: str = "", context_examples: str = ""):
    """Compatibility wrapper; callers should prefer shortlist classification."""
    scores = classify_processes(sentence, [process_name])
    if scores is None:
        return None, None
    score = scores.get(process_name, 0.0)
    return score >= 0.50, score
