"""
text_preprocessor.py  (your preprocessing file, cleaned up + ISL ordering)

Two functions:
    preprocess(text)      -> normal lemma lists, for embedding / process detection
    preprocess_isl(text)  -> same tokens, reordered into ISL order, for the sign sequence

Requires isl_grammar.py in the same folder (for the word lists).
"""

import re
import spacy
from isl_grammar import TIME_WORDS, QUESTION_WORDS, NEGATIONS, DOMAIN_VERBS

try:
    from preprocessing import nlp          # reuse the model your project already loaded
except ImportError:
    nlp = spacy.load("en_core_web_sm")

# spaCy treats these as stop words, but ISL needs them (they carry
# negation and question meaning), so switch the stop flag off.
KEEP_WORDS = {"not", "no", "never", "cannot", "n't"} | QUESTION_WORDS | TIME_WORDS
for _w in KEEP_WORDS:
    nlp.vocab[_w].is_stop = False

CLAUSE_BREAKS = {"and", "then", "before"}   # split multi-action sentences


def split_sentences(text):
    return [s.text.strip() for s in nlp(text).sents]


def clean_sentence(sentence):
    return re.sub(r"\s+", " ", sentence.lower()).strip()


def _keep(token):
    """True if this token should survive preprocessing."""
    if token.is_punct or token.is_space or token.like_num:
        return False
    if token.is_stop or token.pos_ == "AUX":
        return False
    return token.lemma_.isalpha()


def preprocess(text):
    """
    List[List[str]] - one lemma list per sentence (original word order).
    Negation, question and time words are now kept.
    """
    result = []
    for sentence in split_sentences(text):
        doc = nlp(clean_sentence(sentence))
        result.append([t.lemma_.lower() for t in doc if _keep(t)])
    return result


def _reorder_clause(tokens):
    """tokens: list of (lemma, pos). Returns ISL-ordered lemma list."""
    time_part, content, verbs, neg, question = [], [], [], False, []

    for lemma, pos in tokens:
        if lemma in NEGATIONS or lemma == "not":
            neg = True
        elif lemma in QUESTION_WORDS:
            question.append(lemma)
        elif lemma in TIME_WORDS:
            time_part.append(lemma)
        elif pos == "VERB":
            verbs.append(lemma)
        else:
            content.append((lemma, pos))

    # spaCy sometimes tags an agriculture verb as a noun ("farmer sprays ...").
    # If no verb was found, promote the first known domain verb.
    if not verbs:
        for i, (lemma, pos) in enumerate(content):
            if lemma in DOMAIN_VERBS:
                verbs.append(lemma)
                content.pop(i)
                break

    out = time_part + [l for l, _ in content] + verbs
    if neg:
        out.append("not")
    return out + question


def preprocess_isl(text):
    """
    List[List[str]] - one ISL-ordered token list per sentence:
    TIME + SUBJECT/OBJECT + VERB + NOT + QUESTION
    """
    result = []
    for sentence in split_sentences(text):
        doc = nlp(clean_sentence(sentence))

        clauses, current = [], []
        for i, t in enumerate(doc):
            # "and" splits clauses only when a verb follows ("plough and harvest"),
            # not for lists of nouns ("wheat and rice"). spaCy sometimes tags
            # farm verbs as nouns, so the DOMAIN_VERBS list is checked too.
            is_break = t.text in CLAUSE_BREAKS and (
                t.text != "and" or (i + 1 < len(doc) and (doc[i + 1].pos_ == "VERB"
                                        or doc[i + 1].lemma_.lower() in DOMAIN_VERBS)))
            if is_break:
                if current:
                    clauses.append(current)
                current = []
                continue
            if _keep(t):
                current.append((t.lemma_.lower(), t.pos_))
        if current:
            clauses.append(current)

        ordered = []
        for clause in clauses:
            ordered.extend(_reorder_clause(clause))
        result.append(ordered)
    return result


if __name__ == "__main__":
    samples = [
        "How to apply fertilizer in the morning?",
        "Do not water the crop.",
        "Farmer sprays pesticide.",
        "Sow seeds in the morning.",
        "The farmer prepared the land before sowing certified seeds.",
    ]
    for s in samples:
        print(s)
        print("  normal:", preprocess(s)[0])
        print("  ISL   :", preprocess_isl(s)[0])