import re
import spacy

# Load spaCy model only once
nlp = spacy.load("en_core_web_sm")


def split_sentences(text):
    """
    Split a document into individual sentences.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def clean_sentence(sentence):
    """
    Remove unwanted spaces and punctuation.
    """
    sentence = sentence.lower()
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def preprocess(text):
    """
    Preprocess a complete document.

    Returns:
        List[List[str]]

        Example:
        [
            ['farmer', 'prepare', 'land'],
            ['seed', 'sow'],
            ['irrigation', 'water']
        ]
    """

    processed_sentences = []

    sentences = split_sentences(text)

    for sentence in sentences:

        sentence = clean_sentence(sentence)

        doc = nlp(sentence)

        tokens = []

        for token in doc:

            # Ignore punctuation, numbers and spaces
            if token.is_punct or token.is_space or token.like_num:
                continue

            # Remove stopwords
            if token.is_stop:
                continue

            lemma = token.lemma_.lower()

            # Ignore single characters
            if not lemma.isalpha():
              continue

            tokens.append(lemma)

        processed_sentences.append(tokens)

    return processed_sentences


if __name__ == "__main__":

    sample_text = """
    The farmer prepared the land before sowing certified seeds.
    Drip irrigation was used to supply water regularly.
    Organic compost was applied to improve crop growth.
    """

    result = preprocess(sample_text)

    for i, sentence in enumerate(result, start=1):
        print(f"Sentence {i}: {sentence}")
