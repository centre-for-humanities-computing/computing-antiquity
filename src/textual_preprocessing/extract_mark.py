from typing import List

import pandas as pd
import spacy
from utils.nlp.greek import normalize


def token_filter(token: spacy.tokens.Token, filter_stops: bool = True) -> bool:
    return (
        not (token.is_stop and filter_stops)
        and not token.is_punct
        and not token.is_digit
        and not token.is_space
        and not token.is_quote
        and not token.is_bracket
        and not token.is_currency
    )


def doc_to_sentences(
    doc: spacy.tokens.Doc, filter_stops: bool = True
) -> List[List[str]]:
    """Turns spaCy document into a list of sentences in the form
    of list of tokens.

    Parameters
    ----------
    doc: spacy.Doc
        Spacy document object.
    filter_stops: bool, default True
        Specifies whether stopwords should be filtered.

    Return
    ------
    list of list of str
        List of sentences in the document.
    """
    sentences = []
    for sent in doc.sents:
        sentence = []
        for token in sent:
            if token_filter(token, filter_stops):
                token_text = normalize(token.lemma_, keep_sentences=False)
                token_text = token_text.strip()
                if token_text:
                    sentence.append(token_text)
        if sentence:
            sentences.append(sentence)
    return sentences


def doc_path_to_sentences(
    path: str, filter_stops: bool = True
) -> List[List[str]]:
    """Loads SpaCy document from disk and turns it into a list of
    sentences in the form of a list of tokens.

    Parameters
    ----------
    path: str
        Path to the spaCy document.
    filter_stops: bool, default True
        Specifies whether stopwords should be filtered.

    Return
    ------
    list of list of str
        List of sentences in the document.
    """
    doc = spacy.tokens.Doc(spacy.vocab.Vocab()).from_disk(path)
    return doc_to_sentences(doc)


def join_sentences(sentences: List[List[str]]) -> str:
    """Turns a document of sentences to a single string."""
    return "\n".join(" ".join(sentence) for sentence in sentences)


MARK_PATH = "/work/data_wrangling/dat/greek/clean_data/SEPA_NA28-041MRK"

mark_with_stopwords = join_sentences(
    doc_path_to_sentences(MARK_PATH, filter_stops=False)
)
mark_without_stopwords = join_sentences(
    doc_path_to_sentences(MARK_PATH, filter_stops=True)
)

mark_without_stopwords

doc = spacy.tokens.Doc(spacy.vocab.Vocab()).from_disk(MARK_PATH)

doc

records = []
for token in doc:
    if normalize(token.text, keep_sentences=False).strip():
        records.append(
            dict(
                token=token.text,
                lemma=token.lemma_.lower(),
                pos=token.pos_,
                tag=token.tag_,
                dep=token.dep_,
                is_sentence_start=token.is_sent_start,
                is_sentence_end=token.is_sent_end,
            )
        )
df = pd.DataFrame.from_records(records)

df

df.to_csv("mark_spacy.csv")
