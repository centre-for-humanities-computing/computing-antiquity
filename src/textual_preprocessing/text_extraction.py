"""Extracts sentences and contents from the parsed and cleaned corpus
both with and without stopwords.
"""
from typing import List
import os

import pandas as pd
import spacy

GREEK_DATA_PATH = "/work/data_wrangling/dat/greek/"
SAVE_PATH = "dataset"
CLEAN_INDEX_PATH = "clean_data/index.csv"


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
            if token.is_alpha and not (token.is_stop and filter_stops):
                sentence.append(token.lemma_.lower())
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


def main() -> None:
    print("Loading index data about cleaned files.")
    cleaned_index = pd.read_csv(
        os.path.join(GREEK_DATA_PATH, CLEAN_INDEX_PATH)
    )
    out_path = os.path.join(GREEK_DATA_PATH, SAVE_PATH)
    for filter_stops in [True, False]:
        with_without = "without" if filter_stops else "with"
        print(f"Extraction {with_without} stopwords:")
        print("- Extracting sentences")
        doc_sentences = cleaned_index.dest_path.map(
            doc_path_to_sentences, filter_stops=filter_stops
        )
        print("- Saving sentences")
        df = pd.DataFrame(
            {
                "document_id": cleaned_index.document_id,
                "sentences": doc_sentences,
            }
        )
        sent_filename = f"sentences_{with_without}_stopwords.pkl"
        df.to_pickle(os.path.join(out_path, sent_filename))
        print("- Extracting contents")
        doc_contents = doc_sentences.map(join_sentences)
        print("- Saving contents")
        df = pd.DataFrame(
            {
                "document_id": cleaned_index.document_id,
                "contents": doc_contents,
            }
        )
        contents_filename = f"contents_{with_without}_stopwords.csv"
        df.to_pickle(os.path.join(out_path, contents_filename))
    print("DONE")
