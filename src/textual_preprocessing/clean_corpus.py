"""Utilities for cleaning the corpus."""
import argparse
import os
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Set

import pandas as pd
import spacy
import tqdm
from spacy.language import Language
from spacy.tokens import Doc, DocBin, Token

INDEX_PATH = "dat/greek/processed_data/index.csv"
IN_DIR = "dat/greek/processed_data"
OUT_DIR = "dat/greek/cleaned_data"

UPOSTag = Literal[
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]


def is_latin(char: str) -> bool:
    """Checks if a character is latin."""
    code = ord(char)
    return ((code >= ord("a")) and (code <= ord("z"))) or (
        (code >= ord("A")) and (code <= ord("Z"))
    )


def contains_latin(text: str) -> bool:
    """Checks if text contains latin characters."""
    return any(map(is_latin, text))


def is_acceptable(
    token: Token, remove_stopwords: bool, upos_tags: Set[UPOSTag]
) -> bool:
    """Checks if a token can be accepted for further processing"""
    return (
        # We don't keep punctuation
        not token.is_punct
        # All characters gotta be alphabetical in the token
        and token.is_alpha
        # We don't take no digits
        and not token.is_digit
        # Nor whitespace tokens
        and not token.is_space
        # Only keep stopwords if remove_stopwords is false
        and not (remove_stopwords and token.is_stop)
        # We don't accept tokens that contain Latin characters
        and not (contains_latin(token.norm_))
        # If UPOS tags is empty, then it's acceptable,
        # otherwise it only is, when the token's POS tag is in the provided set
        and (not upos_tags or (token.pos_ in upos_tags))
    )


def extract_normalized_sentences(
    doc: Doc,
    remove_stopwords: bool,
    lemmatize: bool,
    upos_tags: Set[UPOSTag],
) -> List[List[str]]:
    """Extracts normalized sentences from the document in the form of
    list of lists of tokens.
    """
    result_sentences = []
    for sent in doc.sents:
        # We build up a sentence from its tokens
        result_sentence = []
        for token in sent:
            # We check if the token is acceptable according to the rules
            # we outlined
            if is_acceptable(
                token, remove_stopwords=remove_stopwords, upos_tags=upos_tags
            ):
                if lemmatize:
                    result_sentence.append(token.lemma_)
                else:
                    result_sentence.append(token.norm_)
        # If the sentence is not empty we add the token
        if result_sentence:
            result_sentences.append(result_sentence)
    return result_sentences


def join_sentences(sentences: List[List[str]]) -> str:
    """Joins list of sentences to one document,
    where each sentence is on a new line and tokens are space-separated."""
    return "\n".join([" ".join(sentence) for sentence in sentences])


def load_doc(path: str, nlp: Language) -> Doc:
    """Loads document from disk."""
    doc_bin = DocBin().from_disk(path=path)
    (doc,) = list(doc_bin.get_docs(nlp.vocab))
    return doc


def pretty_print_cleaning(doc: Doc) -> None:
    """Prints the document, and colors tokens that are getting deleted.
    Procides explanation for why each token gets deleted.
    """
    # I import colorama here so that it's not a module leve dependency
    import colorama
    from colorama import Back, Fore

    reset_color = colorama.Style.RESET_ALL
    # Initialising colorama
    colorama.init()

    for sent in doc.sents:
        for token in sent:
            token_text = token.orth_ + token.whitespace_
            if token.is_digit:
                print(Fore.BLUE + token_text + reset_color, end="")
            elif token.is_punct:
                print(Fore.CYAN + token_text + reset_color, end="")
            elif not token.is_alpha:
                print(Fore.RED + token_text + reset_color, end="")
            elif token.is_stop:
                print(Fore.GREEN + token_text + reset_color, end="")
            elif contains_latin(token.orth_):
                print(Fore.YELLOW + token_text + reset_color, end="")
            else:
                print(token_text, end="")
        print()

    print("\n")
    print("|---COLOR GUIDE---|")
    print("|   " + Back.BLUE + " " + reset_color + " - Digit     |")
    print("|-----------------|")
    print("|   " + Back.CYAN + " " + reset_color + " - Punct     |")
    print("|-----------------|")
    print("|   " + Back.RED + " " + reset_color + " - Not Alpha |")
    print("|-----------------|")
    print("|   " + Back.GREEN + " " + reset_color + " - Stop      |")
    print("|-----------------|")
    print("|   " + Back.YELLOW + " " + reset_color + " - Latin     |")
    print("|-----------------|")


def clean_corpus(
    ids: Iterable[str],
    paths: Iterable[str],
    lemmatize: bool,
    remove_stopwords: bool,
    nlp: Language,
    upos_tags: Optional[Set[UPOSTag]] = None,
) -> pd.DataFrame:
    """Cleans corpus and turns it into a dataframe where document IDs
    are linked to the cleaned text."""
    texts = []
    doc_ids = []
    # Adding progress bar
    paths = tqdm.tqdm(paths)
    for document_id, path in zip(ids, paths):
        try:
            doc = load_doc(path, nlp=nlp)
            sentences = extract_normalized_sentences(
                doc,
                remove_stopwords=remove_stopwords,
                lemmatize=lemmatize,
                upos_tags=upos_tags or set(),
            )
            text = join_sentences(sentences)
            texts.append(text)
            doc_ids.append(document_id)
        except FileNotFoundError:
            print(f"    - Document not found: {document_id}")
    return pd.DataFrame({"document_id": doc_ids, "text": texts})


def create_parser() -> argparse.ArgumentParser:
    """Creates parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="Corpus cleaner",
        description="Produces cleaned corpus based on processed docs.",
    )
    parser.add_argument("--model", type=str)
    parser.add_argument("--dest", type=str)
    parser.add_argument("--src_index", type=str)
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    print("\n--------------------------")
    print("----Cleaning Procedure----")
    print("--------------------------\n")

    print("Loading spaCy model")
    nlp = spacy.load(args.model)

    print("Loading index")
    index = pd.read_csv(args.src_index, index_col=0)

    print("Streaming documents")
    ids = index.document_id
    paths = index.dest_path

    print("Creating directory")
    Path(args.dest).mkdir(parents=True, exist_ok=True)

    print("Cleaning documents")
    print(" 1. Normalized with stopwords.")
    corpus = clean_corpus(
        ids, paths, lemmatize=False, remove_stopwords=False, nlp=nlp
    )
    corpus.to_csv(os.path.join(args.dest, "with_stopwords.csv"))
    print(" 2. Normalized without stopwords.")
    corpus = clean_corpus(
        ids, paths, lemmatize=False, remove_stopwords=True, nlp=nlp
    )
    corpus.to_csv(os.path.join(args.dest, "without_stopwords.csv"))
    print(" 3. Lemmatized with stopwords.")
    corpus = clean_corpus(
        ids, paths, lemmatize=True, remove_stopwords=False, nlp=nlp
    )
    corpus.to_csv(os.path.join(args.dest, "lemmatized_with_stopwords.csv"))
    print(" 4. Lemmatized without stopwords.")
    corpus = clean_corpus(
        ids, paths, lemmatize=True, remove_stopwords=True, nlp=nlp
    )
    corpus.to_csv(os.path.join(args.dest, "lemmatized_without_stopwords.csv"))
    print(" 5. Lemmatized without stopwords, nouns only.")
    corpus = clean_corpus(
        ids,
        paths,
        lemmatize=True,
        remove_stopwords=True,
        nlp=nlp,
        upos_tags={"NOUN"},
    )
    corpus.to_csv(
        os.path.join(args.dest, "nouns_lemmatized_without_stopwords.csv")
    )
    print(" 6. Lemmatized without stopwords, verbs only.")
    corpus = clean_corpus(
        ids,
        paths,
        lemmatize=True,
        remove_stopwords=True,
        nlp=nlp,
        upos_tags={"VERB"},
    )
    corpus.to_csv(
        os.path.join(args.dest, "verbs_lemmatized_without_stopwords.csv")
    )

    print("DONE")


if __name__ == "__main__":
    main()
