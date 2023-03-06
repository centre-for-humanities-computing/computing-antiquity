import re
import unicodedata as ud
import string
from typing import List

import cltk
import cltk.alphabet.grc.grc as alphabet
from cltk.lemmatize.grc import GreekBackoffLemmatizer

from utils.text import remove_punctuation, only_dots

cltk.data.fetch.FetchCorpus(language="grc").import_corpus("grc_models_cltk")

GREEK_CHARACTERS = (
    "".join(
        alphabet.LOWER
        + alphabet.LOWER_ACUTE
        + alphabet.LOWER_BREVE
        + alphabet.LOWER_CIRCUMFLEX
        + alphabet.LOWER_CONSONANTS
        + alphabet.LOWER_DIAERESIS
        + alphabet.LOWER_DIAERESIS_ACUTE
        + alphabet.LOWER_DIAERESIS_CIRCUMFLEX
        + alphabet.LOWER_DIAERESIS_GRAVE
        + alphabet.LOWER_GRAVE
        + alphabet.LOWER_MACRON
        + [alphabet.LOWER_RHO]
        + alphabet.LOWER_ROUGH
        + [alphabet.LOWER_RHO_ROUGH]
        + [alphabet.LOWER_RHO_SMOOTH]
        + alphabet.LOWER_ROUGH_ACUTE
        + alphabet.LOWER_ROUGH_CIRCUMFLEX
        + alphabet.LOWER_ROUGH_GRAVE
        + alphabet.LOWER_SMOOTH
        + alphabet.LOWER_SMOOTH_ACUTE
        + alphabet.LOWER_SMOOTH_CIRCUMFLEX
        + alphabet.LOWER_SMOOTH_GRAVE
        + alphabet.UPPER
        + alphabet.UPPER_ACUTE
        + alphabet.UPPER_BREVE
        + alphabet.UPPER_CONSONANTS
        + alphabet.UPPER_DIAERESIS
        + alphabet.UPPER_GRAVE
        + alphabet.UPPER_MACRON
        + [alphabet.UPPER_RHO]
        + alphabet.UPPER_ROUGH
        + [alphabet.UPPER_RHO_ROUGH]
        + alphabet.UPPER_ROUGH_ACUTE
        + alphabet.UPPER_ROUGH_CIRCUMFLEX
        + alphabet.UPPER_ROUGH_GRAVE
        + alphabet.UPPER_SMOOTH
        + alphabet.UPPER_SMOOTH_ACUTE
        + alphabet.UPPER_SMOOTH_CIRCUMFLEX
        + alphabet.UPPER_SMOOTH_GRAVE
        + alphabet.NUMERAL_SIGNS
        + alphabet.ACCENTS
    )
    + string.punctuation
)


def remove_non_greek(text: str) -> str:
    """
    Removes non greek characters from the text, except for punctuation.

    Parameters
    ----------
    text: str
        Text to normalize

    Returns
    ----------
    text: str
    """
    non_greek = re.compile(f"[^{GREEK_CHARACTERS}]")
    text = re.sub(non_greek, " ", text)
    return text


def expand_subscript(text: str) -> str:
    """
    Expands greek subscripts to separate characters.

    Parameters
    ----------
    text: str
        Text to normalize

    Returns
    ----------
    text: str
    """
    text = "".join([alphabet.MAP_SUBSCRIPT_NO_SUB.get(c, c) for c in text])
    return text


def replace_tonos_oxia(text: str) -> str:
    """
    For the Ancient Greek language. Converts characters accented with the
    tonos (meant for Modern Greek) into the oxia equivalent. Without this
    normalization, string comparisons will fail.

    Parameters
    ----------
    text: str
        Text to normalize

    Returns
    ----------
    text: str
    """
    trans = str.maketrans(alphabet.TONOS_OXIA)
    return text.translate(trans)


def normalize(text: str, keep_sentences: bool) -> str:
    """
    Removes digits and punctuation from the text supplied.

    Parameters
    ----------
    text: str
        Text to normalize
    keep_sentences: bool
        Specifies whether the normalization should keep sentence borders or not
        (exclamation marks, dots, question marks)

    Returns
    ----------
    text: str
    """
    text = replace_tonos_oxia(text)
    text = expand_subscript(text)
    # Unicode normalization
    text = ud.normalize("NFKC", text)
    text = remove_non_greek(text)
    text = only_dots(text)
    text = remove_punctuation(text, keep_sentences)
    return text.lower()


def tokenize(text: str) -> List[str]:
    """
    Splits text to a list of tokens.

    Parameters
    ----------
    text: str
        Text to tokenize

    Returns
    ---------
    tokens: list of str
    """
    return text.split()


lemmatizer = GreekBackoffLemmatizer()


def lemmatize(tokens: List[str]) -> List[str]:
    """
    Lemmatizes a list of tokens.

    Parameters
    ----------
    tokens: list of str
        Tokens to lemmatize

    Returns
    ---------
    lemmata: list of str
    """
    if not tokens:
        return []
    _, lemmata = zip(*lemmatizer.lemmatize(tokens))
    return list(lemmata)


STOPWORDS = set(
    cltk.stops.words.Stops(iso_code="grc").get_stopwords()
    + [
        "δʼ",
        "ἔχω",
        "kai",
        "πᾶς",
        "εἰμί",
        "αυτός",
        "αὐτός",
        "δ",
        "σύ",
        "ἀλλ",
        "ἐγώ",
        "ἐπί",
    ]
)


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Removes all stopwords from a list of tokens.

    Parameters
    ----------
    tokens: list of str
        Tokens to filter

    Returns
    ---------
    tokens: list of str
    """
    return [token for token in tokens if token not in STOPWORDS]


def _tokenize_sentence(text: str, remove_stops: bool) -> List[str]:
    sentence = tokenize(text)
    sentence = lemmatize(sentence)
    if remove_stops:
        sentence = remove_stopwords(sentence)
    text = " ".join(sentence)
    text = normalize(text, keep_sentences=False)
    sentence = tokenize(text)
    return sentence


def sentencize(text: str, remove_stops: bool) -> List[List[str]]:
    """
    Cleans up the text, sentencizes, tokenizes, and lemmatizes it.

    Parameters
    ----------
    text: str
        Text to sentencize
    remove_stops: bool
        Indicates whether the function should remove stopwords from the text


    Returns
    ----------
    sentences: list of list of str
        List of sentences in the form of list of tokens.
    """
    text = normalize(text, keep_sentences=True)
    sentence_texts = text.split(".")
    sentences = [
        _tokenize_sentence(sentence, remove_stops)
        for sentence in sentence_texts
    ]
    sentences = [sentence for sentence in sentences if sentence]
    return sentences


def clean(text: str, remove_stops: bool) -> str:
    """
    Returns a clean text consisting only of lemmata separated by spaces

    Parameters
    ----------
    text: str
        Text to clean
    remove_stops: bool
        Indicates whether the function should remove stopwords from the text

    Returns
    ----------
    text: str
    """
    sentences = sentencize(text, remove_stops)
    # Concatenates all sentences into one
    tokens: List[str] = []
    for sentence in sentences:
        tokens.extend(sentence)
    # Tokens separated by space
    return " ".join(tokens)
