# TODO: DOCUMENT THIS
import cltk
from cltk.lemmatize.lat import LatinBackoffLemmatizer
cltk.data.fetch.FetchCorpus(language = "lat").import_corpus("lat_models_cltk")
from cltk.alphabet.lat import normalize_lat
from utils.text import remove_punctuation, remove_whitespace, remove_digits

lemmatizer = LatinBackoffLemmatizer()
stopwords = set(cltk.stops.words.Stops(iso_code="lat").get_stopwords())

def lemmatize(tokens):
    _, lemmata = zip(*lemmatizer.lemmatize(tokens))
    return lemmata

def remove_stopwords(tokens_list):
    return [
        remove_digits(token) for token in tokens_list if (token not in stopwords)
    ]

def normalize(s):
    s = remove_punctuation(s)
    s = remove_whitespace(s)
    s = remove_digits(s)
    s = normalize_lat(
        s,
        drop_accents = True,
        drop_macrons = True,
        jv_replacement = True,
        ligature_replacement = True
    )
    return s.lower()

#Since all punctuation gets removed, and all whitespace turned into spaces
#no need to mess with anything but spaces
def tokenize(s):
    return s.split()

def clean(text):
    text = normalize(text)
    tokens = tokenize(text)
    lemmata = lemmatize(tokens)
    lemmata = remove_stopwords(lemmata)
    return " ".join(lemmata)