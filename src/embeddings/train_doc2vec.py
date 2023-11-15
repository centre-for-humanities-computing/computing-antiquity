"""Trains GloVe embeddings using glovpy on Plutarch and Philon."""
from pathlib import Path

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import deaccent

DATA_PATH = "dat/greek/cleaned_data/lemmatized_without_stopwords.csv"
OUT_PATH = "dat/greek/models/doc2vec/"


def main():
    print("Loading data.")
    data = pd.read_csv(DATA_PATH)
    print("Preprocessing corpus.")
    data.text = data.text.fillna("")
    data.text = data.text.str.lower()
    data.text = data.text.map(deaccent)
    data.text = data.text.map(lambda s: s.split())
    print("Producing tagged corpus.")
    corpus = [
        TaggedDocument(text, [doc_id])
        for text, doc_id in zip(data.text, data.document_id)
    ]
    out_dir = Path(OUT_PATH)
    out_dir.mkdir(exist_ok=True, parents=True)
    print("Training model.")
    model = Doc2Vec(corpus)
    print("Saving model.")
    model.save(str(out_dir.joinpath("lemmatized_corpus.doc2vec.gensim")))
    print("Saving keyed vectors.")
    model.dv.save(str(out_dir.joinpath("lemmatized_corpus.vectors.gensim")))
    print("DONE")


if __name__ == "__main__":
    main()
