import os
from pathlib import Path

from embedding_explorer import show_dashboard
from embedding_explorer.model import Model
from gensim.models import KeyedVectors

IN_PATH = "dat/greek/models/word2vec"


def get_models(path: str) -> dict[str, Model]:
    """Get all models in a directory."""
    model_names = [entry.name for entry in os.scandir(path) if entry.is_dir()]
    models = {}
    for model_name in model_names[:5]:
        model_path = Path(path).joinpath(model_name, "model.gensim")
        keyed_vectors = KeyedVectors.load(str(model_path))
        models[model_name] = Model.from_keyed_vectors(keyed_vectors)
        print(model_name, models[model_name].vocab[:5])
    return models


def main():
    models = get_models(IN_PATH)
    show_dashboard(models, fuzzy_search=True)


if __name__ == "__main__":
    main()
