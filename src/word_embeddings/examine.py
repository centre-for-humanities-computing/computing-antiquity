import os
from pathlib import Path

from dash_extensions.enrich import dash
from embedding_explorer.app import get_dash_app
from embedding_explorer.blueprints.dashboard import create_dashboard
from embedding_explorer.model import Model
from embedding_explorer.prepare.gensim import prepare_keyed_vectors
from gensim.models import KeyedVectors

IN_PATH = "dat/greek/models/word2vec"


def get_models(path: str) -> dict[str, Model]:
    """Get all models in a directory."""
    model_names = [entry.name for entry in os.scandir(path) if entry.is_dir()]
    models = {}
    for model_name in model_names:
        model_path = Path(path).joinpath(model_name, "model.gensim")
        keyed_vectors = KeyedVectors.load(str(model_path))
        models[model_name] = prepare_keyed_vectors(keyed_vectors)
        print(model_name, models[model_name].vocab[:5])
    return models


models = get_models(IN_PATH)
blueprint, register_pages = create_dashboard(models)
app = get_dash_app(blueprint)
register_pages(app)


print("Running app")
print([value["path"] for value in dash.page_registry.values()])
if __name__ == "__main__":
    app.run_server(debug=True)
