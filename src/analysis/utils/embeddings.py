"""
Module containing useful functions for handling embeddings for analysis.
"""
import os
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

DATA_PATH = "/work/data_wrangling/dat/greek"


def to_list(embeddings: np.ndarray) -> List[np.ndarray]:
    """
    Turns a matrix into a list of vectors.
    This way they can be added to a data frame.

    Parameters
    ----------
    embeddings: np.ndarray of shape (n_samples, n_features)
        Embedding matrix.

    Returns
    -------
    list of length n_samples, of np.ndarray of shape (n_features,)
    """
    return [embeddings[i, :] for i in range(embeddings.shape[0])]


def to_array(vectors: pd.Series) -> np.ndarray:
    """
    Turns Series of vectors to numpy matrix.

    Parameters
    ----------
    vectors: Series of length n_samples, of np.ndarray of shape (n_features,)
        Series of embeddings

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    return np.stack(vectors.tolist())


def centroid(vectors: pd.Series) -> np.ndarray:
    """
    Calculates the centroid of a series of vectors.

    Parameters
    ----------
    vectors: Series of length n_samples, of np.ndarray of shape (n_features,)
        Series of embeddings

    Returns
    -------
    np.ndarray of shape (n_features,)
    """
    return to_array(vectors).mean(axis=0)


def load_embeddings(embedding_name: str) -> np.ndarray:
    """
    Loads precomputed embeddings from disk.

    Parameters
    ------------
    embedding_names: str
        Name of embedding to be loaded.

    Returns
    -------
    ndarray of shape (n_corpus, n_features)

    Raises
    ------
    ValueError:
        If the given embedding type is not found, ValueError will be raised.
    """
    try:
        embedding_path = os.path.join(
            DATA_PATH, f"embeddings/{embedding_name}.npy"
        )
        return np.load(embedding_path)
    except Exception as exc:
        raise ValueError(
            f"Given embedding type, '{embedding_name}' could not be loaded."
        ) from exc


def combine(*embedding_list: np.ndarray) -> np.ndarray:
    """
    Combines an arbitrarily large set of embeddings.

    Parameters
    ----------
    embedding_0: ndarray of shape (n_corpus, n_features_0)
    ...
    embedding_k: ndarray of shape (n_corpus, n_features_k)
        Set of embeddings to combine

    Returns
    -------
    ndarray of shape (n_corpus, sum(n_features_0, ..., n_features_k))
        Combined embeddings
    """
    return np.concatenate(embedding_list, axis=1)


def reduce(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """
    Performs dimensionality reduction on the embeddings with PCA.

    Parameters
    ----------
    embeddings: ndarray of shape (n_corpus, n_features)
        Embedding matrix to reduce.
    n_features: int
        Number of features we would like to have.

    Returns
    -------
    ndarray of shape (n_corpus, n_components)
    """
    return PCA(n_components).fit_transform(embeddings)


def add_embeddings(data: pd.DataFrame, **mapping: np.ndarray) -> pd.DataFrame:
    """
    Adds embeddings to a DataFrame into a new column.

    Parameters
    ----------
    data: DataFrame
        Pandas data frame to add embeddings to.
        Must have an id_nummer(int) column.
    **mapping: keyword arguments of np.ndarray of shape (n_corpus, n_features)
        A mapping of embedding names to embedding matrices.
    Returns
    -------
    DataFrame
        New data frame containing the emebddings
    """
    embedding_list = list(mapping.values())
    combined = combine(*embedding_list)
    is_undefined = np.isnan(combined).any(axis=1)
    # Indexing each embedding with the id column
    # Turning embeddings to lists, so they can be added
    # to the DataFrame
    mapping = {
        embedding_name: to_list(embedding[data.id_nummer])
        for embedding_name, embedding in mapping.items()
    }
    # Adding all columns
    data = data.assign(is_undefined=is_undefined[data.id_nummer], **mapping)
    return data[~data.is_undefined].drop(columns="is_undefined")


def display_3d(
    data: pd.DataFrame, embedding_column: str, **kwargs
) -> go.Figure:
    """
    Displays the embeddings in 3D.

    Parameters
    ----------
    data: DataFrame of length n_samples
        Pandas data frame
    **kwargs
        Additional arguments passed to Plotly
        (potentionally color 💚💛💜 or node size or whatever)

    Returns
    -------
    Figure
        Plotly 3D scatterplot.
    """
    embeddings = to_array(data[embedding_column])
    pos = reduce(embeddings, 3).T
    return px.scatter_3d(
        data_frame=data, x=pos[0], y=pos[1], z=pos[2], **kwargs
    )
