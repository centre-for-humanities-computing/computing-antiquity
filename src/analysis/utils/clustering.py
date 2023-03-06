"""
Module containing functions for clustering embeddings.
"""
import warnings
from typing import Iterable

import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objs as go
from deprecated import deprecated
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import OPTICS, KMeans, SpectralClustering
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


@deprecated(
    reason="Phased out in favor of utils.embeddings.reduce, definitely doesn't belong here"
)
def reduce_PCA(X: np.ndarray, dim: int = 10, verbose: bool = True) -> np.ndarray:
    """
    DEPRECATED: Use utils.embeddings.reduce instead

    Reduces data matrix X to a given number of features using PCA.
    If verbose is set to true it prints the total amount of variance
    explained after dimensionality reduction.
    """
    pca = PCA(dim).fit(X)
    if verbose:
        total_variance = np.sum(pca.explained_variance_ratio_)
        print(f"Total explained variance: {total_variance}")
    return pca.transform(X)


@deprecated(
    reason="Phased out in favor of utils.embeddings.reduce, definitely doesn't belong here"
)
def reduce_SVD(X, dim=10, verbose=True):
    """
    DEPRECATED: Use utils.embeddings.reduce instead

    Reduces data matrix X to a given number of features using SVD.
    If verbose is set to true it prints the total amount of variance
    explained after dimensionality reduction.
    """
    svd = TruncatedSVD(n_components=dim).fit(X)
    if verbose:
        total_variance = np.sum(svd.explained_variance_ratio_)
        print(f"Total explained variance: {total_variance}")
    return svd.transform(X)


@deprecated(
    reason="Phased out in favor of utils.embeddings.reduce, definitely doesn't belong here"
)
def reduce_NMF(X: np.ndarray, dim: int = 10) -> np.ndarray:
    """DEPRECATED: Use utils.embeddings.reduce instead"""
    return NMF(n_components=dim).fit_transform(X)


def sse(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Computes sum of squared errors for each cluster
    """
    ##Array of sse for each cluster
    error = np.full(centers.shape[0], 0)
    ##Calculate sse for every cluster
    for i, _ in enumerate(error):
        ##points that are in the current cluster
        cluster_points = X[labels == i]
        ##Center of current cluster
        cluster_center = centers[i]
        ##Calculating sse for cluster
        error[i] = np.sum(np.square(cluster_points - cluster_center), (0, 1))
    return error


def kmeans(X: np.ndarray, n: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs K-Means clustering on the given vectors.
    Returns the given labels of the datapoints and the cluster centers.
    """
    _kmeans = KMeans(n_clusters=n).fit(X)
    centers = _kmeans.cluster_centers_
    labels = _kmeans.labels_
    return labels, centers


def bisecting_kmeans(X: np.ndarray, n: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs bisecting K-Means clustering on the given vectors.
    Returns the given labels of the datapoints and the cluster centers.
    """
    # This is gonna be the array where we store the centroids of each cluster
    centers = np.full((n, X.shape[1]), 0)
    # First kmeans
    labels, _centers = kmeans(X, 2)
    centers[:2] = _centers
    current = X
    for i in range(n - 2):
        # Calculates Squared Standard Error for each cluster
        sse_ = sse(X, labels, centers)
        # And selects the one with the highest
        max_sse = np.argmax(sse_)
        # For organizational purposes this piece of code switches around the
        # selected cluster and the last cluster by index.
        # This is gonna come in handy when we have to add the new clusters
        # cause you just add them to the end of the arrays
        _labels = np.copy(labels)
        _labels[labels == max_sse] = i + 1
        _labels[labels == i + 1] = max_sse
        labels = _labels
        _centers = np.copy(centers)
        _centers[i + 1] = centers[max_sse]
        _centers[max_sse] = centers[i + 1]
        centers = _centers
        # Selects all vectors in current cluster
        current = X[labels == i + 1]
        # Calculates kmeans for the cluster with highest sse
        _labels, _centers = kmeans(current, 2)
        # We add to the labels, so that the new clusters are going to be the
        # ones with the two highest indices.
        _labels += i + 1
        # Add the new clustering to both centers and labels
        centers[i + 1 : i + 3] = _centers
        labels[labels == i + 1] = _labels
    return labels, centers


def optics(X: np.ndarray, min_samples: int = 5) -> np.ndarray:
    """
    Performs OPTICS clustering on the data, returns array of cluster labels.
    """
    clustering = OPTICS(min_samples=min_samples).fit(X)
    return clustering.labels_


def affinity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity matrix, and normalises output to [0,1]
    """
    return np.minimum(np.abs(cosine_similarity(X, X)), 1)


def spectral(X: np.ndarray, n: int = 4) -> np.ndarray:
    """
    Performs spectral clustering on the data matrix.
    Uses Cosine similarity as its affinity measure.
    """
    affinity = affinity_matrix(X)
    clustering = SpectralClustering(n_clusters=n, affinity="precomputed").fit(affinity)
    return clustering.labels_


def display_hierarchical(
    embeddings: np.ndarray,
    labels: Iterable[str],
    distance_metric: str = "euclidean",
    linkage_method: str = "single",
    **kwargs,
) -> go.Figure:
    """
    Performs hierarchical clustering on the embeddings in the dataframe
    then returns a plot displaying a ✨dendrogram✨
    of the cluster hierarchy.

    Parameters
    ----------
    embeddings: ndarray of shape (n_observations, n_features)
        Embeddings to cluster.
    labels: iterable of str of length n_observations
        Labels to display on the dendogram.
    distance_metric: str, default 'euclidean'
        Distance metric to be used by the clustering algorithm.
    linkage_method: str, default 'single'
        Linkage method to be used by the clustering algortihm.

    Returns
    ----------
    fig: go.Figure
        Plotly dendogram of the cluster hierarchy
    """
    # Plotly's create_dendogram expects us to give 'em functions,
    # so basically I'm creating a wrapper around scipy's tools
    # I initially used lambdas, but Pylint said it was ugly :(
    def distance_function(embeddings: np.ndarray) -> np.ndarray:
        return pdist(embeddings, metric=distance_metric)

    if "linkage" in kwargs:
        warnings.warn(
            "Argument 'linkage' deprecated in favor of 'linkage_method'.",
            DeprecationWarning,
        )
        linkage_method = kwargs["linkage"]

    def linkage_function(embeddings: np.ndarray) -> np.ndarray:
        return linkage(embeddings, method=linkage_method, metric=distance_metric)

    fig = ff.create_dendrogram(
        embeddings,
        labels=labels,
        distfun=distance_function,
        linkagefun=linkage_function,
        **kwargs,
    )
    return fig
