"""
Module containing useful utilities for evaluating machine learning models
on the corpus.
"""
from functools import partial
from typing import Callable, Iterable, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import (KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold)

import utils.embeddings as em

CVStream = Iterable[Tuple[pd.DataFrame, pd.DataFrame]]
CrossValidator = Callable[[pd.DataFrame], CVStream]


def _from_sklearn(sk_cross_validator, data: pd.DataFrame) -> CVStream:
    ids = data.id_nummer.unique()
    for train_index, test_index in sk_cross_validator.split(ids):
        train_ids = ids[train_index]
        test_ids = ids[test_index]
        train_data = data[data.id_nummer.isin(train_ids)]
        test_data = data[data.id_nummer.isin(test_ids)]
        yield train_data, test_data


def _k_fold(data: pd.DataFrame, n_folds: int) -> CVStream:
    ids = data.id_nummer.unique()
    folds = KFold(n_splits=n_folds)
    return _from_sklearn(folds, data)


def k_fold(n_folds: int = 10) -> CrossValidator:
    """
    Returns k-fold cross validator with the specified number of folds.
    """
    return partial(_k_fold, n_folds=n_folds)


def leave_one_out(data: pd.DataFrame) -> CVStream:
    """
    Leave one out cross validator.
    """
    ids = data.id_nummer.unique()
    n_folds = len(ids)
    return k_fold(n_folds)(data)


def _rep_kfold(data: pd.DataFrame, n_folds: int, n_reps: int) -> CVStream:
    return _from_sklearn(RepeatedKFold(n_splits=n_folds, n_repeats=n_reps), data)


def repeated_k_fold(n_folds: int = 10, n_repetitions: int = 10) -> CrossValidator:
    """
    Returns repeated k-fold cross validator with the specified number of
    folds and repetitions.
    """
    return partial(_rep_kfold, n_folds=n_folds, n_reps=n_repetitions)


def _rep_strat_kfold(data: pd.DataFrame, n_folds: int, n_reps: int) -> CVStream:
    return _from_sklearn(
        RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_reps), data
    )


def repeated_stratified_k_fold(
    n_folds: int = 10, n_repetitions: int = 10
) -> CrossValidator:
    """
    Returns repeated stratified k-fold cross validator with the specified number of
    folds and repetitions.
    """
    return partial(_rep_strat_kfold, n_folds=n_folds, n_reps=n_repetitions)


def evaluate(
    model_class,
    data: pd.DataFrame,
    embeddings: Union[np.ndarray, str],
    labels: str,
    cross_validator: CrossValidator = k_fold(10),
    oversampling: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Evaluates the given classifier on multiple metrics with k-fold cross validation.

    Parameters
    ----------
    model_class: Model
        Any machine learning model class using the scikit-learn API.
    data: DataFrame of length n_samples
        Pandas data frame containing metadata.
    embeddings: np.ndarray of shape (n_corpus, n_features) or str
        Embedding matrix or name of the column in data, where the embeddings are.
    labels: str
        Name of the column where the labels are.
    cross_validator: CrossValidator, default k_fold(10)
        Cross validation strategy for the model
    oversampling: bool, default True
        Flag indicating, whether the evaluation should oversample
        training sets to balance for unbalanced classes.

    Returns
    ----------
    report: pd.DataFrame
        DataFrame containing classification metrics, such as
        precision, recall and f1-score for all classes,
        as well as the same results for a DummyClassifier
        to provide a baseline for comparison.
    """
    reports = []
    if oversampling:
        oversampler = RandomOverSampler(sampling_strategy="minority")
    for train_data, test_data in cross_validator(data):
        if isinstance(embeddings, str):
            X_train = em.to_array(train_data[embeddings])
            X_test = em.to_array(test_data[embeddings])
        else:
            X_train = embeddings[train_data.id_nummer]
            X_test = embeddings[test_data.id_nummer]
        y_train = train_data[labels]
        y_test = test_data[labels]
        if oversampling:
            X_train, y_train = oversampler.fit_resample(X_train, y_train)
        dummy_model = DummyClassifier().fit(X_train, y_train)
        model = model_class(**kwargs).fit(X_train, y_train)
        report = classification_report(
            y_test, model.predict(X_test), output_dict=True, zero_division=0
        )
        dummy_report = classification_report(
            y_test, dummy_model.predict(X_test), output_dict=True, zero_division=0
        )
        reports.extend(
            [
                {"model": "real", "label": label, **d}
                for label, d in report.items()
                if isinstance(d, dict)
            ]
        )
        reports.extend(
            [
                {"model": "dummy", "label": label, **d}
                for label, d in dummy_report.items()
                if isinstance(d, dict)
            ]
        )
    report = pd.DataFrame.from_records(reports)
    return report
