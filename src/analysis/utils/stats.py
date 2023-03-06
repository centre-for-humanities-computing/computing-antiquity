"""
Module containing useful utilities for statistical computation.
"""
from typing import Optional, Iterable

import pandas as pd
import numpy as np
from scipy.stats import zscore
from numpy.typing import ArrayLike
from pandas.api.types import is_numeric_dtype


def is_outlier(a: ArrayLike) -> np.ndarray:
    """
    Checks where a sequence contains outliers.

    Parameters
    ----------
    a: array_like
        Numerical sequence

    Returns
    -------
    ndarray
        A boolean array indicating where the original sequence
        contains outliers.
    """
    return np.abs(zscore(a)) > 3


def remove_outliers(
    df: pd.DataFrame, subset: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """
    Removes outlier rows from a data frame.

    Parameters
    ----------
    df: DataFrame
        Pandas data frame containing the data you want to filter
    subset: iterable of str or None, default None
        Indicates which columns you would like to remove outliers from.
        If not specified, all columns containing numerical data will
        be filtered.

    Returns
    -------
    DataFrame
    """
    if subset is None:
        # That's a mouthfull sheeesh
        # I'm basically extracting all column names, which are of a numeric dtype
        subset = df.dtypes[df.dtypes.map(is_numeric_dtype)].index.to_list()
    outliers = []
    for column in subset:
        # Collecting outliers in each column
        outliers.append(is_outlier(df[column]))
    # Clever way of calculating if any of the columns contain
    # an outlier at any given index
    outliers = np.stack(outliers).any(axis=0)
    # Excluding outliers
    return df[~outliers]
