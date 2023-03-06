"""Utilities for dealing with fetching and manipulating metadata,
replaces metadata_tools.py
"""
from typing import Iterable, List, Tuple, Union

import msgspec
import numpy as np
import pandas as pd

DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    + "181pbNCULuYKO5yPrWIfdmOLkrps2WUpfoJ8mtT56vuw/edit#gid=1762774185"
)


def fetch_metadata(sheet_url: str = DEFAULT_SHEET_URL) -> pd.DataFrame:
    """Fetches metadata in for of a Pandas Dataframe
    from the supplied Google Sheets URL.
    """
    sheet_url = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
    metadata = pd.read_csv(sheet_url)
    return metadata


class Query(msgspec.Struct):
    """One selection query for selecting a group of works in the metadata.

    Attributes
    ----------
    by: str
        Column by which the values should be selected.
    values: list of str or list of int
        The values one wants to select
    """

    by: str
    values: List[Union[int, str]]


class Group(msgspec.Struct):
    """Describes a grouping of values in a data frame.

    Attributes
    ----------
    name: str
        Name of the group
    queries: list of Query
        List of queries to select the works in the group.
        If one of the quieries selects the work it
        is considered to be part of the group.
    """

    name: str
    queries: List[Query]


_group_decoder = msgspec.json.Decoder(List[Group])


def parse_groups(text: bytes) -> List[Group]:
    """Parses the given json text to a list of groups with msgspec.
    Ahhh I love this library, like it's fast and it has schemas,
    that are validated, like can I marry a library please.

    Parameters
    ----------
    text: bytes
        Byte representation of the json data to be parsed.

    Returns
    -------
    list of Group
        All the groups parsed from the text.
    """
    return _group_decoder.decode(text)


def execute_query(data: pd.DataFrame, query: Query) -> pd.Series:
    """Executes a selection query on the given dataframe.

    Parameters
    ----------
    data: DataFrame
        Data to select from.
    query: Query
        Query to execute.

    Returns
    -------
    Series of boolean
        Series of values with which rows of the
        dataframe can be selected.
        Contains True for the rows that the selection should include.

    Notes
    -----
    If the query does not contain any selection values,
    all rows will be selected where 'by' is not undefined.
    """
    if not query.values:
        return data[query.by].notna()
    return data[query.by].isin(query.values)


def agg_queries(data: pd.DataFrame, queries: List[Query]) -> pd.Series:
    """
    Aggregates all queries on the dataframe and gives a joint result.

    Parameters
    ----------
    data: DataFrame
        Data to select from.
    queries: list of Query
        List of queries to aggregate.

    Returns
    -------
    Series of boolean
        Series of values with which rows of the
        dataframe can be selected.
    """
    # Collecting the result of each query into a list of numpy arrays
    results = [execute_query(data, query).to_numpy() for query in queries]
    # Hehe dirty numpy tricks again :P
    # Stacking up all results into a matrix, and then
    # checking which rows contain True values
    # Thereby if any query yielded True for a given row,
    # it will be selected.
    return np.stack(results).any(axis=0)


def group_data(
    data: pd.DataFrame,
    groups: Iterable[Group],
    drop: bool = True,
    exclusive: bool = False,
) -> pd.DataFrame:
    """Selects certain subsets of rows in the dataframe based on
    the supplied groups and adds a 'group' column, where these
    values are grouped.

    Parameters
    ----------
    data: DataFrame
        Data to group.
    groups: iterable of Group
        Groups should contain a group name,
        based on which column they select rows,
        and which values in that column should be selected.
    drop: bool, default True
        Specifies whether the function should drop data that
        does not belong to any group.
    exclusive: bool, default False
        Specifies whether groups should be mutually exclusive.

    Returns
    -------
    DataFrame
        Data with the given groups selected and
        'group' column added.
    """
    # Creating empty series with the same length as the DataFrame
    # YOU NEED TO COPY; OTHERWISE THE ORIGINAL OBJECTS MEMORY GETS OVERWRITTEN
    group_column = pd.Series(data.tlg_genre, copy=True)
    group_column[:] = None
    for group in groups:
        is_empty = group_column.isna()
        is_in_group = agg_queries(data, group.queries)
        # Assign group label, where no group is assigned yet
        # and the row belongs to the group
        group_column = group_column.mask(is_empty & is_in_group, group.name)
        # If the group column isn't empty, add the other group comma separated
        if not exclusive:
            group_column = group_column.mask(
                ~is_empty & is_in_group, lambda s: s + "," + group.name
            )
    data = data.assign(group=group_column)
    if drop:
        data = data.dropna(subset="group")
    if not exclusive:
        # Split the group column into multiple different rows,
        # so that entries that appear in multiple will be
        # included in all groups
        data = data.assign(group=data.group.str.split(","))
        data = data.explode("group")
    return data


def stream_groups(
    data: pd.DataFrame, groups: Iterable[Group]
) -> Iterable[Tuple[str, pd.DataFrame]]:
    """Streams all groups from the data based on a set of groups.

    Parameters
    ----------
    data: DataFrame
        Data to group.
    groups: iterable of Group
        Groups to select.

    Yields
    ------
    group_name: str
        Name of the current group.
    group_data: DataFrame
        Data belonging to that group.

    Notes
    -----
    Unlike group_data(), stream_groups() interprets groups as being
    mutually inclusive. If you want mutually exclusive groups, you
    might want to use group_data()
    """
    for group in groups:
        is_in_group = agg_queries(data, group.queries)
        yield group.name, data[is_in_group]
