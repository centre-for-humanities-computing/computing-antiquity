"""
Script to extract top words from the corpus based on grouping by different attributes.
"""
import json

# import os
import subprocess
import re
from typing import Callable, List, Optional, Tuple

import datapane as dp
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from deprecated import deprecated
from utils.metadata import fetch_metadata, parse_groups, stream_groups

MARK_COUNTS_PATH = (
    "/work/data_wrangling/dat/greek/results/mark_word_counts.csv"
)


def lemma_counts(data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes lemma counts, ranks and relative frequencies based on CoNLL data.
    Groups values based on upos.

    Parameters
    ----------
    data: DataFrame
        Original data containing a upos and lemma column.

    Returns
    -------
    DataFrame
        Table containing lemma counts, frequencies and ranks.
        Indexed by upos and lemma.
    """
    data = data[["upos", "lemma"]]
    counts = (
        data.groupby("upos")
        .value_counts("lemma")
        .reset_index("lemma", name="lemma_count")
    )
    overall = counts.groupby("upos").sum()
    overall = overall.lemma_count.rename("total_count")
    counts = counts.join(overall, how="left")
    counts = counts.assign(lemma_freq=counts.lemma_count / counts.total_count)
    counts = counts.assign(
        lemma_rank=counts.lemma_count.groupby("upos")
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return counts.set_index("lemma", append=True)


def top_lemmata(data: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """
    Selects top N lemma for each upos tag.
    Formats lemmata to include count and percentage information.

    Parameters
    ----------
    data: DataFrame
        Table containing lemma count and frequency data.
    top_n: int
        Amount of top elements to keep.

    Returns
    -------
    DataFrame
        Formatted table with top N elements.
        Lemma and upos fields are kept.
    """
    data = data.reset_index()
    # Removing the lemma column from the index, if it's in there
    # if "lemma" in data.index.names:
    #     data = data.reset_index("lemma")
    # Selecting top n elements for each upos tag
    top = data.groupby("upos").apply(
        lambda group: group.nlargest(top_n, "lemma_count")
    )
    # Turning count into a string for concatenation
    count = top.lemma_count.apply(str)
    # Formatting frequencies as percentages
    percentage = (top.lemma_freq * 100).round(2).apply(str) + "%"
    top = top.assign(
        lemma=top.lemma + "( " + count + " | " + percentage + " )"
    )
    # top = top.reset_index()
    return top[["lemma", "upos"]]


@deprecated("Deprecated in favor of top_lemmata()")
def top_words(
    data: pd.DataFrame,
    by: str,
    values: Optional[List] = None,
    aggregate_name: Optional[str] = None,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Selects top lemmas for each upos.
    Lemmas are grouped by a specific column before being counted up.

    Parameters
    ----------
    data: DataFrame
        Data frame containing the dependency analysis of the text.
        Each row represents a word in the corpus.
        If you intend to group by metadata information it should also be merged
        with the dependency data.
    by: str
        Column to group/select values by other than upos.
    values: list or None, default None
        Specific groups we're interested in in the 'by' column.
        If not specified, no groups are selected.
    aggregate_name: str or None, default None
        If you would like all selected values to be aggregated,
        instead of grouped, you should specify a name for the group
        to be created.
    top_n: int, default 25
        Specifies, how many top words you would like to get.

    Returns
    -------
    DataFrame
        Table containing the top lemmata for all groups and all upos tags.
    """
    # If values is not specified, no groups are selected
    if values is not None:
        data = data[data[by].isin(values)]
    if aggregate_name is not None:
        grouping = ["upos"]
    else:
        grouping = [by, "upos"]
    top = (
        data.groupby(grouping)
        # Pandas magic: counting up all lemmas for each group
        .apply(
            lambda group: (
                group.value_counts("lemma")
                # This is a redundant line, because sorting is an implicit behaviour of
                # groupby, I'm including it for readabilites sake,
                # and also as groupby might change default behaviour in the future
                .sort_values(ascending=False).head(top_n)
            )
        )
        # Adding a lemma_count column by resetting the index of the resulting series
        .reset_index(name="lemma_count")
    )
    if aggregate_name is not None:
        group_labels = aggregate_name
    else:
        group_labels = top[by]
    # As Jacob asked for this, I'm adding the number of occurances
    # to the end of each lemma in parentheses.
    # e.g. "θεός(272)"
    top = top.assign(
        lemma=top.lemma + "(" + top.lemma_count.apply(str) + ")",
        group=group_labels,
    )
    return top[["lemma", "group", "upos"]]


def tabulate(data: pd.DataFrame) -> pd.DataFrame:
    """
    Turns the data frame returned by select_upos into wide format,
    where each group is a column.
    """
    mapping = (
        data
        # This is a rather hacky way of turning each group into a column
        # I couldn't find a better way to do this than to turn the data into a
        # Dictionary of groups and lemmata
        # df.pivot() might work but I couldn't figure out a smart way to use it
        # and would probably be just as hacky.
        .groupby("group")
        .apply(lambda g: list(g.lemma))
        .to_dict()
    )
    # Since not all texts are guaranteed to have enough unique words,
    # we have to be smart not to raise an exception.
    # If we originally orient the DataFrame differently and then
    # transpose it, empty spots will be filled with nan instead of
    # raising an exception.
    return pd.DataFrame.from_dict(mapping, orient="index").T


Color = Tuple[float, float, float, float]


def color_to_str(color: Color) -> str:
    """
    Turns color tuples into a CSS RGBA value indicating
    that an element's color should be that specific color.

    Parameters
    ----------
    color: tuple of (r: float, g: float, b : float, a: float)
        Color tuple, where each value is in range (0,1).

    Returns
    -------
    str
        CSS RGB value.
    """
    # I'm using numpy cause I'm lazy to write three multiplications by hand
    color = np.array(color)
    # Multiplying each value so they are properly scaled.
    color = color * 255
    r, g, b, _ = color
    # Setting string formatting not to show decimals
    return f"rgb({r:.0f}, {g:.0f}, {b:.0f})"


# These are properties for styling an HTML table with pandas,
# so that they look alright when pasted into Google Docs.
# Unfortunately Google Docs doesn't handle all HTML tables very nicely :((
table_style = {"selector": "table", "props": "border-collapse: collapse;"}
th_style = {
    "selector": "th",
    "props": """
        border-width: 1px;
        border-color: black;
        border-style: solid;
    """,
}
td_style = {
    "selector": "td",
    "props": """
        text-align: center;
        vertical-align: center;
        max-width:100%;
        white-space:nowrap;
    """,
}

WHITE: Color = (1.0, 1.0, 1.0, 1.0)


def continuous_color_mapping(
    mark_counts: pd.DataFrame,
) -> Callable[[str, str], Color]:
    """
    Creates a continuos color mapping for words based on Mark's gospel.
    The words that are more frequent in Mark, will be associated
    with red, while the least common shall be blue.

    Parameters
    ----------
    mark_counts: DataFrame
        Dataframe containing information about word counts
        in Mark's gospel.

    Returns
    -------
    Callable of str, str to Color
        Function mapping each lemma-upos pair in the corpus to a color
        based on its frequency in Mark's gospel.
    """
    # Loading a continuos color palette from Seaborn
    palette = sns.color_palette("Spectral_r", as_cmap=True)
    # coldness = mark_counts.lemma_rank
    heat = mark_counts.lemma_count
    heat = np.log(heat)
    # We have to do normalization in groups
    # Otherwise everything will be normalized over all words
    heat = heat.groupby("upos").apply(lambda group: group / group.max())

    def color_mapping(lemma: str, upos: str) -> Color:
        """Inner function mapping lemma-upos pairs to colors"""
        try:
            intensity = heat.loc[upos, lemma]
        except KeyError:
            intensity = 0
        return palette(intensity)

    return color_mapping


def add_alpha(
    rgb_color: Tuple[float, float, float], alpha: float = 1.0
) -> Color:
    """
    Adds alpha to RGB color.
    """
    return (*rgb_color, alpha)


def discrete_color_mapping(
    mark_ranks: pd.DataFrame,
) -> Callable[[str, str], Color]:
    """
    Creates a discrete color mapping based on word ranks in Mark's gospel.

    Parameters
    ----------
    mark_counts: DataFrame
        Dataframe containing information about word ranks
        in Mark's gospel.

    Returns
    -------
    Callable of str, str to Color
        Function mapping each lemma-upos pair in the corpus to a color
        based on its rank in Mark's gospel.
    """
    n_colors = 7
    # We need five categories for the top five words
    # we go one over, so those words don't get blue
    palette = sns.color_palette("Spectral", n_colors)
    ranks = mark_ranks.lemma_rank

    def color_mapping(lemma: str, upos: str) -> Color:
        """Inner function mapping lemma-upos pairs to colors"""
        try:
            rank = ranks.loc[upos, lemma]
            if rank <= 25:
                # We -1 cause ranks start from one,
                # and we wanna have 5 groups in the top five.
                coldness = (rank - 1) // 5
            else:
                coldness = n_colors - 1
            return add_alpha(palette[coldness])
        except KeyError:
            return WHITE

    return color_mapping


def color_words(
    data: pd.DataFrame, upos: str, color_mapping: Callable[[str, str], Color]
) -> pd.DataFrame:
    """
    Colors words in a dataframe according to some color mapping.

    Parameters
    ----------
    data: DataFrame
        Table created by tabulate()
    color_mapping: Callable of str, str to Color
        Function mapping lemma-upos pairs to colors.
    upos: str
        Current upos being processed.

    Returns
    -------
    DataFrame
        Properly styled data frame.
    """
    # Function to map the lemmata with added counts to the CSS properties
    def to_color(lemma: str) -> str:
        # If nan gets passed, we return black
        if pd.isna(lemma):
            return "background-color:black"
        # Removing parentheses
        lemma = re.sub(r"\(([^\)]+)\)", "", lemma)
        color = color_mapping(lemma, upos)
        return f"background-color:{color_to_str(color)};"

    styled_df = (
        data.style
        # Setting base style properties
        .set_table_styles([table_style, th_style, td_style])
        # Applying the mapped CSS properties to each cell of the table
        .applymap(to_color)
    )
    return styled_df


CONLL_PATH = "/work/data_wrangling/dat/greek/dataset/conll.feather"


def main(top_n: int, save_path: str, upos_tags: List[str]) -> None:
    """
    Main function of the script.
    Finds all the top words for works and authors,
    formats them into tables and saves them to disk.

    Parameters
    ----------
    top_n: int
        Top N words that should be extracted.
    save_path: str
        Path to save the resulting tables.
    upos_tags: list of str
        List of upos tags we're interested in.
    """
    print("Logging into Datapane")
    subprocess.call(["bash", "datapane_login.sh"])
    print("Loading data")
    # Loading table containing dependency relations
    # Only loading necessary columns to avoid overusage of memory
    dep_df = pd.read_feather(CONLL_PATH)[["id_nummer", "upos", "lemma"]]
    metadata = fetch_metadata()
    metadata = metadata[~metadata.skal_fjernes]
    # Only loading necessary columns,
    # otherwise we run out of memory because of denormalization
    metadata = metadata[["id_nummer", "forfatter", "værk"]]
    # Adding metadata
    data = dep_df.merge(metadata, on="id_nummer", how="left")
    with open("top_words_groups.json", "rb") as f_groups:
        groups = parse_groups(f_groups.read())
    with open("top_words_ordering.json", "r") as f_ordering:
        ordering = json.loads(f_ordering.read())
    try:
        mark_counts = pd.read_csv(
            MARK_COUNTS_PATH, index_col=["upos", "lemma"]
        )
    except FileNotFoundError:
        print("Mark's lemma counts not found, computing now...")
        mark_counts = lemma_counts(dep_df[dep_df.id_nummer == 783])
        mark_counts.to_csv(MARK_COUNTS_PATH)

    top_25_mark = mark_counts[mark_counts.lemma_rank < 25]
    print("Computing top words for all groups...")
    group_stream = tqdm(stream_groups(data, groups))
    groups_top_n = []
    groups_mark_rank = []
    for group_name, group_data in group_stream:
        counts = lemma_counts(group_data)
        top = top_lemmata(counts, top_n=top_n).assign(group=group_name)
        groups_top_n.append(top)
        mark_rank = counts[["lemma_rank"]].join(
            top_25_mark[["lemma_rank"]], rsuffix="_mark", how="right"
        )
        mark_rank = mark_rank.assign(
            lemma_rank=mark_rank.lemma_rank.fillna(0).astype(int),
            group=group_name,
        ).reset_index()
        groups_mark_rank.append(mark_rank)
    # Concatenating all results together
    joint_top_n = pd.concat(groups_top_n, ignore_index=True)
    # Selecting to upos tags, we are interested in
    joint_top_n = joint_top_n[joint_top_n.upos.isin(upos_tags)]
    # Creating color mapping
    color_mapping = discrete_color_mapping(mark_counts)
    # Formatting tables and saving for each upos
    datapane_objects = []
    datapane_objects.append(dp.Text("## Top 25 Words"))
    for upos, group in joint_top_n.groupby("upos"):
        datapane_objects.append(dp.Text(f"### {upos}"))
        table = tabulate(group)[ordering]
        table = color_words(table, upos, color_mapping)
        datapane_objects.append(dp.Table(table))
        # table_path = os.path.join(save_path, f"top_{top_n}_{upos.lower()}s.html")
        # table.to_html(table_path)
    joint_mark_ranks = pd.concat(groups_mark_rank, ignore_index=True)
    joint_mark_ranks = joint_mark_ranks[joint_mark_ranks.upos.isin(upos_tags)]
    datapane_objects.append(dp.Text("## Ranks of top 25 words in Mark"))
    for upos, group in joint_mark_ranks.groupby("upos"):
        datapane_objects.append(dp.Text(f"### {upos}"))
        table = (
            joint_mark_ranks.pivot(
                index="lemma", columns="group", values="lemma_rank"
            )
            .sort_values("The Gospel of Mark - MRK-ΚΑΤΑ ΜΑΡΚΟΝ")
            .drop(columns="The Gospel of Mark - MRK-ΚΑΤΑ ΜΑΡΚΟΝ")
            .reset_index()
        )
        datapane_objects.append(dp.DataTable(table))
        # table_path = os.path.join(save_path, f"rank_mark_{upos.lower()}s.html")
        # table.to_html(table_path)
    dp.Report(*datapane_objects).upload(
        "Top 25 Words Analysis", publicly_visible=True
    )
    print("Done")


if __name__ == "__main__":
    main(
        top_n=25,
        save_path="/work/data_wrangling/dat/greek/results",
        upos_tags=["ADJ", "NOUN", "VERB"],
    )
