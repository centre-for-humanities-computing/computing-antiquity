"""Module for formally describing analyses in the project
as well as running them"""

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, List

import datapane as dp
import pandas as pd

from utils.metadata import fetch_metadata

# TODO: type this a bit better, this is pathetic
DatapaneElement = Any
RESULTS_PATH = "/work/data_wrangling/dat/greek/results/"


@dataclass
class Analysis:
    """Class for describing analyses.

    Attributes
    ----------
    short_name: str
        Name to save the results to.
    datapane_name: str
        Name of the datapane report.
    load_input: function () -> DataFrame
        Function to load input data for the analysis.
    conduct_analysis: function (DataFrame) -> DataFrame
        Function, that analyses input data and outputs results.
    produce_elements: function (DataFrame) -> list of Any
        Function that produces the Datapane elements
        based on the results.
    """

    short_name: str
    datapane_name: str
    load_input: Callable[[], pd.DataFrame]
    conduct_analysis: Callable[[pd.DataFrame], pd.DataFrame]
    produce_elements: Callable[[pd.DataFrame], List[DatapaneElement]]

    def run(self, lazy: bool = False) -> None:
        """Method that runs the analysis.

        Parameters
        ----------
        lazy: bool, default False
            Describes whether the method should try to load
            the results from disk or not.
        """
        print(f"Running analysis: {self.datapane_name}")
        print("    Logging into Datapane")
        subprocess.call(["bash", "datapane_login.sh"])
        print("    1. Loading data")
        input_data = self.load_input()
        metadata = fetch_metadata()
        metadata = metadata[~metadata.skal_fjernes]
        metadata = metadata[
            ["id_nummer", "værk", "forfatter", "group", "etnicitet"]
        ]
        out_file = os.path.join(RESULTS_PATH, f"{self.short_name}.pkl")
        if not lazy:
            print("    2. Running analysis")
            results = self.conduct_analysis(input_data)
            print("    3. Saving results")
            results.to_pickle(out_file)
        else:
            try:
                print("    Found results on disk, skipping analysis.")
                results = pd.read_pickle(out_file)
            except FileNotFoundError:
                print("    2. Running analysis")
                results = self.conduct_analysis(input_data)
                print("    3. Saving results")
                results.to_pickle(out_file)
        results = results.merge(metadata, on="id_nummer", how="inner")
        print("    4. Producing report")
        datapane_elements = self.produce_elements(results)
        print("    5. Uploading to Datapane")
        dp.Report(*datapane_elements).upload(
            self.datapane_name, publicly_visible=True
        )
