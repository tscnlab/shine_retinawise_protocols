# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 18:17:31 2025

@author: malonso
"""
import pandas as pd
from pathlib import Path


eyes = ["left", "right"]


# get paths
script_dir = Path(__file__).resolve().parent
results_path = script_dir / "calibration results"


for e in eyes:

    # load file
    file_name = f"{results_path}/retinawise_calibration_results_{e}_eye.pkl"
    results = pd.read_pickle(file_name)

    # build new dataframe
    spectra = pd.DataFrame(
        results["spectrum"].to_list(), columns=list(range(380, 781))
    )

    new_results = pd.concat([results[["LED", "intensity"]], spectra], axis=1)

    # rename columns
    new_results = new_results.rename(
        columns={"LED": "Primary", "intensity": "Setting"}
    )

    # repeat intensity zero row
    zero_row = new_results[new_results["Primary"] == 0].iloc[0]

    # get other primaries
    primaries = new_results.loc[
        new_results["Primary"] != 0, "Primary"
    ].unique()

    # duplicate row and set primary value
    duplicates = pd.DataFrame([zero_row] * len(primaries))
    duplicates["Primary"] = primaries

    # append to main dataframe
    new_results = pd.concat(
        [new_results[new_results["Primary"] != 0], duplicates],
        ignore_index=True,
    )

    # sort data frame
    new_results = new_results.sort_values("Primary").reset_index(drop=True)
    new_results = new_results.sort_values(by=["Primary", "Setting"])

    # subtract one from primaries, to use same convention
    new_results["Primary"] -= 1

    # save new results as csv
    new_file_name = f"{results_path}/retinawise_spds_{e}_eye.csv"
    new_results.to_csv(new_file_name, index=False)
