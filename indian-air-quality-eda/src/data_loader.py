"""
data_loader.py

This file has one job: get the raw CSV into memory with the right dtypes and
hand it back. Nothing else.

I split it out from the cleaning logic on purpose. When loading and cleaning
live in the same function it gets tangled quickly, and then when a number looks
wrong you cannot tell whether the file was read badly or the cleaning did
something odd to it. Everything to do with missing values, outliers and feature
engineering lives in preprocessing.py instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "city_day.csv"

# The pollutant columns the dataset carries. Kept in one place so that adding
# or removing a pollutant does not mean hunting through four other files.
POLLUTANT_COLUMNS = [
    "PM2.5", "PM10", "NO2", "NH3", "SO2", "CO", "O3",
    "Benzene", "Toluene", "Xylene",
]


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the city_day air quality CSV into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Where the CSV lives. Defaults to data/city_day.csv.

    Returns
    -------
    pd.DataFrame
        The raw dataframe, with Date parsed as a real datetime and every
        pollutant column forced to numeric.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the dataset at {path}. "
            f"Run `python src/generate_dataset.py` first to create it."
        )

    df = pd.read_csv(path, parse_dates=["Date"])

    # A CSV will happily load a numeric column as object dtype if even one cell
    # has a stray space or a dash in it, and then every mean and correlation
    # downstream silently misbehaves. Forcing numeric here turns those cells
    # into NaN, which the interpolation step is already built to handle.
    for col in POLLUTANT_COLUMNS + ["AQI"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Category dtype cuts the memory footprint down and speeds up the groupby
    # calls, which matters once you are grouping by city on 26k rows repeatedly.
    df["City"] = df["City"].astype("category")
    if "Region" in df.columns:
        df["Region"] = df["Region"].astype("category")
    if "AQI_Bucket" in df.columns:
        df["AQI_Bucket"] = df["AQI_Bucket"].astype("category")

    return df


if __name__ == "__main__":
    # Quick sanity check for when I run this file on its own.
    data = load_data()
    print(data.info())
    print(data.head())
