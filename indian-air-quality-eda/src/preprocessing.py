"""
preprocessing.py

This is where the actual cleaning happens. Air quality sensors go offline all
the time because of power cuts, maintenance, or plain old hardware failure, so
real world data like this always arrives with gaps in it.

The steps, in the order they run:

    1. report how much data is missing and which columns it is missing from
    2. fill those gaps by interpolating, done separately for each city
    3. flag outliers with the IQR method, without removing them, because a
       firecracker spike on Diwali is a real event and not junk data
    4. add extra columns like Year, Month, Season and Weekday that make the
       later grouping and plotting much less painful

Every function here returns a new dataframe instead of editing the one it was
given. I learned that the hard way after a couple of confusing sessions where I
could not tell which version of the dataframe I was actually looking at.
"""

from __future__ import annotations

import pandas as pd

from src.data_loader import POLLUTANT_COLUMNS


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table of how many values each column is missing, and what
    share of the column that works out to.

    Columns with nothing missing are dropped from the result, since a report
    full of zeros is just noise to read past.
    """
    n = len(df)
    missing = df.isna().sum()
    pct = (missing / n * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return report[report["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Split the Date column out into Year, Month, Weekday and Season.

    Doing this once up front means I am not re-deriving the same values every
    time I want to group by month or by season further down.
    """
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.month_name()
    df["Weekday"] = df["Date"].dt.day_name()
    df["IsWeekend"] = df["Date"].dt.weekday >= 5

    # Indian seasons do not line up with the usual four season calendar, so I
    # am mapping them by hand based on when the winter smog and the monsoon
    # rains actually turn up.
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Summer", 4: "Summer", 5: "Summer",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Post-Monsoon", 11: "Post-Monsoon",
    }
    df["Season"] = df["Month"].map(season_map)
    return df


def interpolate_by_city(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Fill missing pollutant readings with time based interpolation, one city
    at a time.

    Doing it per city is the whole point. If you interpolate across the full
    dataset in one go, a missing Delhi winter reading can end up filled with
    something influenced by Chennai's much lower summer numbers, which is
    nonsense. City by city means each place only ever borrows from its own
    trend.

    Interpolation is done against the actual dates rather than row position, so
    a two week outage is weighted like a two week outage. Any gaps left at the
    very start or end of a city's timeline get the nearest available value in
    whichever direction has one, because there is nothing on the other side to
    interpolate from.
    """
    columns = columns or [
        c for c in POLLUTANT_COLUMNS + ["AQI"] if c in df.columns
    ]
    df = df.sort_values(["City", "Date"]).copy()

    # Deliberately a plain loop rather than groupby.apply. The apply version
    # needs the include_groups argument to avoid a deprecation warning, and
    # that argument only exists from pandas 2.2 onwards, which quietly breaks
    # the project on older pandas and on whatever Colab happens to ship.
    filled = []
    for _, group in df.groupby("City", observed=True, sort=False):
        group = group.set_index("Date")
        group[columns] = group[columns].interpolate(
            method="time", limit_direction="both"
        )
        filled.append(group.reset_index())

    return pd.concat(filled, ignore_index=True)


def flag_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.Series:
    """Standard IQR outlier check, returning True or False for every row.

    Anything more than k interquartile ranges outside the middle 50% counts as
    an outlier. k of 1.5 is the usual convention.
    """
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (df[column] < lower) | (df[column] > upper)


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run everything above in the right order and hand back a dataframe that
    is ready to analyse."""
    df = add_time_features(df)
    df = interpolate_by_city(df)

    # A few rows can still be missing PM2.5 if a city's series happened to
    # start or end inside a gap. There is nothing left to interpolate from in
    # that case, so those rows get dropped.
    df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)

    # Flagging outliers rather than deleting them. Spikes from festivals and
    # crop burning are genuinely part of the story this data is telling.
    df["PM2.5_outlier"] = flag_outliers_iqr(df, "PM2.5")

    return df


if __name__ == "__main__":
    from src.data_loader import load_data

    raw = load_data()
    print("Missing value report before cleaning:")
    print(missing_value_report(raw))

    cleaned = clean_pipeline(raw)
    print("\nMissing value report after cleaning:")
    print(missing_value_report(cleaned))
    print(f"\nOutliers flagged: {cleaned['PM2.5_outlier'].sum()}")
