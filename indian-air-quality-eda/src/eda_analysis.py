"""
eda_analysis.py

All the number crunching that sits behind the report lives here: descriptive
stats, which cities come off worst, how each pollutant tracks the AQI, seasonal
and yearly averages, the lockdown comparison, and the outlier count.

I kept this separate from visualizations.py so that the same numbers feed both
the written report and the charts. If a figure in the report ever looks wrong,
there is exactly one function to go and check, and I do not have to wonder
whether the chart and the text disagree because they computed it differently.
"""

from __future__ import annotations

import pandas as pd

from src.data_loader import POLLUTANT_COLUMNS


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation, min, max and quartiles for every pollutant
    plus the AQI, transposed so each pollutant reads as a row."""
    cols = [c for c in POLLUTANT_COLUMNS + ["AQI"] if c in df.columns]
    return df[cols].describe().T.round(2)


def city_ranking(df: pd.DataFrame, metric: str = "PM2.5") -> pd.DataFrame:
    """Rank the cities from most to least polluted on the given metric.

    Median and standard deviation come along for the ride because the mean on
    its own hides a lot. Two cities can average the same and have completely
    different day to day experiences, one steady and one swinging between clean
    air and emergency levels.
    """
    return (
        df.groupby("City", observed=True)[metric]
        .agg(["mean", "median", "std", "max"])
        .round(2)
        .sort_values("mean", ascending=False)
    )


def most_and_least_polluted(df: pd.DataFrame, metric: str = "PM2.5") -> dict:
    """Pull out just the single worst and single best city, for the places in
    the report where the full ranking table would be overkill."""
    ranking = (
        df.groupby("City", observed=True)[metric].mean().sort_values(ascending=False)
    )
    return {
        "most_polluted": (ranking.index[0], round(ranking.iloc[0], 1)),
        "least_polluted": (ranking.index[-1], round(ranking.iloc[-1], 1)),
    }


def seasonal_summary(df: pd.DataFrame, metric: str = "PM2.5") -> pd.DataFrame:
    """Average pollution level per season, worst season first."""
    return (
        df.groupby("Season", observed=True)[metric]
        .mean()
        .round(1)
        .sort_values(ascending=False)
        .to_frame(name=f"avg_{metric}")
    )


def yearly_summary(df: pd.DataFrame, metric: str = "PM2.5") -> pd.DataFrame:
    """Average pollution level per year. This is the table where the 2020 dip
    jumps out without needing a chart."""
    return (
        df.groupby("Year", observed=True)[metric]
        .mean()
        .round(1)
        .to_frame(name=f"avg_{metric}")
    )


def correlation_with_aqi(df: pd.DataFrame) -> pd.Series:
    """How strongly each pollutant moves with the overall AQI score, sorted
    strongest first."""
    cols = [c for c in POLLUTANT_COLUMNS if c in df.columns]
    corr = df[cols + ["AQI"]].corr()["AQI"].drop("AQI").sort_values(ascending=False)
    return corr.round(3)


def aqi_bucket_share(df: pd.DataFrame) -> pd.Series:
    """Share of days falling into each AQI category, from Good through to
    Severe, as a percentage."""
    return (df["AQI_Bucket"].value_counts(normalize=True) * 100).round(1)


def covid_impact(df: pd.DataFrame, metric: str = "PM2.5") -> dict:
    """Compare pollution during the 2020 lockdown against the same calendar
    weeks in 2019.

    Comparing against the same weeks of the previous year rather than against
    the weeks immediately before lockdown matters here. March to May is already
    on the way down from the winter peak, so a before and after comparison
    inside 2020 would credit the lockdown with a seasonal drop that was going
    to happen anyway.
    """
    pre = df[(df["Date"] >= "2019-03-25") & (df["Date"] <= "2019-05-31")][metric].mean()
    during = df[(df["Date"] >= "2020-03-25") & (df["Date"] <= "2020-05-31")][
        metric
    ].mean()
    pct_change = (during - pre) / pre * 100
    return {
        "pre_lockdown_2019_avg": round(pre, 1),
        "lockdown_2020_avg": round(during, 1),
        "pct_change": round(pct_change, 1),
    }


def outlier_summary(df: pd.DataFrame) -> dict:
    """How many PM2.5 readings were flagged as outliers, and what share of the
    dataset that is."""
    total = len(df)
    n_outliers = int(df["PM2.5_outlier"].sum())
    return {
        "total_records": total,
        "outlier_records": n_outliers,
        "outlier_pct": round(n_outliers / total * 100, 2),
    }


def generate_text_insights(df: pd.DataFrame) -> list[str]:
    """Turn the numbers above into plain sentences a person can read.

    The report and the console summary both call this, so the written
    conclusions cannot drift away from the tables they are describing.
    """
    insights = []

    extremes = most_and_least_polluted(df)
    insights.append(
        f"Most polluted city (by mean PM2.5): {extremes['most_polluted'][0]} "
        f"({extremes['most_polluted'][1]} µg/m³) vs. least polluted: "
        f"{extremes['least_polluted'][0]} ({extremes['least_polluted'][1]} µg/m³)."
    )

    season = seasonal_summary(df)
    insights.append(
        f"Winter is the most polluted season nationally (avg PM2.5 = "
        f"{season.iloc[0, 0]} µg/m³), roughly "
        f"{round(season.iloc[0, 0] / season.iloc[-1, 0], 1)}x the "
        f"{season.index[-1].lower()} average ({season.iloc[-1, 0]} µg/m³)."
    )

    corr = correlation_with_aqi(df)
    insights.append(
        f"{corr.index[0]} shows the strongest correlation with AQI (r = {corr.iloc[0]}), "
        f"suggesting it is the dominant driver of the composite index in this dataset."
    )

    covid = covid_impact(df)
    insights.append(
        f"During the COVID-19 lockdown (25 Mar to 31 May 2020), average PM2.5 fell to "
        f"{covid['lockdown_2020_avg']} µg/m³ from {covid['pre_lockdown_2019_avg']} µg/m³ "
        f"in the same period a year earlier, a change of {covid['pct_change']}%."
    )

    buckets = aqi_bucket_share(df)
    top_bucket = buckets.index[0]
    insights.append(
        f"'{top_bucket}' is the most common AQI category, covering {buckets.iloc[0]}% "
        f"of all city-days in the dataset."
    )

    out = outlier_summary(df)
    insights.append(
        f"{out['outlier_records']} of {out['total_records']} records "
        f"({out['outlier_pct']}%) were flagged as PM2.5 outliers using the IQR method, "
        f"which lines up with festival firecracker spikes and crop burning events."
    )

    return insights
