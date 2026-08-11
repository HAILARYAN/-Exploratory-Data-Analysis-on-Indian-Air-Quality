"""
visualizations.py

All the plotting lives here. Each function draws one chart, saves it as a PNG,
and returns the path it wrote to so that main.py can log where everything went.

Keeping the plotting apart from the number crunching in eda_analysis.py means I
can fiddle with how a chart looks without going anywhere near the maths, and it
lets the notebook import the same functions instead of keeping a second,
slightly different copy of every plot that then drifts out of sync.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib

# Agg is the non-interactive backend, and a script running headless on a CI
# box or over plain SSH needs it in order to write a PNG rather than trying to
# open a window.
#
# The check around it is not optional though, and I found that out the
# annoying way. This module used to call matplotlib.use("Agg") unconditionally
# at import. That is fine from main.py, but in a notebook it is quietly
# destructive: `%matplotlib inline` sets the inline backend, then importing
# this module for one helper function swaps it out for Agg, and from that cell
# onward every plt.show() renders precisely nothing. No error, no warning, no
# output. Half the charts in my notebook vanished and I assumed I had broken
# the plotting code.
#
# So only switch when the current backend is not one of the notebook ones.
_BACKEND = matplotlib.get_backend().lower()
if not any(name in _BACKEND for name in ("inline", "nbagg", "ipympl", "widget")):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 13

MONTH_ORDER = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
AQI_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
AQI_COLORS = ["#2ecc71", "#a3d977", "#f4d03f", "#f39c12", "#e74c3c", "#7b241c"]


def _savefig(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def ordered_city_labels(
    df: pd.DataFrame, metric: str = "PM2.5", stat: str = "mean"
) -> list[str]:
    """Return city names as plain strings, ordered worst to best on the metric.

    The `astype(str)` is not cosmetic, it is the whole point of this function.

    The City column is a pandas categorical, so `groupby(...).mean().sort_values()`
    hands back a CategoricalIndex. When that index goes into seaborn as an
    axis, seaborn orders the axis by the dtype's own category order, which is
    alphabetical, and quietly throws away the sort. The bars then sit in
    alphabetical positions while anything drawn by enumerating the sorted
    values, such as the value label on the end of each bar, sits in ranked
    positions. Nothing errors. You get a chart captioned "sorted by average"
    that is not sorted, with every number attached to the wrong city.

    I only spotted it because Ahmedabad was sitting at the top of a chart
    labelled 154.3 when its actual mean is about 80. Converting to plain
    strings makes seaborn respect the order it is given.
    """
    values = df.groupby("City", observed=True)[metric].agg(stat)
    return values.sort_values(ascending=False).index.astype(str).tolist()


def plot_missing_values(report: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=report["missing_pct"], y=report.index, ax=ax, color="#e67e22")
    ax.set_xlabel("% Missing")
    ax.set_ylabel("")
    ax.set_title("Missing Values by Column (Raw Data)")
    for i, v in enumerate(report["missing_pct"]):
        ax.text(v + 0.05, i, f"{v}%", va="center", fontsize=9)
    return _savefig(fig, out_dir, "01_missing_values")


def plot_pollutant_distributions(df: pd.DataFrame, pollutants: list[str], out_dir: Path) -> Path:
    n = len(pollutants)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, pol in enumerate(pollutants):
        sns.histplot(df[pol].dropna(), kde=True, ax=axes[i], color="#3498db")
        axes[i].set_title(pol)
        axes[i].set_xlabel("")

    for j in range(len(pollutants), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Distribution of Pollutant Concentrations", y=1.02, fontsize=15)
    fig.tight_layout()
    return _savefig(fig, out_dir, "02_pollutant_distributions")


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str], out_dir: Path) -> Path:
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Between Pollutants & AQI")
    return _savefig(fig, out_dir, "03_correlation_heatmap")


def plot_city_avg_pm25(df: pd.DataFrame, out_dir: Path) -> Path:
    labels = ordered_city_labels(df, "PM2.5", "mean")
    city_avg = df.groupby("City", observed=True)["PM2.5"].mean()
    values = [city_avg[city] for city in labels]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        x=values, y=labels, hue=labels, order=labels, hue_order=labels,
        palette="rocket_r", legend=False, ax=ax,
    )
    ax.set_xlabel("Average PM2.5 (µg/m³)")
    ax.set_ylabel("")
    ax.set_title("Average PM2.5 by City (2015-2020)")

    # India's own annual limit. The WHO guideline is 5 µg/m³, which would sit
    # off the left edge of this chart entirely, so the national one is the
    # more useful line to draw.
    ax.axvline(40, color="#2c3e50", linestyle="--", linewidth=1.5)
    ax.text(41, len(labels) - 0.55, "India NAAQS annual limit (40)",
            fontsize=9, color="#2c3e50")

    for i, value in enumerate(values):
        ax.text(value + 1, i, f"{value:.1f}", va="center", fontsize=9)
    return _savefig(fig, out_dir, "04_city_avg_pm25")


def plot_monthly_trend(df: pd.DataFrame, out_dir: Path) -> Path:
    monthly = (
        df.groupby(["MonthName", "City"], observed=True)["PM2.5"]
        .mean()
        .reset_index()
    )
    monthly["MonthName"] = pd.Categorical(monthly["MonthName"], MONTH_ORDER, ordered=True)
    monthly = monthly.sort_values("MonthName")

    top_cities = (
        df.groupby("City", observed=True)["PM2.5"].mean().sort_values(ascending=False).head(6).index
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    for city in top_cities:
        sub = monthly[monthly["City"] == city]
        ax.plot(sub["MonthName"], sub["PM2.5"], marker="o", label=city)
    ax.set_title("Seasonal PM2.5 Pattern (Top 6 Most Polluted Cities)")
    ax.set_ylabel("Average PM2.5 (µg/m³)")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    ax.legend(title="City", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return _savefig(fig, out_dir, "05_seasonal_pm25_trend")


def plot_yearly_trend(df: pd.DataFrame, out_dir: Path) -> Path:
    yearly = df.groupby(["Year"], observed=True)["PM2.5"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=yearly, x="Year", y="PM2.5", marker="o", ax=ax, color="#8e44ad", linewidth=2.5)
    ax.set_title("National Average PM2.5 Trend by Year (incl. 2020 COVID dip)")
    ax.set_ylabel("Average PM2.5 (µg/m³)")
    ax.set_xticks(yearly["Year"])
    return _savefig(fig, out_dir, "06_yearly_trend")


def plot_daily_timeseries(df: pd.DataFrame, city: str, out_dir: Path) -> Path:
    # the raw daily line is noisy on its own, so a rolling mean is added
    # on top to make the actual trend easier to see
    sub = df[df["City"] == city].sort_values("Date")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sub["Date"], sub["PM2.5"], color="#c0392b", linewidth=0.8, alpha=0.85)
    ax.plot(sub["Date"], sub["PM2.5"].rolling(30).mean(), color="#2c3e50", linewidth=2,
            label="30-day rolling mean")
    ax.axvspan(pd.Timestamp("2020-03-25"), pd.Timestamp("2020-05-31"),
               color="grey", alpha=0.25, label="COVID lockdown")
    ax.set_title(f"Daily PM2.5 Over Time in {city}")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend()
    fig.tight_layout()
    return _savefig(fig, out_dir, f"07_daily_timeseries_{city.lower()}")


def plot_aqi_bucket_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = df["AQI_Bucket"].value_counts()
    counts = counts.reindex([b for b in AQI_ORDER if b in counts.index])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color=AQI_COLORS[: len(counts)])
    ax.set_title("Distribution of AQI Categories (All Cities, All Days)")
    ax.set_ylabel("Number of Days")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, f"{v:,}", ha="center", fontsize=9)
    return _savefig(fig, out_dir, "08_aqi_bucket_distribution")


def plot_city_boxplot(df: pd.DataFrame, out_dir: Path) -> Path:
    order = ordered_city_labels(df, "PM2.5", "median")
    fig, ax = plt.subplots(figsize=(11, 6))
    # hue_order has to be passed alongside order. Without it the boxes sit in
    # ranked positions but take their colour from the alphabetical category
    # order, so the palette gradient comes out shuffled.
    sns.boxplot(
        data=df, x="City", y="PM2.5", hue="City", order=order, hue_order=order,
        palette="mako", showfliers=False, legend=False, ax=ax,
    )
    ax.set_title("PM2.5 Spread by City (outliers hidden for readability)")
    ax.set_xlabel("")
    ax.set_ylabel("PM2.5 (µg/m³)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    return _savefig(fig, out_dir, "09_city_boxplot")


def plot_weekday_effect(df: pd.DataFrame, out_dir: Path) -> Path:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_avg = df.groupby("Weekday", observed=True)["NO2"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3498db"] * 5 + ["#95a5a6"] * 2
    ax.bar(weekday_avg.index, weekday_avg.values, color=colors)
    ax.set_title("Average NO2 by Day of Week (Traffic Signal)")
    ax.set_ylabel("Average NO2 (µg/m³)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return _savefig(fig, out_dir, "10_weekday_no2_effect")


def plot_region_comparison(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df, x="Region", y="PM2.5", hue="Region", ax=ax, palette="crest", cut=0, legend=False)
    ax.set_title("PM2.5 Distribution by Region")
    fig.tight_layout()
    return _savefig(fig, out_dir, "11_region_comparison")
