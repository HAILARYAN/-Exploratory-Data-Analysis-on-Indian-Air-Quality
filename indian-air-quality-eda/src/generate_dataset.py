"""
generate_dataset.py

The machine I built this on had no internet access, so rather than downloading a
real CPCB extract I wrote this to produce one that behaves the way a real one
does. It writes data/city_day.csv: a day by day air quality record for 12 Indian
cities, carrying the same columns as the Kaggle "Air Quality Data in India"
dataset (PM2.5, PM10, NO2, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI,
AQI_Bucket) plus one extra Region column that I added because grouping north
against south turned out to be one of the more interesting cuts.

To be completely clear about it, this data is synthetic. The numbers are not
measurements of anything. What they are is structured, so that the EDA has real
patterns to find rather than noise:

    - winter (November to February) spikes in the north Indian cities like
      Delhi, Kanpur, Lucknow and Patna, driven in reality by crop burning
      season landing on top of cold air that traps pollution near the ground
    - coastal cities like Chennai, Mumbai and Kochi staying lower and flatter
      all year because the sea breeze disperses everything
    - a small weekday against weekend dip in NO2 and CO, following traffic
    - missing values scattered the way real sensors drop out, including a few
      multi-week outages rather than only isolated single days
    - a sharp COVID-19 lockdown drop from late March to May 2020, followed by a
      partial rather than complete recovery over that summer

Swap this file's output for a real extract whenever you like. As long as the
column names match, nothing else in the project needs to change.

Run it directly to rebuild data/city_day.csv:

    python src/generate_dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Fixed seed so that the dataset, the report numbers and the charts in the
# README all stay in agreement. Change it and every figure quoted downstream
# moves, which is exactly why it is pinned.
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "city_day.csv"

# ---------------------------------------------------------------------------
# City metadata.
#
#   base_pm25     roughly where the city sits on an average day
#   winter_boost  how hard the winter peak hits, 1.0 meaning no seasonality
#   traffic       scales the traffic-linked pollutants (NO2, CO, benzene)
# ---------------------------------------------------------------------------
CITIES = {
    "Delhi":       {"region": "North",  "base_pm25": 115, "winter_boost": 2.6,  "traffic": 1.3},
    "Lucknow":     {"region": "North",  "base_pm25": 100, "winter_boost": 2.3,  "traffic": 1.1},
    "Patna":       {"region": "North",  "base_pm25": 108, "winter_boost": 2.4,  "traffic": 1.0},
    "Kanpur":      {"region": "North",  "base_pm25": 112, "winter_boost": 2.5,  "traffic": 1.1},
    "Ahmedabad":   {"region": "West",   "base_pm25": 78,  "winter_boost": 1.6,  "traffic": 1.2},
    "Mumbai":      {"region": "West",   "base_pm25": 60,  "winter_boost": 1.3,  "traffic": 1.4},
    "Pune":        {"region": "West",   "base_pm25": 55,  "winter_boost": 1.3,  "traffic": 1.1},
    "Kolkata":     {"region": "East",   "base_pm25": 85,  "winter_boost": 1.8,  "traffic": 1.2},
    "Bengaluru":   {"region": "South",  "base_pm25": 45,  "winter_boost": 1.15, "traffic": 1.3},
    "Chennai":     {"region": "South",  "base_pm25": 42,  "winter_boost": 1.1,  "traffic": 1.1},
    "Hyderabad":   {"region": "South",  "base_pm25": 50,  "winter_boost": 1.2,  "traffic": 1.2},
    "Kochi":       {"region": "South",  "base_pm25": 30,  "winter_boost": 1.05, "traffic": 0.9},
}

START_DATE = "2015-01-01"
END_DATE = "2020-12-31"

# Every other pollutant is derived from that day's PM2.5 and then knocked about
# with its own noise level. That is what gives the correlation heatmap
# something real to show: the pollutants share a common driver, but the ones
# with high noise correlate visibly more weakly, which is also how it looks in
# the real data.
POLLUTANT_RATIOS = {
    "PM10":     dict(mult=1.7,   noise=0.15),
    "NO2":      dict(mult=0.35,  noise=0.25),
    "NH3":      dict(mult=0.20,  noise=0.30),
    "SO2":      dict(mult=0.12,  noise=0.35),
    "CO":       dict(mult=0.018, noise=0.30),
    "O3":       dict(mult=0.30,  noise=0.40),
    "Benzene":  dict(mult=0.03,  noise=0.45),
    "Toluene":  dict(mult=0.09,  noise=0.45),
    "Xylene":   dict(mult=0.02,  noise=0.50),
}


def seasonal_factor(day_of_year: int, winter_boost: float) -> float:
    """Multiplier for the time of year: high in winter, low through monsoon.

    A cosine shifted so the peak sits in mid January and the trough in mid
    July. The positive and negative halves are scaled differently because the
    winter climb is much steeper than the monsoon dip is deep.
    """
    radians = 2 * np.pi * (day_of_year - 15) / 365.25
    raw = np.cos(radians)
    factor = 1 + (winter_boost - 1) * max(raw, 0) - 0.35 * max(-raw, 0)
    return max(factor, 0.25)


def compute_aqi_from_pm25(pm25: float) -> float:
    """Convert a PM2.5 concentration into an AQI value.

    These are the CPCB PM2.5 sub-index breakpoints, applied with the usual
    linear interpolation inside each band. The real AQI is the worst sub-index
    across all pollutants rather than PM2.5 alone, but PM2.5 is the driver on
    the overwhelming majority of Indian city-days, so this is a fair
    simplification for the purpose here.
    """
    breakpoints = [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ]
    pm25 = max(pm25, 0)
    for lo_conc, hi_conc, lo_idx, hi_idx in breakpoints:
        if lo_conc <= pm25 <= hi_conc:
            return lo_idx + (hi_idx - lo_idx) * (pm25 - lo_conc) / (hi_conc - lo_conc)
    return 500.0


def aqi_bucket(aqi: float) -> str:
    """CPCB category name for an AQI value."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def generate() -> pd.DataFrame:
    """Build the full dataframe, city by city and day by day."""
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    rows = []

    for city, meta in CITIES.items():
        base = meta["base_pm25"]
        winter_boost = meta["winter_boost"]
        traffic = meta["traffic"]

        for d in dates:
            season = seasonal_factor(d.dayofyear, winter_boost)

            # Weekends run lighter on the traffic-linked pollutants.
            weekday_factor = 0.85 if d.weekday() >= 5 else 1.0

            # Lockdown proper is the hard drop. The months after it reopened
            # partially, so they get a softer discount rather than snapping
            # straight back to normal.
            covid_factor = 1.0
            if pd.Timestamp("2020-03-25") <= d <= pd.Timestamp("2020-05-31"):
                covid_factor = 0.35
            elif pd.Timestamp("2020-06-01") <= d <= pd.Timestamp("2020-09-30"):
                covid_factor = 0.75

            # A slow background improvement of just under 1% a year, standing
            # in for emissions policy gradually biting.
            years_elapsed = (d - pd.Timestamp(START_DATE)).days / 365.25
            trend_factor = 1 - 0.008 * years_elapsed

            noise = rng.normal(1.0, 0.18)
            pm25 = max(base * season * covid_factor * trend_factor * noise, 3)

            row = {"City": city, "Date": d, "PM2.5": round(pm25, 1)}

            for pollutant, cfg in POLLUTANT_RATIOS.items():
                value = pm25 * cfg["mult"] * traffic * weekday_factor
                value *= rng.normal(1.0, cfg["noise"])
                row[pollutant] = round(max(value, 0), 2)

            aqi_value = compute_aqi_from_pm25(pm25)
            row["AQI"] = round(aqi_value, 0)
            row["AQI_Bucket"] = aqi_bucket(aqi_value)
            row["Region"] = meta["region"]

            rows.append(row)

    df = pd.DataFrame(rows)

    # ---- Extreme events, before missingness is applied ---------------------
    # Festival firecrackers, landfill fires, stubble smoke blowing in. Each
    # spike gets its own multiplier rather than all forty sharing one, and the
    # AQI is recomputed afterwards so that the spike and its index still agree.
    # Getting that wrong leaves 40 rows where PM2.5 has quadrupled and the AQI
    # has not moved, which quietly drags down the headline correlation.
    spike_idx = rng.choice(df.index, size=40, replace=False)
    df.loc[spike_idx, "PM2.5"] = (
        df.loc[spike_idx, "PM2.5"] * rng.uniform(2.5, 4.5, size=len(spike_idx))
    ).round(1)
    df.loc[spike_idx, "AQI"] = (
        df.loc[spike_idx, "PM2.5"].apply(compute_aqi_from_pm25).round(0)
    )
    df.loc[spike_idx, "AQI_Bucket"] = df.loc[spike_idx, "AQI"].apply(aqi_bucket)

    # ---- Missingness, the way real monitoring stations lose data -----------
    # First the everyday scatter: individual readings that failed validation or
    # never made it off the station.
    missing_cols = [
        "PM2.5", "PM10", "NO2", "NH3", "SO2", "CO", "O3",
        "Benzene", "Toluene", "Xylene", "AQI",
    ]
    for col in missing_cols:
        frac = rng.uniform(0.02, 0.08)
        df.loc[rng.random(len(df)) < frac, col] = np.nan

    # Then the ugly kind: four cities lose a monitor for a couple of weeks at a
    # stretch. This matters for the EDA, because a scatter of isolated gaps is
    # trivial to interpolate over and a three week hole genuinely is not. Rows
    # are still grouped by city at this point, so a city's block is contiguous
    # and the slice cannot spill into the next city.
    for city in rng.choice(list(CITIES.keys()), size=4, replace=False):
        city_idx = df.index[df["City"] == city]
        outage_start = int(rng.choice(city_idx[:-30]))
        outage_len = int(rng.integers(10, 30))
        outage_rows = range(outage_start, outage_start + outage_len)
        df.loc[df.index.isin(outage_rows), ["PM2.5", "PM10", "AQI"]] = np.nan

    # The bucket is a label for the AQI, so it cannot survive the AQI going
    # missing.
    df.loc[df["AQI"].isna(), "AQI_Bucket"] = np.nan

    return df.sort_values(["City", "Date"]).reset_index(drop=True)


def main():
    df = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(df):,} rows across {df['City'].nunique()} cities.")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print("\nMissing values per column:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()
