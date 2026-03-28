from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "fare_amount",
    "trip_distance",
    "pulocationid",
]

OPTIONAL_NUMERIC_COLUMNS = [
    "passenger_count",
    "payment_type",
    "ratecodeid",
    "tip_amount",
    "dolocationid",
]

RAW_PREVIEW_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "fare_amount",
    "trip_distance",
    "pulocationid",
]

CLEAN_PREVIEW_COLUMNS = RAW_PREVIEW_COLUMNS + [
    "pickup_date",
    "pickup_day_name",
    "hour",
    "surge",
]

ZONE_LOOKUP_COLUMNS = ["locationid", "borough", "zone"]
ZONE_CENTROID_COLUMNS = ["locationid", "latitude", "longitude", "zone", "borough"]
WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


@dataclass
class BaseAnalysis:
    raw_df: pd.DataFrame
    cleaned_df: pd.DataFrame
    cleaning_report: dict[str, float]
    cleaning_breakdown: pd.DataFrame
    before_sample: pd.DataFrame
    after_sample: pd.DataFrame


@dataclass
class FilteredAnalysis:
    filtered_df: pd.DataFrame
    hourly_summary: pd.DataFrame
    location_summary: pd.DataFrame
    daily_summary: pd.DataFrame
    weekday_summary: pd.DataFrame
    duration_summary: pd.DataFrame
    correlation_matrix: pd.DataFrame
    summary_statistics: pd.DataFrame
    busiest_hours: pd.DataFrame
    surge_hours: pd.DataFrame
    efficient_hours: pd.DataFrame
    inefficient_hours: pd.DataFrame
    outliers: pd.DataFrame


def load_dataset(source: Any) -> pd.DataFrame:
    df = pd.read_csv(source)
    df.columns = [column.strip().lower() for column in df.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing_text}")

    return df


def load_zone_lookup(source: str | Path | Any) -> pd.DataFrame:
    zone_lookup = pd.read_csv(source)
    zone_lookup.columns = [column.strip().lower() for column in zone_lookup.columns]

    missing_columns = [column for column in ZONE_LOOKUP_COLUMNS if column not in zone_lookup.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Taxi zone lookup is missing required columns: {missing_text}")

    zone_lookup = zone_lookup[ZONE_LOOKUP_COLUMNS].copy()
    zone_lookup["locationid"] = pd.to_numeric(zone_lookup["locationid"], errors="coerce")
    zone_lookup = zone_lookup.dropna(subset=["locationid"]).copy()
    zone_lookup["locationid"] = zone_lookup["locationid"].astype(int)
    return zone_lookup


def load_zone_centroids(source: str | Path | Any) -> pd.DataFrame:
    centroids = pd.read_csv(source)
    centroids.columns = [column.strip().lower() for column in centroids.columns]

    missing_columns = [
        column for column in ZONE_CENTROID_COLUMNS if column not in centroids.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Taxi zone centroid file is missing required columns: {missing_text}")

    centroids = centroids[ZONE_CENTROID_COLUMNS].copy()
    centroids["locationid"] = pd.to_numeric(centroids["locationid"], errors="coerce")
    centroids["latitude"] = pd.to_numeric(centroids["latitude"], errors="coerce")
    centroids["longitude"] = pd.to_numeric(centroids["longitude"], errors="coerce")
    centroids = centroids.dropna(subset=["locationid", "latitude", "longitude"]).copy()
    centroids["locationid"] = centroids["locationid"].astype(int)
    return centroids


def add_zone_names(cleaned_df: pd.DataFrame, zone_lookup: pd.DataFrame) -> pd.DataFrame:
    columns_to_remove = ["pickup_borough", "pickup_zone", "pickup_zone_label"]
    cleaned_df = cleaned_df.drop(
        columns=[column for column in columns_to_remove if column in cleaned_df.columns],
        errors="ignore",
    )

    renamed_lookup = zone_lookup.rename(
        columns={
            "locationid": "pulocationid",
            "borough": "pickup_borough",
            "zone": "pickup_zone",
        }
    )
    enriched_df = cleaned_df.merge(renamed_lookup, on="pulocationid", how="left")
    enriched_df["pickup_borough"] = enriched_df["pickup_borough"].fillna("Unknown Borough")
    enriched_df["pickup_zone"] = enriched_df["pickup_zone"].fillna(
        "Unknown Pickup Location"
    )
    enriched_df["pickup_zone_label"] = (
        enriched_df["pickup_zone"]
        + ", "
        + enriched_df["pickup_borough"]
        + " (ID "
        + enriched_df["pulocationid"].astype(str)
        + ")"
    )
    return enriched_df


def _compute_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if series.empty or pd.isna(std) or np.isclose(std, 0):
        return pd.Series(0.0, index=series.index, dtype=float)
    return (series - series.mean()) / std


def _scale_series(series: pd.Series, target_min: float, target_max: float) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float, index=series.index)

    series_min = series.min()
    series_max = series.max()
    if pd.isna(series_min) or pd.isna(series_max) or np.isclose(series_min, series_max):
        midpoint = (target_min + target_max) / 2
        return pd.Series(midpoint, index=series.index, dtype=float)

    normalized = (series - series_min) / (series_max - series_min)
    return normalized * (target_max - target_min) + target_min


def _build_cleaning_breakdown(report: dict[str, float]) -> pd.DataFrame:
    total_rows = report["initial_rows"] or 1
    rows = [
        ("Initial rows", report["initial_rows"]),
        ("Rows removed", report["rows_removed"]),
        ("Null rows removed", report["null_rows_removed"]),
        ("Invalid distance rows", report["invalid_distance_rows"]),
        ("Invalid fare rows", report["invalid_fare_rows"]),
        ("Invalid trips removed", report["invalid_trips_removed"]),
        ("Final clean rows", report["final_rows"]),
    ]
    breakdown = pd.DataFrame(rows, columns=["metric", "count"])
    breakdown["percent_of_initial"] = (breakdown["count"] / total_rows * 100).round(2)
    return breakdown


def clean_and_engineer_data(df: pd.DataFrame) -> BaseAnalysis:
    working_df = df.copy()

    working_df["tpep_pickup_datetime"] = pd.to_datetime(
        working_df["tpep_pickup_datetime"], errors="coerce"
    )
    working_df["tpep_dropoff_datetime"] = pd.to_datetime(
        working_df["tpep_dropoff_datetime"], errors="coerce"
    )

    numeric_columns = [
        "fare_amount",
        "trip_distance",
        "pulocationid",
        *[column for column in OPTIONAL_NUMERIC_COLUMNS if column in working_df.columns],
    ]
    for column in numeric_columns:
        working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

    before_sample = working_df[RAW_PREVIEW_COLUMNS].head(8).copy()

    null_mask = working_df[REQUIRED_COLUMNS].isna().any(axis=1)
    invalid_distance_mask = working_df["trip_distance"] <= 0
    invalid_fare_mask = working_df["fare_amount"] <= 0
    invalid_trip_mask = invalid_distance_mask | invalid_fare_mask

    cleaned_df = working_df.loc[~null_mask & ~invalid_trip_mask].copy()
    cleaned_df["pulocationid"] = cleaned_df["pulocationid"].astype(int)
    cleaned_df["pickup_date"] = cleaned_df["tpep_pickup_datetime"].dt.date
    cleaned_df["pickup_day_name"] = cleaned_df["tpep_pickup_datetime"].dt.day_name()
    cleaned_df["day_type"] = np.where(
        cleaned_df["tpep_pickup_datetime"].dt.weekday >= 5, "Weekend", "Weekday"
    )
    cleaned_df["hour"] = cleaned_df["tpep_pickup_datetime"].dt.hour.astype(int)
    cleaned_df["surge"] = cleaned_df["fare_amount"] / cleaned_df["trip_distance"]
    cleaned_df["trip_duration_minutes"] = (
        cleaned_df["tpep_dropoff_datetime"] - cleaned_df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    cleaned_df["trip_duration_minutes"] = cleaned_df["trip_duration_minutes"].clip(lower=0)
    cleaned_df["surge_zscore"] = _compute_zscore(cleaned_df["surge"])
    cleaned_df["is_surge_outlier"] = cleaned_df["surge_zscore"].abs() >= 3

    after_sample = cleaned_df[CLEAN_PREVIEW_COLUMNS].head(8).copy()

    cleaning_report = {
        "initial_rows": int(len(working_df)),
        "null_rows_removed": int(null_mask.sum()),
        "invalid_distance_rows": int(invalid_distance_mask.sum()),
        "invalid_fare_rows": int(invalid_fare_mask.sum()),
        "invalid_trips_removed": int(invalid_trip_mask.sum()),
        "rows_removed": int(len(working_df) - len(cleaned_df)),
        "final_rows": int(len(cleaned_df)),
        "retained_percentage": round((len(cleaned_df) / max(len(working_df), 1)) * 100, 2),
    }

    return BaseAnalysis(
        raw_df=working_df,
        cleaned_df=cleaned_df,
        cleaning_report=cleaning_report,
        cleaning_breakdown=_build_cleaning_breakdown(cleaning_report),
        before_sample=before_sample,
        after_sample=after_sample,
    )


def filter_dataset(
    cleaned_df: pd.DataFrame,
    hour_range: tuple[int, int] = (0, 23),
    location: str | int = "All",
    date_range: tuple[Any, Any] | None = None,
    weekday_names: list[str] | None = None,
    day_type: str = "All",
) -> pd.DataFrame:
    start_hour, end_hour = hour_range
    filtered_df = cleaned_df.loc[cleaned_df["hour"].between(start_hour, end_hour)].copy()

    if date_range is not None:
        start_date, end_date = date_range
        filtered_df = filtered_df.loc[
            (filtered_df["pickup_date"] >= start_date)
            & (filtered_df["pickup_date"] <= end_date)
        ].copy()

    if weekday_names:
        filtered_df = filtered_df.loc[
            filtered_df["pickup_day_name"].isin(weekday_names)
        ].copy()

    if day_type in {"Weekday", "Weekend"}:
        filtered_df = filtered_df.loc[filtered_df["day_type"] == day_type].copy()

    if location != "All":
        filtered_df = filtered_df.loc[filtered_df["pulocationid"] == int(location)].copy()

    return filtered_df


def _empty_filtered_analysis() -> FilteredAnalysis:
    empty = pd.DataFrame()
    return FilteredAnalysis(
        filtered_df=empty,
        hourly_summary=empty,
        location_summary=empty,
        daily_summary=empty,
        weekday_summary=empty,
        duration_summary=empty,
        correlation_matrix=empty,
        summary_statistics=empty,
        busiest_hours=empty,
        surge_hours=empty,
        efficient_hours=empty,
        inefficient_hours=empty,
        outliers=empty,
    )


def _build_hourly_summary(filtered_df: pd.DataFrame) -> pd.DataFrame:
    hourly_summary = (
        filtered_df.groupby("hour", as_index=False)
        .agg(
            demand=("hour", "size"),
            observed_surge=("surge", "mean"),
            mean_fare=("fare_amount", "mean"),
            mean_distance=("trip_distance", "mean"),
            avg_duration=("trip_duration_minutes", "mean"),
        )
        .set_index("hour")
        .reindex(range(24))
        .reset_index()
    )

    hourly_summary["demand"] = hourly_summary["demand"].fillna(0).astype(int)
    for column in ["observed_surge", "mean_fare", "mean_distance", "avg_duration"]:
        hourly_summary[column] = hourly_summary[column].fillna(0.0)

    active_hours = hourly_summary["demand"] > 0
    hourly_summary["expected_surge"] = 0.0
    if active_hours.any():
        target_min = float(hourly_summary.loc[active_hours, "observed_surge"].min())
        target_max = float(hourly_summary.loc[active_hours, "observed_surge"].max())
        hourly_summary.loc[active_hours, "expected_surge"] = _scale_series(
            hourly_summary.loc[active_hours, "demand"].astype(float),
            target_min=target_min,
            target_max=target_max,
        )

    hourly_summary["deviation"] = (
        hourly_summary["observed_surge"] - hourly_summary["expected_surge"]
    )
    hourly_summary["deviation_zscore"] = _compute_zscore(hourly_summary["deviation"])
    hourly_summary["absolute_deviation"] = hourly_summary["deviation"].abs()
    hourly_summary["is_anomaly"] = active_hours & (
        hourly_summary["deviation_zscore"].abs() >= 1.5
    )
    return hourly_summary


def _build_location_summary(filtered_df: pd.DataFrame) -> pd.DataFrame:
    aggregation_config: dict[str, tuple[str, str]] = {
        "demand": ("pulocationid", "size"),
        "observed_surge": ("surge", "mean"),
        "mean_fare": ("fare_amount", "mean"),
        "mean_distance": ("trip_distance", "mean"),
        "avg_duration": ("trip_duration_minutes", "mean"),
    }
    if "pickup_zone" in filtered_df.columns:
        aggregation_config["pickup_zone"] = ("pickup_zone", "first")
    if "pickup_borough" in filtered_df.columns:
        aggregation_config["pickup_borough"] = ("pickup_borough", "first")

    location_summary = (
        filtered_df.groupby("pulocationid", as_index=False)
        .agg(**aggregation_config)
        .sort_values("demand", ascending=False)
    )

    if "pickup_zone" not in location_summary.columns:
        location_summary["pickup_zone"] = (
            "Location ID " + location_summary["pulocationid"].astype(str)
        )
    if "pickup_borough" not in location_summary.columns:
        location_summary["pickup_borough"] = "Unknown Borough"

    location_summary["pickup_zone_label"] = (
        location_summary["pickup_zone"]
        + ", "
        + location_summary["pickup_borough"]
        + " (ID "
        + location_summary["pulocationid"].astype(str)
        + ")"
    )

    location_summary["expected_surge"] = _scale_series(
        location_summary["demand"].astype(float),
        target_min=float(location_summary["observed_surge"].min()),
        target_max=float(location_summary["observed_surge"].max()),
    )
    location_summary["deviation"] = (
        location_summary["observed_surge"] - location_summary["expected_surge"]
    )
    location_summary["deviation_zscore"] = _compute_zscore(location_summary["deviation"])
    location_summary["absolute_deviation"] = location_summary["deviation"].abs()
    location_summary["is_anomaly"] = location_summary["deviation_zscore"].abs() >= 1.5

    return location_summary.sort_values(
        ["absolute_deviation", "demand"], ascending=[False, False]
    ).reset_index(drop=True)


def _build_daily_summary(filtered_df: pd.DataFrame) -> pd.DataFrame:
    daily_summary = (
        filtered_df.groupby("pickup_date", as_index=False)
        .agg(
            demand=("pickup_date", "size"),
            observed_surge=("surge", "mean"),
            avg_duration=("trip_duration_minutes", "mean"),
        )
        .sort_values("pickup_date")
    )
    return daily_summary


def _build_weekday_summary(filtered_df: pd.DataFrame) -> pd.DataFrame:
    weekday_summary = (
        filtered_df.groupby("pickup_day_name", as_index=False)
        .agg(
            demand=("pickup_day_name", "size"),
            observed_surge=("surge", "mean"),
            avg_duration=("trip_duration_minutes", "mean"),
        )
    )
    weekday_summary["pickup_day_name"] = pd.Categorical(
        weekday_summary["pickup_day_name"],
        categories=WEEKDAY_ORDER,
        ordered=True,
    )
    return weekday_summary.sort_values("pickup_day_name").reset_index(drop=True)


def _build_duration_summary(hourly_summary: pd.DataFrame) -> pd.DataFrame:
    return hourly_summary[["hour", "avg_duration", "demand", "observed_surge"]].copy()


def _build_correlation_matrix(hourly_summary: pd.DataFrame) -> pd.DataFrame:
    correlation_source = hourly_summary.loc[
        hourly_summary["demand"] > 0,
        [
            "demand",
            "observed_surge",
            "mean_distance",
            "mean_fare",
            "avg_duration",
            "deviation",
        ],
    ].rename(
        columns={
            "observed_surge": "surge",
            "mean_distance": "distance",
            "mean_fare": "fare",
            "avg_duration": "duration",
        }
    )

    if correlation_source.empty or len(correlation_source) < 2:
        return pd.DataFrame()

    return correlation_source.corr(numeric_only=True).round(2)


def _build_summary_statistics(filtered_df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = ["fare_amount", "trip_distance", "surge", "trip_duration_minutes"]
    if "passenger_count" in filtered_df.columns:
        numeric_columns.append("passenger_count")
    return filtered_df[numeric_columns].describe().T.round(2)


def _build_rankings(
    hourly_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active_hours = hourly_summary.loc[hourly_summary["demand"] > 0].copy()

    busiest_hours = (
        active_hours.nlargest(5, "demand")[["hour", "demand", "observed_surge"]]
        .rename(columns={"observed_surge": "avg_surge"})
        .round({"avg_surge": 2})
        .reset_index(drop=True)
    )
    surge_hours = (
        active_hours.nlargest(5, "observed_surge")[["hour", "observed_surge", "demand"]]
        .rename(columns={"observed_surge": "avg_surge"})
        .round({"avg_surge": 2})
        .reset_index(drop=True)
    )
    efficient_hours = (
        active_hours.nsmallest(5, "absolute_deviation")[
            ["hour", "demand", "observed_surge", "expected_surge", "deviation"]
        ]
        .rename(columns={"observed_surge": "avg_surge"})
        .round({"avg_surge": 2, "expected_surge": 2, "deviation": 2})
        .reset_index(drop=True)
    )
    inefficient_hours = (
        active_hours.nlargest(5, "absolute_deviation")[
            ["hour", "demand", "observed_surge", "expected_surge", "deviation"]
        ]
        .rename(columns={"observed_surge": "avg_surge"})
        .round({"avg_surge": 2, "expected_surge": 2, "deviation": 2})
        .reset_index(drop=True)
    )

    return busiest_hours, surge_hours, efficient_hours, inefficient_hours


def summarize_filtered_data(
    cleaned_df: pd.DataFrame,
    hour_range: tuple[int, int] = (0, 23),
    location: str | int = "All",
    date_range: tuple[Any, Any] | None = None,
    weekday_names: list[str] | None = None,
    day_type: str = "All",
) -> FilteredAnalysis:
    filtered_df = filter_dataset(
        cleaned_df,
        hour_range=hour_range,
        location=location,
        date_range=date_range,
        weekday_names=weekday_names,
        day_type=day_type,
    )
    if filtered_df.empty:
        return _empty_filtered_analysis()

    filtered_df = filtered_df.copy()
    filtered_df["surge_zscore"] = _compute_zscore(filtered_df["surge"])
    filtered_df["is_surge_outlier"] = filtered_df["surge_zscore"].abs() >= 3

    hourly_summary = _build_hourly_summary(filtered_df)
    location_summary = _build_location_summary(filtered_df)
    daily_summary = _build_daily_summary(filtered_df)
    weekday_summary = _build_weekday_summary(filtered_df)
    duration_summary = _build_duration_summary(hourly_summary)
    correlation_matrix = _build_correlation_matrix(hourly_summary)
    summary_statistics = _build_summary_statistics(filtered_df)
    busiest_hours, surge_hours, efficient_hours, inefficient_hours = _build_rankings(
        hourly_summary
    )
    outliers = filtered_df.loc[filtered_df["is_surge_outlier"]].copy()

    return FilteredAnalysis(
        filtered_df=filtered_df,
        hourly_summary=hourly_summary,
        location_summary=location_summary,
        daily_summary=daily_summary,
        weekday_summary=weekday_summary,
        duration_summary=duration_summary,
        correlation_matrix=correlation_matrix,
        summary_statistics=summary_statistics,
        busiest_hours=busiest_hours,
        surge_hours=surge_hours,
        efficient_hours=efficient_hours,
        inefficient_hours=inefficient_hours,
        outliers=outliers,
    )


def run_full_analysis(source: str | Path | Any) -> BaseAnalysis:
    raw_df = load_dataset(source)
    return clean_and_engineer_data(raw_df)


def plot_time_series(hourly_summary: pd.DataFrame) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax2 = ax1.twinx()

    sns.lineplot(
        data=hourly_summary,
        x="hour",
        y="demand",
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        ax=ax1,
        label="Demand",
    )
    sns.lineplot(
        data=hourly_summary,
        x="hour",
        y="observed_surge",
        marker="o",
        linewidth=2.2,
        color="#d62728",
        ax=ax2,
        label="Observed Surge",
    )
    sns.lineplot(
        data=hourly_summary,
        x="hour",
        y="expected_surge",
        linewidth=2,
        linestyle="--",
        color="#ff7f0e",
        ax=ax2,
        label="Expected Surge",
    )

    ax1.set_title("Demand vs Surge by Hour")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Trip Count")
    ax2.set_ylabel("Surge Proxy (Fare / Distance)")
    ax1.set_xticks(range(24))

    left_handles, left_labels = ax1.get_legend_handles_labels()
    right_handles, right_labels = ax2.get_legend_handles_labels()
    ax1.legend(left_handles + right_handles, left_labels + right_labels, loc="upper left")

    fig.tight_layout()
    return fig


def plot_demand_vs_surge_scatter(hourly_summary: pd.DataFrame) -> plt.Figure:
    active_hours = hourly_summary.loc[hourly_summary["demand"] > 0].copy()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.scatterplot(
        data=active_hours,
        x="demand",
        y="observed_surge",
        hue="deviation",
        palette="coolwarm",
        size="demand",
        sizes=(60, 260),
        ax=ax,
    )
    ax.set_title("Demand vs Surge Scatter")
    ax.set_xlabel("Demand (Trip Count)")
    ax.set_ylabel("Observed Surge")
    fig.tight_layout()
    return fig


def plot_surge_distribution(filtered_df: pd.DataFrame) -> plt.Figure:
    capped_surge = filtered_df["surge"].clip(upper=filtered_df["surge"].quantile(0.99))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.histplot(capped_surge, bins=30, kde=True, color="#59a14f", ax=ax)
    ax.set_title("Surge Distribution (Clipped at 99th Percentile)")
    ax.set_xlabel("Surge Proxy")
    fig.tight_layout()
    return fig


def plot_surge_box_by_hour(filtered_df: pd.DataFrame) -> plt.Figure:
    plot_df = filtered_df.copy()
    plot_df["plot_surge"] = plot_df["surge"].clip(upper=plot_df["surge"].quantile(0.99))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(data=plot_df, x="hour", y="plot_surge", color="#76b7b2", showfliers=False, ax=ax)
    ax.set_title("Surge per Hour (Core Range)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Surge Proxy")
    fig.tight_layout()
    return fig


def plot_hourly_heatmap(hourly_summary: pd.DataFrame) -> plt.Figure:
    heatmap_values = pd.DataFrame(
        [hourly_summary["observed_surge"].round(2).tolist()],
        index=["Avg Surge"],
        columns=hourly_summary["hour"].tolist(),
    )

    fig, ax = plt.subplots(figsize=(10, 2.6))
    sns.heatmap(heatmap_values, cmap="YlOrRd", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Hour vs Surge Heatmap")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.heatmap(correlation_matrix, annot=True, cmap="Blues", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    return fig


def plot_deviation(hourly_summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = np.where(hourly_summary["is_anomaly"], "#e15759", "#4e79a7")
    ax.bar(hourly_summary["hour"], hourly_summary["deviation"], color=colors)
    ax.axhline(0, color="#2f2f2f", linewidth=1)
    ax.set_title("Observed Surge - Expected Surge")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Deviation")
    ax.set_xticks(range(24))
    fig.tight_layout()
    return fig


def plot_location_deviation(location_summary: pd.DataFrame) -> plt.Figure:
    top_locations = location_summary.head(10).sort_values("deviation")

    fig, ax = plt.subplots(figsize=(8.8, 5))
    colors = np.where(top_locations["is_anomaly"], "#e15759", "#59a14f")
    labels = top_locations["pickup_zone_label"].astype(str)
    ax.barh(labels, top_locations["deviation"], color=colors)
    ax.axvline(0, color="#2f2f2f", linewidth=1)
    ax.set_title("Top Abnormal Pickup Locations")
    ax.set_xlabel("Deviation")
    ax.set_ylabel("Pickup Zone")
    fig.tight_layout()
    return fig


def plot_outlier_scatter(filtered_df: pd.DataFrame) -> plt.Figure:
    plot_df = filtered_df.copy()
    plot_df["outlier_label"] = np.where(
        plot_df["is_surge_outlier"], "Extreme Surge", "Normal"
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.scatterplot(
        data=plot_df,
        x="trip_distance",
        y="surge",
        hue="outlier_label",
        palette={"Normal": "#4e79a7", "Extreme Surge": "#e15759"},
        alpha=0.65,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("Outlier Detection View (Log Scale)")
    ax.set_xlabel("Trip Distance")
    ax.set_ylabel("Surge Proxy")
    fig.tight_layout()
    return fig


def plot_duration_by_hour(duration_summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.lineplot(
        data=duration_summary,
        x="hour",
        y="avg_duration",
        marker="o",
        linewidth=2.2,
        color="#9467bd",
        ax=ax,
    )
    ax.set_title("Average Trip Duration by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Duration (Minutes)")
    ax.set_xticks(range(24))
    fig.tight_layout()
    return fig


def plot_daily_patterns(daily_summary: pd.DataFrame) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax2 = ax1.twinx()

    sns.lineplot(
        data=daily_summary,
        x="pickup_date",
        y="demand",
        marker="o",
        color="#2f6b2f",
        linewidth=2,
        ax=ax1,
        label="Demand",
    )
    sns.lineplot(
        data=daily_summary,
        x="pickup_date",
        y="observed_surge",
        marker="o",
        color="#c44e52",
        linewidth=2,
        ax=ax2,
        label="Observed Surge",
    )

    ax1.set_title("Daily Demand and Surge Pattern")
    ax1.set_xlabel("Pickup Date")
    ax1.set_ylabel("Trip Count")
    ax2.set_ylabel("Observed Surge")
    fig.autofmt_xdate(rotation=30)

    left_handles, left_labels = ax1.get_legend_handles_labels()
    right_handles, right_labels = ax2.get_legend_handles_labels()
    ax1.legend(left_handles + right_handles, left_labels + right_labels, loc="upper left")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    default_path = Path("data.csv")
    base_analysis = run_full_analysis(default_path)
    filtered_analysis = summarize_filtered_data(base_analysis.cleaned_df)

    print("Ride Pricing Inefficiency Analysis Dashboard")
    print(f"Rows loaded: {len(base_analysis.raw_df):,}")
    print(f"Rows after cleaning: {len(base_analysis.cleaned_df):,}")
    print(f"Retained percentage: {base_analysis.cleaning_report['retained_percentage']:.2f}%")
    print(f"Outliers found: {len(filtered_analysis.outliers):,}")
