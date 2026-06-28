"""
Continuous Modality Features
============================
Extract per-day features from continuous modalities:
Heart Rate, Stress, Respiratory Rate, SpO2.

These modalities have point-in-time measurements at ~1-3 minute intervals.

Features produced (13 total):
    HR:    hr_mean, hr_std, hr_min, hr_max, hr_range     (5)
    Stress: stress_mean, stress_std, stress_high_pct      (3)
    Resp:  resp_mean, resp_std                            (2)
    SpO2:  spo2_mean, spo2_std, spo2_below95_pct          (3)

Usage:
    from src.features.continuous import extract_continuous_features
    features_dict = extract_continuous_features(person_id, cohort_row)
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    DATASET_PATH,
    CONTINUOUS_MODALITIES,
    STRESS_HIGH_THRESHOLD,
    SPO2_LOW_THRESHOLD,
)
from src.data.loaders import load_continuous_modality


def _hr_daily_features(day_values):
    """5 features from heart rate readings for one day."""
    if len(day_values) == 0:
        return [np.nan] * 5
    return [
        np.mean(day_values),
        np.std(day_values),
        np.min(day_values),
        np.max(day_values),
        np.max(day_values) - np.min(day_values),
    ]


def _stress_daily_features(day_values):
    """3 features from stress readings for one day."""
    if len(day_values) == 0:
        return [np.nan] * 3
    return [
        np.mean(day_values),
        np.std(day_values),
        np.mean(day_values > STRESS_HIGH_THRESHOLD) * 100,
    ]


def _resp_daily_features(day_values):
    """2 features from respiratory rate readings for one day."""
    if len(day_values) == 0:
        return [np.nan] * 2
    return [
        np.mean(day_values),
        np.std(day_values),
    ]


def _spo2_daily_features(day_values):
    """3 features from SpO2 readings for one day."""
    if len(day_values) == 0:
        return [np.nan] * 3
    return [
        np.mean(day_values),
        np.std(day_values),
        np.mean(day_values < SPO2_LOW_THRESHOLD) * 100,
    ]


# Maps modality name to (feature_func, num_features, feature_name_prefix)
_FEATURE_EXTRACTORS = {
    "heart_rate":       (_hr_daily_features,     5, "hr"),
    "stress":           (_stress_daily_features,  3, "stress"),
    "respiratory_rate": (_resp_daily_features,    2, "resp"),
    "spo2":             (_spo2_daily_features,    3, "spo2"),
}

# Feature names in order (must match DAILY_FEATURE_NAMES[:13] in config)
CONTINUOUS_FEATURE_NAMES = [
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range",
    "stress_mean", "stress_std", "stress_high_pct",
    "resp_mean", "resp_std",
    "spo2_mean", "spo2_std", "spo2_below95_pct",
]


def extract_continuous_features(cohort_row):
    """
    Extract daily continuous features for one participant.

    For each of the 4 continuous modalities:
        1. Load the JSON file
        2. Group valid readings by date
        3. Compute daily statistics

    Args:
        cohort_row: one row from the cohort DataFrame (has filepath columns)

    Returns:
        dict mapping date → list of 13 feature values
        Dates with no data for a modality get NaN for those features.
    """

    # Collect per-modality daily data: {date: {modality: np.array of values}}
    all_dates = set()
    modality_data = {}

    for mod_name, mod_cfg in CONTINUOUS_MODALITIES.items():
        raw_path = cohort_row.get(mod_cfg["filepath_col"])

        if pd.isna(raw_path) or raw_path == "":
            modality_data[mod_name] = {}
            continue

        filepath = str(DATASET_PATH) + str(raw_path)

        df = load_continuous_modality(
            filepath,
            mod_cfg["body_key"],
            mod_cfg["value_key"],
            mod_cfg["invalid_values"],
        )

        if df.empty:
            modality_data[mod_name] = {}
            continue

        # Group by date
        daily = {}
        for date, group in df.groupby("date"):
            daily[date] = group["value"].values
            all_dates.add(date)

        modality_data[mod_name] = daily

    if not all_dates:
        return {}

    # For each date, compute all 13 features
    result = {}
    for date in sorted(all_dates):
        features = []
        for mod_name in ["heart_rate", "stress", "respiratory_rate", "spo2"]:
            extractor, n_features, _ = _FEATURE_EXTRACTORS[mod_name]
            day_values = modality_data[mod_name].get(date, np.array([]))
            features.extend(extractor(day_values))
        result[date] = features

    return result
