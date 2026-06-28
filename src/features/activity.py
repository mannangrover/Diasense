"""
Activity & Calorie Feature Extraction
======================================
Parse activity JSON (interval-based segments) into daily features:
total steps, sedentary minutes, active minutes.

Parse calorie JSON (point measurements) into daily totals.

Features produced (4):
    steps_total        — total steps per day
    sedentary_minutes  — total minutes in sedentary activity
    active_minutes     — total minutes in non-sedentary activity (walking+running+generic)
    calories_total     — total active calories burned

Usage:
    from src.features.activity import extract_activity_features
    features_dict = extract_activity_features(cohort_row)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DATASET_PATH

ACTIVITY_FEATURE_NAMES = [
    "steps_total",
    "sedentary_minutes",
    "active_minutes",
    "calories_total",
]


def _load_activity_data(cohort_row):
    """Load and parse activity segment intervals."""

    raw_path = cohort_row.get("physical_activity_filepath")
    if pd.isna(raw_path) or raw_path == "":
        return pd.DataFrame()

    filepath = Path(str(DATASET_PATH) + str(raw_path))
    if not filepath.exists():
        return pd.DataFrame()

    with open(filepath, "r") as f:
        data = json.load(f)

    records = data["body"]["activity"]
    if not records:
        return pd.DataFrame()

    activities = []
    dates = []
    durations = []
    steps_list = []

    for r in records:
        activity = r["activity_name"]
        if activity == "":
            continue

        ti = r["effective_time_frame"]["time_interval"]
        start_str = ti["start_date_time"].replace("Z", "+00:00")
        end_str = ti["end_date_time"].replace("Z", "+00:00")
        start_dt = datetime.fromisoformat(start_str)
        end_dt = datetime.fromisoformat(end_str)

        activities.append(activity)
        dates.append(start_dt.date())
        durations.append((end_dt - start_dt).total_seconds() / 60.0)

        steps_val = r["base_movement_quantity"]["value"]
        steps_list.append(int(steps_val) if steps_val != "" else 0)

    if not activities:
        return pd.DataFrame()

    return pd.DataFrame({
        "activity": activities,
        "date": dates,
        "duration_min": durations,
        "steps": steps_list,
    })


def _load_calorie_data(cohort_row):
    """Load and parse calorie point measurements."""

    raw_path = cohort_row.get("active_calories_filepath")
    if pd.isna(raw_path) or raw_path == "":
        return pd.DataFrame()

    filepath = Path(str(DATASET_PATH) + str(raw_path))
    if not filepath.exists():
        return pd.DataFrame()

    with open(filepath, "r") as f:
        data = json.load(f)

    records = data["body"]["activity"]
    if not records:
        return pd.DataFrame()

    cal_dates = []
    cal_values = []
    for r in records:
        dt_str = r["effective_time_frame"]["date_time"].replace("Z", "+00:00")
        cal_dates.append(datetime.fromisoformat(dt_str).date())
        cal_values.append(float(r["calories_value"]["value"]))

    if not cal_dates:
        return pd.DataFrame()

    return pd.DataFrame({"date": cal_dates, "calories": cal_values})


def extract_activity_features(cohort_row):
    """
    Extract daily activity + calorie features for one participant.

    Activity segments are intervals classified as sedentary/walking/running/generic.
    Calories are point measurements summed per day.

    Args:
        cohort_row: one row from the cohort DataFrame

    Returns:
        dict mapping date → [steps_total, sedentary_minutes, active_minutes, calories_total]
    """

    activity_df = _load_activity_data(cohort_row)
    calorie_df = _load_calorie_data(cohort_row)

    # Collect all dates from both sources
    all_dates = set()
    activity_daily = {}
    calorie_daily = {}

    if not activity_df.empty:
        for date, group in activity_df.groupby("date"):
            all_dates.add(date)
            steps = group["steps"].sum()
            sed_min = group.loc[group["activity"] == "sedentary", "duration_min"].sum()
            active_min = group.loc[group["activity"] != "sedentary", "duration_min"].sum()
            activity_daily[date] = (steps, sed_min, active_min)

    if not calorie_df.empty:
        for date, group in calorie_df.groupby("date"):
            all_dates.add(date)
            calorie_daily[date] = group["calories"].max()

    if not all_dates:
        return {}

    result = {}
    for date in sorted(all_dates):
        steps, sed, active = activity_daily.get(date, (0, 0.0, 0.0))
        cals = calorie_daily.get(date, 0.0)
        result[date] = [float(steps), sed, active, cals]

    return result
