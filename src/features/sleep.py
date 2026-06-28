"""
Sleep Feature Extraction
========================
Parse sleep JSON (interval-based stage data) into daily features:
total sleep hours, stage percentages (deep/rem/light), awake count.

Sleep data is EVENT-BASED: each record is a sleep stage interval
with start_date_time and end_date_time.

Features produced (5):
    sleep_total_hrs    — total sleep duration in hours
    sleep_deep_pct     — % of sleep time in deep stage
    sleep_rem_pct      — % of sleep time in REM stage
    sleep_light_pct    — % of sleep time in light stage
    sleep_awake_count  — number of awake episodes

Usage:
    from src.features.sleep import extract_sleep_features
    features_dict = extract_sleep_features(cohort_row)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DATASET_PATH

SLEEP_FEATURE_NAMES = [
    "sleep_total_hrs",
    "sleep_deep_pct",
    "sleep_rem_pct",
    "sleep_light_pct",
    "sleep_awake_count",
]


def extract_sleep_features(cohort_row):
    """
    Extract daily sleep features for one participant.

    Sleep stages come as intervals: {stage, start_time, end_time}.
    We assign each interval to the night it belongs to (the date of
    the start_time), compute total duration per stage, then derive
    percentages.

    Args:
        cohort_row: one row from the cohort DataFrame

    Returns:
        dict mapping date → [sleep_total_hrs, deep_pct, rem_pct, light_pct, awake_count]
    """

    raw_path = cohort_row.get("sleep_filepath")
    if pd.isna(raw_path) or raw_path == "":
        return {}

    filepath = Path(str(DATASET_PATH) + str(raw_path))
    if not filepath.exists():
        return {}

    with open(filepath, "r") as f:
        data = json.load(f)

    records = data["body"]["sleep"]
    if not records:
        return {}

    # Parse all intervals
    stages = []
    dates = []
    durations = []
    for r in records:
        ti = r["effective_time_frame"]["time_interval"]
        start_str = ti["start_date_time"].replace("Z", "+00:00")
        end_str = ti["end_date_time"].replace("Z", "+00:00")
        start_dt = datetime.fromisoformat(start_str)
        end_dt = datetime.fromisoformat(end_str)
        stages.append(r["sleep_stage_state"])
        dates.append(start_dt.date())
        durations.append((end_dt - start_dt).total_seconds() / 60.0)

    df = pd.DataFrame({"stage": stages, "date": dates, "duration_min": durations})

    # Group by date and compute features
    result = {}
    for date, group in df.groupby("date"):
        total_min = group["duration_min"].sum()
        total_hrs = total_min / 60.0

        # Stage durations (only non-awake stages count as actual sleep)
        deep_min = group.loc[group["stage"] == "deep", "duration_min"].sum()
        rem_min = group.loc[group["stage"] == "rem", "duration_min"].sum()
        light_min = group.loc[group["stage"] == "light", "duration_min"].sum()

        sleep_min = deep_min + rem_min + light_min
        if sleep_min > 0:
            deep_pct = deep_min / sleep_min * 100
            rem_pct = rem_min / sleep_min * 100
            light_pct = light_min / sleep_min * 100
        else:
            deep_pct = rem_pct = light_pct = 0.0

        awake_count = (group["stage"] == "awake").sum()

        result[date] = [total_hrs, deep_pct, rem_pct, light_pct, float(awake_count)]

    return result
