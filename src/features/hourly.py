"""
Hourly Feature Extraction
=========================
Extracts 10 features per hour × 24 hours × 14 days = 336 timesteps per participant.

Features per hour:
    HR:       mean, std, min, max          (4)
    Stress:   mean, high_pct               (2)
    SpO2:     mean, min                    (2)
    Activity: steps, active_min            (2)
    Total: 10 features/hour

Output shape: (n_participants, 336, 10)
Sequence length: actual_days × 24 (rest is zero-padded)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DATASET_PATH, CONTINUOUS_MODALITIES, STRESS_HIGH_THRESHOLD

HOURLY_FEATURE_NAMES = [
    "hr_mean", "hr_std", "hr_min", "hr_max",        # 4
    "stress_mean", "stress_high_pct",                # 2
    "spo2_mean", "spo2_min",                         # 2
    "steps", "active_min",                           # 2
]
NUM_HOURLY_FEATURES = len(HOURLY_FEATURE_NAMES)  # 10

SEQ_DAYS = 14
SEQ_HOURS = SEQ_DAYS * 24  # 336


def _load_raw_by_hour(filepath, body_key, value_key, invalid_values):
    """Load raw sensor file, group readings by (date, hour)."""
    fp = Path(filepath)
    if not fp.exists():
        return {}

    with open(fp, "r") as f:
        data = json.load(f)

    records = data["body"][body_key]
    if not records:
        return {}

    invalid_set = set(invalid_values)
    by_hour = defaultdict(list)

    for r in records:
        val = r[value_key]["value"]
        if val in invalid_set:
            continue
        dt_str = r["effective_time_frame"]["date_time"].replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt_str)
        by_hour[(dt.date(), dt.hour)].append(float(val))

    return dict(by_hour)


def _load_activity_by_hour(filepath):
    """Load activity records, allocate steps + active_min to start hour."""
    fp = Path(filepath)
    if not fp.exists():
        return {}, {}

    with open(fp, "r") as f:
        data = json.load(f)

    records = data["body"]["activity"]
    if not records:
        return {}, {}

    steps_by_hour = defaultdict(float)
    active_min_by_hour = defaultdict(float)

    for r in records:
        name = r["activity_name"]
        if name == "" or name == "sedentary":
            continue
        ti = r["effective_time_frame"]["time_interval"]
        start_dt = datetime.fromisoformat(ti["start_date_time"].replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(ti["end_date_time"].replace("Z", "+00:00"))
        dur_min = (end_dt - start_dt).total_seconds() / 60.0
        steps_val = r["base_movement_quantity"]["value"]
        steps = float(steps_val) if steps_val != "" else 0.0
        key = (start_dt.date(), start_dt.hour)
        steps_by_hour[key] += steps
        active_min_by_hour[key] += dur_min

    return dict(steps_by_hour), dict(active_min_by_hour)


def extract_hourly_features(cohort_row, existing_dates):
    """
    Extract hourly features for one participant.

    Args:
        cohort_row: one row from cohort DataFrame
        existing_dates: sorted list of date objects (up to 14 days)

    Returns:
        X: np.array of shape (SEQ_HOURS, NUM_HOURLY_FEATURES) — zero-padded
        actual_length: int — number of valid hourly timesteps (actual_days × 24)
    """
    # Load HR
    hr_cfg = CONTINUOUS_MODALITIES["heart_rate"]
    hr_by_hour = {}
    raw_path = cohort_row.get(hr_cfg["filepath_col"])
    if not pd.isna(raw_path) and raw_path != "":
        hr_by_hour = _load_raw_by_hour(
            str(DATASET_PATH) + str(raw_path),
            hr_cfg["body_key"], hr_cfg["value_key"], hr_cfg["invalid_values"]
        )

    # Load Stress
    stress_cfg = CONTINUOUS_MODALITIES["stress"]
    stress_by_hour = {}
    raw_path = cohort_row.get(stress_cfg["filepath_col"])
    if not pd.isna(raw_path) and raw_path != "":
        stress_by_hour = _load_raw_by_hour(
            str(DATASET_PATH) + str(raw_path),
            stress_cfg["body_key"], stress_cfg["value_key"], stress_cfg["invalid_values"]
        )

    # Load SpO2
    spo2_cfg = CONTINUOUS_MODALITIES["spo2"]
    spo2_by_hour = {}
    raw_path = cohort_row.get(spo2_cfg["filepath_col"])
    if not pd.isna(raw_path) and raw_path != "":
        spo2_by_hour = _load_raw_by_hour(
            str(DATASET_PATH) + str(raw_path),
            spo2_cfg["body_key"], spo2_cfg["value_key"], spo2_cfg["invalid_values"]
        )

    # Load Activity
    steps_by_hour = {}
    active_min_by_hour = {}
    act_path = cohort_row.get("physical_activity_filepath")
    if not pd.isna(act_path) and act_path != "":
        steps_by_hour, active_min_by_hour = _load_activity_by_hour(
            str(DATASET_PATH) + str(act_path)
        )

    # Limit to first SEQ_DAYS days
    dates = sorted(existing_dates)[:SEQ_DAYS]
    n_days = len(dates)
    actual_length = n_days * 24

    X = np.full((SEQ_HOURS, NUM_HOURLY_FEATURES), np.nan, dtype=np.float32)

    for day_idx, date in enumerate(dates):
        for hour in range(24):
            step_idx = day_idx * 24 + hour
            key = (date, hour)
            feats = np.full(NUM_HOURLY_FEATURES, np.nan, dtype=np.float32)

            # HR features
            hr_vals = hr_by_hour.get(key, [])
            if len(hr_vals) >= 2:
                arr = np.array(hr_vals)
                feats[0] = np.mean(arr)
                feats[1] = np.std(arr)
                feats[2] = np.min(arr)
                feats[3] = np.max(arr)
            elif len(hr_vals) == 1:
                feats[0] = hr_vals[0]
                feats[2] = hr_vals[0]
                feats[3] = hr_vals[0]

            # Stress features
            stress_vals = stress_by_hour.get(key, [])
            if len(stress_vals) >= 1:
                arr = np.array(stress_vals)
                feats[4] = np.mean(arr)
                feats[5] = np.mean(arr > STRESS_HIGH_THRESHOLD) * 100

            # SpO2 features
            spo2_vals = spo2_by_hour.get(key, [])
            if len(spo2_vals) >= 1:
                arr = np.array(spo2_vals)
                feats[6] = np.mean(arr)
                feats[7] = np.min(arr)

            # Activity features
            feats[8] = float(steps_by_hour.get(key, 0.0))
            feats[9] = float(active_min_by_hour.get(key, 0.0))

            X[step_idx] = feats

    # Zero-pad the NaN slots that correspond to padding (beyond actual_length)
    # Keep NaN within real timesteps — imputation handles those later
    X[actual_length:] = 0.0

    return X, actual_length
