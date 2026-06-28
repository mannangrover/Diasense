"""
Enhanced Feature Extraction (Option C + Option B)
==================================================
Extracts richer features from raw sensor data beyond simple daily aggregates.

Option C features (richer patterns from raw data):
    HR:  HRV (successive differences), circadian amplitude, resting HR,
         HR recovery proxy, nocturnal HR dip
    Stress: sustained high-stress episodes, stress variability
    SpO2: desaturation events, overnight SpO2 dip
    Activity: sedentary bout count/max length, activity bout count
    Sleep: sleep efficiency, wake-after-sleep-onset

Option B features (segment-level: night/morning/afternoon/evening):
    HR split into 4 time segments with mean per segment

Combined = Option C + Option B on top of original 22.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    DATASET_PATH,
    CONTINUOUS_MODALITIES,
    STRESS_HIGH_THRESHOLD,
    SPO2_LOW_THRESHOLD,
)


# =====================================================================
# RAW DATA LOADER (returns timestamps + values, not just date + value)
# =====================================================================

def _load_raw_with_timestamps(filepath, body_key, value_key, invalid_values):
    filepath = Path(filepath)
    if not filepath.exists():
        return [], []

    with open(filepath, "r") as f:
        data = json.load(f)

    records = data["body"][body_key]
    if not records:
        return [], []

    invalid_set = set(invalid_values)
    timestamps = []
    values = []

    for r in records:
        val = r[value_key]["value"]
        if val in invalid_set:
            continue
        dt_str = r["effective_time_frame"]["date_time"].replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt_str)
        timestamps.append(dt)
        values.append(float(val))

    return timestamps, values


# =====================================================================
# OPTION C: RICHER FEATURES FROM RAW DATA
# =====================================================================

OPTION_C_FEATURE_NAMES = [
    # HR advanced (6)
    "hr_rmssd",             # HRV: root mean square of successive differences
    "hr_successive_diff_std",  # HRV: std of successive differences
    "hr_circadian_amp",     # difference between day (8am-8pm) and night (12am-6am) mean HR
    "hr_resting_est",       # estimated resting HR (5th percentile)
    "hr_nocturnal_dip_pct", # % drop from daytime to nighttime HR
    "hr_entropy",           # sample entropy of HR (regularity measure)
    # Stress advanced (3)
    "stress_sustained_high_count",  # episodes of >50 stress lasting >30 min
    "stress_iqr",                   # interquartile range (variability)
    "stress_low_pct",               # % of readings below 25 (relaxed)
    # SpO2 advanced (3)
    "spo2_desat_events",    # number of drops below 90%
    "spo2_min",             # minimum SpO2 reading
    "spo2_range",           # max - min SpO2
    # Activity advanced (4)
    "sedentary_bout_count",     # number of sedentary bouts
    "sedentary_max_bout_min",   # longest uninterrupted sedentary period
    "active_bout_count",        # number of active bouts
    "steps_per_active_min",     # step efficiency
    # Sleep advanced (3)
    "sleep_efficiency",     # sleep time / total time in bed
    "sleep_waso_min",       # wake after sleep onset (minutes)
    "sleep_fragmentation",  # awake episodes per hour of sleep
]

NUM_OPTION_C_FEATURES = len(OPTION_C_FEATURE_NAMES)  # 19


def _hr_advanced_daily(timestamps, values, date):
    """6 advanced HR features for one day."""
    if len(values) < 10:
        return [np.nan] * 6

    vals = np.array(values)

    # RMSSD: root mean square of successive differences
    diffs = np.diff(vals)
    rmssd = np.sqrt(np.mean(diffs ** 2)) if len(diffs) > 0 else np.nan

    # Std of successive differences
    diff_std = np.std(diffs) if len(diffs) > 0 else np.nan

    # Circadian amplitude: day HR - night HR
    day_vals = [v for t, v in zip(timestamps, values) if 8 <= t.hour < 20]
    night_vals = [v for t, v in zip(timestamps, values) if t.hour < 6 or t.hour >= 0 and t.hour < 6]
    night_vals_proper = [v for t, v in zip(timestamps, values) if 0 <= t.hour < 6]

    day_mean = np.mean(day_vals) if len(day_vals) >= 5 else np.nan
    night_mean = np.mean(night_vals_proper) if len(night_vals_proper) >= 5 else np.nan
    circadian_amp = (day_mean - night_mean) if not (np.isnan(day_mean) or np.isnan(night_mean)) else np.nan

    # Resting HR estimate (5th percentile)
    resting = np.percentile(vals, 5)

    # Nocturnal dip percentage
    if not np.isnan(day_mean) and not np.isnan(night_mean) and day_mean > 0:
        nocturnal_dip = (day_mean - night_mean) / day_mean * 100
    else:
        nocturnal_dip = np.nan

    # Approximate entropy (simplified - use coefficient of variation of diffs as proxy)
    if len(diffs) > 1 and np.std(vals) > 0:
        entropy = np.std(diffs) / np.mean(vals)
    else:
        entropy = np.nan

    return [rmssd, diff_std, circadian_amp, resting, nocturnal_dip, entropy]


def _stress_advanced_daily(timestamps, values, date):
    """3 advanced stress features for one day."""
    if len(values) < 5:
        return [np.nan] * 3

    vals = np.array(values)

    # Sustained high stress: count episodes where stress > 50 for > 30 consecutive minutes
    sustained_count = 0
    current_run = 0
    for i in range(len(timestamps)):
        if values[i] > STRESS_HIGH_THRESHOLD:
            if current_run == 0:
                run_start = timestamps[i]
            current_run += 1
            if i == len(timestamps) - 1 or values[i + 1] <= STRESS_HIGH_THRESHOLD if i + 1 < len(timestamps) else True:
                if current_run > 0:
                    run_duration = (timestamps[i] - run_start).total_seconds() / 60
                    if run_duration >= 30:
                        sustained_count += 1
                current_run = 0
        else:
            current_run = 0

    # IQR
    iqr = np.percentile(vals, 75) - np.percentile(vals, 25)

    # Low stress percentage
    low_pct = np.mean(vals < 25) * 100

    return [float(sustained_count), iqr, low_pct]


def _spo2_advanced_daily(timestamps, values, date):
    """3 advanced SpO2 features for one day."""
    if len(values) < 5:
        return [np.nan] * 3

    vals = np.array(values)

    # Desaturation events: readings below 90
    desat_events = int(np.sum(vals < 90))

    spo2_min = np.min(vals)
    spo2_range = np.max(vals) - np.min(vals)

    return [float(desat_events), spo2_min, spo2_range]


def _activity_advanced_daily(cohort_row, date):
    """4 advanced activity features for one day."""
    raw_path = cohort_row.get("physical_activity_filepath")
    if pd.isna(raw_path) or raw_path == "":
        return [np.nan] * 4

    filepath = Path(str(DATASET_PATH) + str(raw_path))
    if not filepath.exists():
        return [np.nan] * 4

    with open(filepath, "r") as f:
        data = json.load(f)

    records = data["body"]["activity"]
    if not records:
        return [np.nan] * 4

    sed_bouts = []
    active_bouts = []
    total_steps = 0
    total_active_min = 0

    for r in records:
        activity = r["activity_name"]
        if activity == "":
            continue
        ti = r["effective_time_frame"]["time_interval"]
        start_dt = datetime.fromisoformat(ti["start_date_time"].replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(ti["end_date_time"].replace("Z", "+00:00"))

        if start_dt.date() != date:
            continue

        dur_min = (end_dt - start_dt).total_seconds() / 60.0
        steps_val = r["base_movement_quantity"]["value"]
        steps = int(steps_val) if steps_val != "" else 0

        if activity == "sedentary":
            sed_bouts.append(dur_min)
        else:
            active_bouts.append(dur_min)
            total_steps += steps
            total_active_min += dur_min

    sed_bout_count = len(sed_bouts)
    sed_max_bout = max(sed_bouts) if sed_bouts else 0.0
    active_bout_count = len(active_bouts)
    steps_per_active = total_steps / total_active_min if total_active_min > 0 else 0.0

    return [float(sed_bout_count), sed_max_bout, float(active_bout_count), steps_per_active]


def _sleep_advanced_daily(cohort_row, date):
    """3 advanced sleep features for one day."""
    raw_path = cohort_row.get("sleep_filepath")
    if pd.isna(raw_path) or raw_path == "":
        return [np.nan] * 3

    filepath = Path(str(DATASET_PATH) + str(raw_path))
    if not filepath.exists():
        return [np.nan] * 3

    with open(filepath, "r") as f:
        data = json.load(f)

    records = data["body"]["sleep"]
    if not records:
        return [np.nan] * 3

    # Filter to this date
    sleep_min = 0
    awake_min = 0
    total_min = 0
    awake_count = 0
    found = False

    for r in records:
        ti = r["effective_time_frame"]["time_interval"]
        start_dt = datetime.fromisoformat(ti["start_date_time"].replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(ti["end_date_time"].replace("Z", "+00:00"))

        if start_dt.date() != date:
            continue

        found = True
        dur = (end_dt - start_dt).total_seconds() / 60.0
        total_min += dur

        if r["sleep_stage_state"] == "awake":
            awake_min += dur
            awake_count += 1
        else:
            sleep_min += dur

    if not found or total_min == 0:
        return [np.nan] * 3

    efficiency = sleep_min / total_min * 100 if total_min > 0 else np.nan
    waso = awake_min
    fragmentation = awake_count / (sleep_min / 60.0) if sleep_min > 0 else np.nan

    return [efficiency, waso, fragmentation]


def extract_option_c_features(cohort_row, existing_dates):
    """
    Extract Option C features for one participant.

    Args:
        cohort_row: one row from cohort DataFrame
        existing_dates: list of dates from the original feature extraction

    Returns:
        dict mapping date → list of 19 Option C features
    """
    # Load raw HR with timestamps
    hr_timestamps_by_date = defaultdict(list)
    hr_values_by_date = defaultdict(list)

    hr_cfg = CONTINUOUS_MODALITIES["heart_rate"]
    raw_path = cohort_row.get(hr_cfg["filepath_col"])
    if not pd.isna(raw_path) and raw_path != "":
        fp = str(DATASET_PATH) + str(raw_path)
        ts_list, val_list = _load_raw_with_timestamps(
            fp, hr_cfg["body_key"], hr_cfg["value_key"], hr_cfg["invalid_values"]
        )
        for t, v in zip(ts_list, val_list):
            hr_timestamps_by_date[t.date()].append(t)
            hr_values_by_date[t.date()].append(v)

    # Load raw stress with timestamps
    stress_timestamps_by_date = defaultdict(list)
    stress_values_by_date = defaultdict(list)

    stress_cfg = CONTINUOUS_MODALITIES["stress"]
    raw_path = cohort_row.get(stress_cfg["filepath_col"])
    if not pd.isna(raw_path) and raw_path != "":
        fp = str(DATASET_PATH) + str(raw_path)
        ts_list, val_list = _load_raw_with_timestamps(
            fp, stress_cfg["body_key"], stress_cfg["value_key"], stress_cfg["invalid_values"]
        )
        for t, v in zip(ts_list, val_list):
            stress_timestamps_by_date[t.date()].append(t)
            stress_values_by_date[t.date()].append(v)

    # Load raw SpO2 with timestamps
    spo2_timestamps_by_date = defaultdict(list)
    spo2_values_by_date = defaultdict(list)

    spo2_cfg = CONTINUOUS_MODALITIES["spo2"]
    raw_path = cohort_row.get(spo2_cfg["filepath_col"])
    if not pd.isna(raw_path) and raw_path != "":
        fp = str(DATASET_PATH) + str(raw_path)
        ts_list, val_list = _load_raw_with_timestamps(
            fp, spo2_cfg["body_key"], spo2_cfg["value_key"], spo2_cfg["invalid_values"]
        )
        for t, v in zip(ts_list, val_list):
            spo2_timestamps_by_date[t.date()].append(t)
            spo2_values_by_date[t.date()].append(v)

    # Cache activity and sleep files (read once, use per date)
    result = {}
    _activity_cache = {}
    _sleep_cache = {}

    for date in existing_dates:
        # HR advanced
        hr_feats = _hr_advanced_daily(
            hr_timestamps_by_date.get(date, []),
            hr_values_by_date.get(date, []),
            date,
        )

        # Stress advanced
        stress_feats = _stress_advanced_daily(
            stress_timestamps_by_date.get(date, []),
            stress_values_by_date.get(date, []),
            date,
        )

        # SpO2 advanced
        spo2_feats = _spo2_advanced_daily(
            spo2_timestamps_by_date.get(date, []),
            spo2_values_by_date.get(date, []),
            date,
        )

        # Activity advanced
        act_feats = _activity_advanced_daily(cohort_row, date)

        # Sleep advanced
        sleep_feats = _sleep_advanced_daily(cohort_row, date)

        result[date] = hr_feats + stress_feats + spo2_feats + act_feats + sleep_feats

    return result


# =====================================================================
# OPTION D: HOURLY-DERIVED DAILY FEATURES (timing + rhythm)
# =====================================================================

OPTION_D_FEATURE_NAMES = [
    "hr_peak_hour",          # hour-of-day (0-23) when HR is highest
    "hr_trough_hour",        # hour-of-day when HR is lowest
    "hr_hourly_cv",          # CV of hourly HR means (rhythm width)
    "hr_post_breakfast",     # mean HR 8-10am (post-meal response)
    "hr_post_lunch",         # mean HR 12-2pm
    "hr_post_dinner",        # mean HR 6-8pm
    "stress_peak_hour",      # hour-of-day when stress is highest
    "active_hours_count",    # distinct hours with steps > 0 (activity spread)
    "peak_active_hour",      # hour with most steps (activity timing)
    "hr_evening_vs_night",   # (9-11pm HR) minus (2-5am HR): evening arousal
]

NUM_OPTION_D_FEATURES = len(OPTION_D_FEATURE_NAMES)  # 10


def extract_option_d_features(cohort_row, existing_dates):
    """
    Extract Option D (hourly-derived daily) features for one participant.
    Reuses the same raw file reads as Option C — call after loading raw data.

    Returns:
        dict mapping date → list of 10 Option D features
    """
    # Load HR by (date, hour)
    hr_by_hour = defaultdict(lambda: defaultdict(list))
    hr_cfg = CONTINUOUS_MODALITIES["heart_rate"]
    raw_path = cohort_row.get(hr_cfg["filepath_col"])
    if not pd.isna(raw_path) and raw_path != "":
        ts_list, val_list = _load_raw_with_timestamps(
            str(DATASET_PATH) + str(raw_path),
            hr_cfg["body_key"], hr_cfg["value_key"], hr_cfg["invalid_values"]
        )
        for t, v in zip(ts_list, val_list):
            hr_by_hour[t.date()][t.hour].append(v)

    # Load Stress by (date, hour)
    stress_by_hour = defaultdict(lambda: defaultdict(list))
    stress_cfg = CONTINUOUS_MODALITIES["stress"]
    raw_path = cohort_row.get(stress_cfg["filepath_col"])
    if not pd.isna(raw_path) and raw_path != "":
        ts_list, val_list = _load_raw_with_timestamps(
            str(DATASET_PATH) + str(raw_path),
            stress_cfg["body_key"], stress_cfg["value_key"], stress_cfg["invalid_values"]
        )
        for t, v in zip(ts_list, val_list):
            stress_by_hour[t.date()][t.hour].append(v)

    # Load Activity steps by (date, hour)
    steps_by_hour = defaultdict(lambda: defaultdict(float))
    act_path = cohort_row.get("physical_activity_filepath")
    if not pd.isna(act_path) and act_path != "":
        fp = Path(str(DATASET_PATH) + str(act_path))
        if fp.exists():
            with open(fp) as f:
                data = json.load(f)
            for r in data["body"]["activity"]:
                if r["activity_name"] in ("", "sedentary"):
                    continue
                ti = r["effective_time_frame"]["time_interval"]
                start_dt = datetime.fromisoformat(
                    ti["start_date_time"].replace("Z", "+00:00")
                )
                steps_val = r["base_movement_quantity"]["value"]
                steps = float(steps_val) if steps_val != "" else 0.0
                steps_by_hour[start_dt.date()][start_dt.hour] += steps

    result = {}
    for date in existing_dates:
        hr_hours = hr_by_hour.get(date, {})
        stress_hours = stress_by_hour.get(date, {})
        steps_hours = steps_by_hour.get(date, {})

        # Hourly HR means for valid hours
        hr_hourly_means = {
            h: np.mean(vals)
            for h, vals in hr_hours.items()
            if len(vals) >= 2
        }

        if len(hr_hourly_means) >= 4:
            peak_hour = max(hr_hourly_means, key=hr_hourly_means.get)
            trough_hour = min(hr_hourly_means, key=hr_hourly_means.get)
            means_arr = np.array(list(hr_hourly_means.values()))
            cv = np.std(means_arr) / np.mean(means_arr) if np.mean(means_arr) > 0 else np.nan
        else:
            peak_hour = np.nan
            trough_hour = np.nan
            cv = np.nan

        def _window_mean(hour_dict, h_start, h_end):
            vals = [v for h in range(h_start, h_end) for v in hour_dict.get(h, [])]
            return np.mean(vals) if len(vals) >= 3 else np.nan

        post_breakfast = _window_mean(hr_hours, 8, 10)
        post_lunch = _window_mean(hr_hours, 12, 14)
        post_dinner = _window_mean(hr_hours, 18, 20)

        # Stress peak hour
        stress_hourly_means = {
            h: np.mean(vals)
            for h, vals in stress_hours.items()
            if len(vals) >= 2
        }
        stress_peak = (
            float(max(stress_hourly_means, key=stress_hourly_means.get))
            if len(stress_hourly_means) >= 3 else np.nan
        )

        # Activity spread
        active_hours = sum(1 for h, s in steps_hours.items() if s > 0)
        peak_act = (
            float(max(steps_hours, key=steps_hours.get))
            if steps_hours else np.nan
        )

        # Evening vs night HR (autonomic residual before sleep)
        evening_vals = [v for h in range(21, 23) for v in hr_hours.get(h, [])]
        night_vals = [v for h in range(2, 5) for v in hr_hours.get(h, [])]
        if len(evening_vals) >= 3 and len(night_vals) >= 3:
            evening_vs_night = np.mean(evening_vals) - np.mean(night_vals)
        else:
            evening_vs_night = np.nan

        result[date] = [
            float(peak_hour) if not np.isnan(peak_hour) else np.nan,
            float(trough_hour) if not np.isnan(trough_hour) else np.nan,
            cv,
            post_breakfast,
            post_lunch,
            post_dinner,
            stress_peak,
            float(active_hours),
            float(peak_act) if not np.isnan(peak_act) else np.nan,
            evening_vs_night,
        ]

    return result


# =====================================================================
# OPTION B: SEGMENT-LEVEL FEATURES (4 time segments per day)
# =====================================================================

OPTION_B_FEATURE_NAMES = [
    # HR by segment (4)
    "hr_night_mean",    # 12am-6am
    "hr_morning_mean",  # 6am-12pm
    "hr_afternoon_mean",  # 12pm-6pm
    "hr_evening_mean",  # 6pm-12am
    # Stress by segment (4)
    "stress_night_mean",
    "stress_morning_mean",
    "stress_afternoon_mean",
    "stress_evening_mean",
    # SpO2 by segment (2 - mainly night matters)
    "spo2_night_mean",
    "spo2_day_mean",    # 6am-12am combined
    # Activity by segment (2)
    "steps_morning",
    "steps_afternoon",
]

NUM_OPTION_B_FEATURES = len(OPTION_B_FEATURE_NAMES)  # 12


def _segment_hour(hour):
    """Map hour to segment: 0=night(0-5), 1=morning(6-11), 2=afternoon(12-17), 3=evening(18-23)."""
    if hour < 6:
        return 0
    elif hour < 12:
        return 1
    elif hour < 18:
        return 2
    else:
        return 3


def extract_option_b_features(cohort_row, existing_dates):
    """
    Extract Option B (segment-level) features for one participant.

    Returns:
        dict mapping date → list of 12 Option B features
    """
    # Load raw HR
    hr_by_date_segment = defaultdict(lambda: defaultdict(list))

    hr_cfg = CONTINUOUS_MODALITIES["heart_rate"]
    raw_path = cohort_row.get(hr_cfg["filepath_col"])
    if not pd.isna(raw_path) and raw_path != "":
        fp = str(DATASET_PATH) + str(raw_path)
        ts_list, val_list = _load_raw_with_timestamps(
            fp, hr_cfg["body_key"], hr_cfg["value_key"], hr_cfg["invalid_values"]
        )
        for t, v in zip(ts_list, val_list):
            seg = _segment_hour(t.hour)
            hr_by_date_segment[t.date()][seg].append(v)

    # Load raw stress
    stress_by_date_segment = defaultdict(lambda: defaultdict(list))

    stress_cfg = CONTINUOUS_MODALITIES["stress"]
    raw_path = cohort_row.get(stress_cfg["filepath_col"])
    if not pd.isna(raw_path) and raw_path != "":
        fp = str(DATASET_PATH) + str(raw_path)
        ts_list, val_list = _load_raw_with_timestamps(
            fp, stress_cfg["body_key"], stress_cfg["value_key"], stress_cfg["invalid_values"]
        )
        for t, v in zip(ts_list, val_list):
            seg = _segment_hour(t.hour)
            stress_by_date_segment[t.date()][seg].append(v)

    # Load raw SpO2
    spo2_by_date_segment = defaultdict(lambda: defaultdict(list))

    spo2_cfg = CONTINUOUS_MODALITIES["spo2"]
    raw_path = cohort_row.get(spo2_cfg["filepath_col"])
    if not pd.isna(raw_path) and raw_path != "":
        fp = str(DATASET_PATH) + str(raw_path)
        ts_list, val_list = _load_raw_with_timestamps(
            fp, spo2_cfg["body_key"], spo2_cfg["value_key"], spo2_cfg["invalid_values"]
        )
        for t, v in zip(ts_list, val_list):
            seg = _segment_hour(t.hour)
            spo2_by_date_segment[t.date()][seg].append(v)

    # Load activity for steps by segment
    steps_by_date_segment = defaultdict(lambda: defaultdict(int))
    raw_path = cohort_row.get("physical_activity_filepath")
    if not pd.isna(raw_path) and raw_path != "":
        fp = Path(str(DATASET_PATH) + str(raw_path))
        if fp.exists():
            with open(fp) as f:
                data = json.load(f)
            for r in data["body"]["activity"]:
                if r["activity_name"] == "" or r["activity_name"] == "sedentary":
                    continue
                ti = r["effective_time_frame"]["time_interval"]
                start_dt = datetime.fromisoformat(ti["start_date_time"].replace("Z", "+00:00"))
                steps_val = r["base_movement_quantity"]["value"]
                steps = int(steps_val) if steps_val != "" else 0
                seg = _segment_hour(start_dt.hour)
                steps_by_date_segment[start_dt.date()][seg] += steps

    result = {}
    for date in existing_dates:
        hr_segs = hr_by_date_segment.get(date, {})
        stress_segs = stress_by_date_segment.get(date, {})
        spo2_segs = spo2_by_date_segment.get(date, {})
        step_segs = steps_by_date_segment.get(date, {})

        def _seg_mean(data_dict, seg):
            vals = data_dict.get(seg, [])
            return np.mean(vals) if len(vals) >= 3 else np.nan

        feats = [
            # HR by segment
            _seg_mean(hr_segs, 0),   # night
            _seg_mean(hr_segs, 1),   # morning
            _seg_mean(hr_segs, 2),   # afternoon
            _seg_mean(hr_segs, 3),   # evening
            # Stress by segment
            _seg_mean(stress_segs, 0),
            _seg_mean(stress_segs, 1),
            _seg_mean(stress_segs, 2),
            _seg_mean(stress_segs, 3),
            # SpO2 by segment (night vs day)
            _seg_mean(spo2_segs, 0),
            np.mean(
                [v for s in [1, 2, 3] for v in spo2_segs.get(s, [])]
            ) if sum(len(spo2_segs.get(s, [])) for s in [1, 2, 3]) >= 3 else np.nan,
            # Steps by segment
            float(step_segs.get(1, 0)),  # morning
            float(step_segs.get(2, 0)),  # afternoon
        ]

        result[date] = feats

    return result
