"""
CGM Feature Extraction (Dexcom G6)
===================================
Extracts daily glucose features used as auxiliary regression targets
in the multi-task LSTM. At test time these targets are not needed —
only the classification head is used.

Daily features per participant (aggregated across all valid days):
    glucose_mean       — mean glucose (mg/dL)
    time_in_range      — % readings 70-180 mg/dL (gold standard metric)
    hyperglycemia_pct  — % readings > 180 mg/dL
    glucose_cv         — coefficient of variation (std/mean * 100)
    nocturnal_mean     — mean 12am-6am glucose

'High' sentinel (>400 mg/dL) → clipped to 400.
'Low'  sentinel (<40  mg/dL) → clipped to 40.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DATASET_PATH

CGM_MANIFEST = DATASET_PATH / "wearable_blood_glucose" / "manifest.tsv"

CGM_FEATURE_NAMES = [
    "glucose_mean",
    "time_in_range",
    "hyperglycemia_pct",
    "glucose_cv",
    "nocturnal_mean",
]
NUM_CGM_FEATURES = len(CGM_FEATURE_NAMES)  # 5

# Clinical thresholds
TIR_LOW  = 70.0   # mg/dL
TIR_HIGH = 180.0  # mg/dL
HIGH_SENTINEL = 400.0
LOW_SENTINEL  = 40.0


def _parse_glucose_value(val):
    """Handle int, float, 'High', 'Low', '' sentinels."""
    if val == "High" or val == "HIGH":
        return HIGH_SENTINEL
    if val == "Low" or val == "LOW":
        return LOW_SENTINEL
    if val == "" or val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def load_cgm_manifest():
    """Load CGM manifest. Returns dict pid -> glucose_filepath."""
    df = pd.read_csv(CGM_MANIFEST, sep="\t")
    return {
        int(row["person_id"]): str(DATASET_PATH) + str(row["glucose_filepath"])
        for _, row in df.iterrows()
        if not pd.isna(row["glucose_filepath"])
    }


def extract_cgm_features(pid, cgm_path_map):
    """
    Extract 5 aggregate glucose features for one participant.

    Args:
        pid: participant ID (int)
        cgm_path_map: dict from load_cgm_manifest()

    Returns:
        np.array of shape (5,) — participant-level glucose summary
        or None if no CGM data available
    """
    fp_str = cgm_path_map.get(int(pid))
    if fp_str is None:
        return None

    fp = Path(fp_str)
    if not fp.exists():
        return None

    with open(fp, "r") as f:
        data = json.load(f)

    records = data["body"]["cgm"]
    if not records:
        return None

    timestamps = []
    values = []

    for r in records:
        val = _parse_glucose_value(r["blood_glucose"]["value"])
        if val is None:
            continue
        dt_str = (r["effective_time_frame"]["time_interval"]
                   ["start_date_time"].replace("Z", "+00:00"))
        dt = datetime.fromisoformat(dt_str)
        timestamps.append(dt)
        values.append(val)

    if len(values) < 50:
        return None

    vals = np.array(values)
    ts   = timestamps

    glucose_mean      = np.mean(vals)
    time_in_range     = np.mean((vals >= TIR_LOW) & (vals <= TIR_HIGH)) * 100
    hyperglycemia_pct = np.mean(vals > TIR_HIGH) * 100
    glucose_cv        = (np.std(vals) / np.mean(vals) * 100) if np.mean(vals) > 0 else np.nan

    nocturnal_vals = [v for t, v in zip(ts, values) if 0 <= t.hour < 6]
    nocturnal_mean = np.mean(nocturnal_vals) if len(nocturnal_vals) >= 10 else np.nan

    return np.array([
        glucose_mean,
        time_in_range,
        hyperglycemia_pct,
        glucose_cv,
        nocturnal_mean if not np.isnan(nocturnal_mean) else glucose_mean,
    ], dtype=np.float32)
