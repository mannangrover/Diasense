"""
Data Loaders
=============
Load raw JSON files for each modality and return clean DataFrames.

Usage:
    from src.data.loaders import load_continuous_modality
    df = load_continuous_modality(filepath, body_key, value_key, invalid_values)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_continuous_modality(filepath, body_key, value_key, invalid_values):
    """
    Load a continuous modality JSON file and return a clean DataFrame.

    Each JSON has structure:
        {"body": {body_key: [{"value_key": {"value": X}, "effective_time_frame": {"date_time": "..."}}]}}

    Args:
        filepath: full path to the JSON file
        body_key: key in data["body"] that holds the records list
        value_key: key in each record that holds the value dict
        invalid_values: list of sentinel values to filter out (e.g. [0], [-2, -1])

    Returns:
        pd.DataFrame with columns:
            - datetime: pandas Timestamp (UTC)
            - value: float (only valid readings)
            - date: date object (for grouping by day)
        Returns empty DataFrame if file doesn't exist or has no valid data.
    """

    filepath = Path(filepath)
    if not filepath.exists():
        return pd.DataFrame(columns=["datetime", "value", "date"])

    with open(filepath, "r") as f:
        data = json.load(f)

    records = data["body"][body_key]

    if not records:
        return pd.DataFrame(columns=["datetime", "value", "date"])

    dates = []
    values = []
    invalid_set = set(invalid_values)

    for r in records:
        val = r[value_key]["value"]
        if val in invalid_set:
            continue
        dt_str = r["effective_time_frame"]["date_time"].replace("Z", "+00:00")
        dates.append(datetime.fromisoformat(dt_str).date())
        values.append(float(val))

    if not values:
        return pd.DataFrame(columns=["datetime", "value", "date"])

    return pd.DataFrame({"date": dates, "value": values})
