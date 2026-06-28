"""
Data Quality
============
Check file availability and data quality across the cohort.

Usage:
    from src.data.quality import check_file_availability
    availability_df = check_file_availability(cohort_df)
"""

import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from config import (
    DATASET_PATH,
    CONTINUOUS_MODALITIES,
    EVENT_MODALITIES,
    CALORIE_CONFIG,
)


# All modality filepath columns we need to check
_ALL_FILEPATH_COLS = {
    "heart_rate": "heartrate_filepath",
    "stress": "stress_level_filepath",
    "respiratory_rate": "respiratory_rate_filepath",
    "spo2": "oxygen_saturation_filepath",
    "sleep": "sleep_filepath",
    "activity": "physical_activity_filepath",
    "calories": "active_calories_filepath",
}


def find_dead_sensor_pids(cohort_df):
    """
    Identify participants whose HR data is entirely zeros (sensor never worked).

    These participants also have all-invalid stress/resp values — essentially
    no usable wearable data despite having JSON files on disk.

    Args:
        cohort_df: output of build_cohort()

    Returns:
        list of person_id values to exclude
    """
    import json

    dead = []
    for _, row in cohort_df.iterrows():
        filepath = str(DATASET_PATH) + row["heartrate_filepath"]
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            records = data["body"]["heart_rate"]
            if all(r["heart_rate"]["value"] == 0 for r in records):
                dead.append(row["person_id"])
        except (FileNotFoundError, KeyError):
            dead.append(row["person_id"])

    return dead


def check_file_availability(cohort_df):
    """
    For each participant, check which modality JSON files actually exist on disk.

    The manifest gives us relative file paths, but:
    - Some paths might be NaN (modality not recorded for that participant)
    - Some paths might point to files that don't exist
    - Some filenames might not match exactly (e.g., naming conventions differ)

    Returns:
        pd.DataFrame with columns:
            - person_id
            - One boolean column per modality: True = file exists, False = missing
            - 'available_count': how many of 7 modalities are available
    """

    results = []

    for _, row in cohort_df.iterrows():
        person = {"person_id": row["person_id"]}

        for modality_name, filepath_col in _ALL_FILEPATH_COLS.items():

            # Check if the manifest even has a path for this modality
            raw_path = row.get(filepath_col)

            if pd.isna(raw_path) or raw_path == "":
                person[modality_name] = False
                continue

            # Build the full path from dataset root + relative path
            full_path = Path(str(DATASET_PATH) + str(raw_path))

            # Check if the file actually exists on disk
            person[modality_name] = full_path.exists()

        results.append(person)

    availability = pd.DataFrame(results)

    # Count how many modalities each participant has
    modality_cols = list(_ALL_FILEPATH_COLS.keys())
    availability["available_count"] = availability[modality_cols].sum(axis=1)

    return availability


def summarize_availability(availability_df):
    """
    Print a summary of modality availability across the cohort.

    Args:
        availability_df: output of check_file_availability()
    """

    modality_cols = list(_ALL_FILEPATH_COLS.keys())
    n_total = len(availability_df)

    print("=" * 55)
    print("MODALITY AVAILABILITY SUMMARY")
    print("=" * 55)
    print(f"\nTotal participants: {n_total}\n")

    print(f"{'Modality':<20} {'Available':>10} {'Missing':>10} {'%':>8}")
    print("-" * 55)

    for col in modality_cols:
        available = availability_df[col].sum()
        missing = n_total - available
        pct = available / n_total * 100
        print(f"{col:<20} {available:>10} {missing:>10} {pct:>7.1f}%")

    print("-" * 55)

    # How many participants have all 7 modalities?
    all_available = (availability_df["available_count"] == 7).sum()
    print(f"\nParticipants with ALL 7 modalities: {all_available} ({all_available/n_total*100:.1f}%)")

    # Distribution of available modality counts
    print(f"\nAvailable modality count distribution:")
    print(availability_df["available_count"].value_counts().sort_index().to_string())
