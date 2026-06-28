"""
Cohort Construction
===================
Load participants.tsv and wearable manifest.tsv,
merge, filter by sensor duration, create 4-class labels,
and return a clean cohort DataFrame.

Usage:
    from src.data.cohort import build_cohort
    cohort_df = build_cohort()
"""

import pandas as pd

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from config import (
    PARTICIPANTS_TSV,
    WEARABLE_MANIFEST_TSV,
    MIN_SENSOR_DAYS,
    LABEL_MAP_4CLASS,
    DEAD_SENSOR_PIDS,
)


def build_cohort(exclude_dead_sensors=False):
    """
    Build the filtered cohort DataFrame.

    Steps:
        1. Load participants.tsv (demographics + study group)
        2. Load wearable manifest.tsv (sensor metadata + file paths)
        3. Inner-merge on person_id (keep only participants with wearable data)
        4. Filter: keep only participants with >= MIN_SENSOR_DAYS of sensor data
        5. Optionally filter out dead-sensor participants (all HR=0)
        6. Create 4-class integer label from study_group
        7. Return clean DataFrame

    Args:
        exclude_dead_sensors: if True, runs the dead-sensor check (slow: reads
            HR JSON for every participant, ~2 min). Use True for pipeline runs,
            False for quick cohort inspection.

    Returns:
        pd.DataFrame with columns from both TSVs plus:
            - 'label': integer 0-3 (4-class)
            - All original columns preserved
    """

    # --- Step 1: Load participants ---
    # This file has: person_id, clinical_site, study_group, age,
    #                recommended_split, and boolean flags for each modality
    participants = pd.read_csv(PARTICIPANTS_TSV, sep="\t")

    # --- Step 2: Load wearable manifest ---
    # This file has: person_id, file paths for each modality,
    #                record counts, averages, sensor_sampling_duration_days
    manifest = pd.read_csv(WEARABLE_MANIFEST_TSV, sep="\t")

    # --- Step 3: Merge ---
    # Inner join: only keep participants who have wearable data
    # If a participant is in participants.tsv but NOT in the manifest,
    # they didn't wear the device — we drop them.
    merged = participants.merge(manifest, on="person_id", how="inner")

    # --- Step 4: Filter by sensor duration ---
    # We require at least MIN_SENSOR_DAYS (14) days of continuous recording.
    # This ensures every participant has enough data for a 14-day sequence.
    filtered = merged[
        merged["sensor_sampling_duration_days"] >= MIN_SENSOR_DAYS
    ].copy()

    # --- Step 5: Create 4-class label ---
    # Map study_group strings to integers using our config mapping.
    # Any study_group not in LABEL_MAP_4CLASS will become NaN — that would
    # indicate an unexpected value in the data, which we should catch.
    # --- Step 5: Optionally exclude dead-sensor participants ---
    if exclude_dead_sensors:
        filtered = filtered[~filtered["person_id"].isin(DEAD_SENSOR_PIDS)].copy()

    # --- Step 6: Create 4-class label ---
    filtered["label"] = filtered["study_group"].map(LABEL_MAP_4CLASS)

    # Safety check: no unmapped labels
    unmapped = filtered["label"].isna().sum()
    if unmapped > 0:
        bad_groups = filtered.loc[filtered["label"].isna(), "study_group"].unique()
        raise ValueError(
            f"{unmapped} participants have unmapped study_group values: {bad_groups}"
        )

    filtered["label"] = filtered["label"].astype(int)

    # Reset index for clean sequential indexing
    filtered = filtered.reset_index(drop=True)

    return filtered
