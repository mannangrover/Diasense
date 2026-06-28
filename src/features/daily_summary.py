"""
Daily Summary Assembly
======================
Combine features from all modalities (continuous + event-based)
into final sequences for LSTM and flat features for baseline.

Output shapes:
    LSTM:     (n_participants, 14, 22) — padded sequences with masking
    Baseline: (n_participants, 22) — mean of daily features across days

Handles missing modalities, day alignment, padding, and NaN management.

Usage:
    from src.features.daily_summary import build_all_features
    X, y, lengths, pids = build_all_features()
"""

import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    DATASET_PATH,
    SEQUENCE_LENGTH,
    NUM_DAILY_FEATURES,
    DAILY_FEATURE_NAMES,
)
from src.data.cohort import build_cohort
from src.features.continuous import extract_continuous_features, CONTINUOUS_FEATURE_NAMES
from src.features.sleep import extract_sleep_features, SLEEP_FEATURE_NAMES
from src.features.activity import extract_activity_features, ACTIVITY_FEATURE_NAMES


def _merge_daily_features(continuous, sleep, activity):
    """
    Merge daily features from all three sources into a single dict.

    For each date, concatenate:
        [13 continuous features] + [5 sleep features] + [4 activity features] = 22

    Missing modalities get NaN for their slots.

    Returns:
        dict mapping date → np.array of 22 features
    """

    all_dates = set()
    if continuous:
        all_dates.update(continuous.keys())
    if sleep:
        all_dates.update(sleep.keys())
    if activity:
        all_dates.update(activity.keys())

    if not all_dates:
        return {}

    result = {}
    for date in sorted(all_dates):
        cont_feats = continuous.get(date, [np.nan] * 13) if continuous else [np.nan] * 13
        sleep_feats = sleep.get(date, [np.nan] * 5) if sleep else [np.nan] * 5
        act_feats = activity.get(date, [np.nan] * 4) if activity else [np.nan] * 4

        combined = list(cont_feats) + list(sleep_feats) + list(act_feats)
        assert len(combined) == NUM_DAILY_FEATURES, f"Expected {NUM_DAILY_FEATURES}, got {len(combined)}"
        result[date] = np.array(combined, dtype=np.float32)

    return result


def _to_padded_sequence(daily_features, seq_len):
    """
    Convert a dict of {date: features} into a padded sequence.

    Takes the first `seq_len` days. If fewer days available, pads
    remaining timesteps with zeros.

    Returns:
        sequence: np.array of shape (seq_len, n_features)
        actual_length: int — how many real days (for masking)
    """

    dates = sorted(daily_features.keys())
    n_features = NUM_DAILY_FEATURES

    # Take first seq_len days
    dates = dates[:seq_len]
    actual_length = len(dates)

    sequence = np.zeros((seq_len, n_features), dtype=np.float32)
    for i, date in enumerate(dates):
        sequence[i] = daily_features[date]

    return sequence, actual_length


def build_all_features(save=True, verbose=True):
    """
    Extract all 22 daily features for all participants and assemble sequences.

    Returns:
        X: np.array of shape (n_participants, 14, 22) — padded sequences
        y: np.array of shape (n_participants,) — integer labels (0-3)
        lengths: np.array of shape (n_participants,) — actual sequence lengths
        person_ids: np.array of shape (n_participants,) — participant IDs
    """

    cohort = build_cohort(exclude_dead_sensors=True)
    n = len(cohort)

    if verbose:
        print(f"Processing {n} participants...")

    X = np.zeros((n, SEQUENCE_LENGTH, NUM_DAILY_FEATURES), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    person_ids = np.zeros(n, dtype=np.int64)

    skipped = []
    start = time.time()

    for i, (_, row) in enumerate(cohort.iterrows()):
        try:
            continuous = extract_continuous_features(row)
            sleep = extract_sleep_features(row)
            activity = extract_activity_features(row)

            daily = _merge_daily_features(continuous, sleep, activity)

            if not daily:
                skipped.append(row["person_id"])
                continue

            seq, length = _to_padded_sequence(daily, SEQUENCE_LENGTH)
            X[i] = seq
            y[i] = row["label"]
            lengths[i] = length
            person_ids[i] = row["person_id"]

        except Exception as e:
            if verbose:
                print(f"  ERROR PID {row['person_id']}: {e}")
            skipped.append(row["person_id"])

        if verbose and (i + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (n - i - 1) / rate / 60
            print(f"  {i+1}/{n} ({elapsed:.0f}s elapsed, ~{remaining:.1f}min remaining)")

    # Remove skipped participants (rows that stayed all zeros)
    if skipped:
        mask = np.isin(person_ids, skipped, invert=True) & (person_ids > 0)
        X = X[mask]
        y = y[mask]
        lengths = lengths[mask]
        person_ids = person_ids[mask]
        if verbose:
            print(f"  Skipped {len(skipped)} participants with no valid data")

    if verbose:
        elapsed = time.time() - start
        print(f"\nDone! {len(X)} participants in {elapsed:.0f}s ({elapsed/60:.1f}min)")
        print(f"  X shape: {X.shape}")
        print(f"  Sequence lengths: min={lengths.min()}, max={lengths.max()}, "
              f"mean={lengths.mean():.1f}")

    if save:
        out_dir = Path(__file__).resolve().parents[2] / "outputs" / "features"
        np.save(out_dir / "X.npy", X)
        np.save(out_dir / "y.npy", y)
        np.save(out_dir / "lengths.npy", lengths)
        np.save(out_dir / "person_ids.npy", person_ids)
        if verbose:
            print(f"  Saved to {out_dir}/")

    return X, y, lengths, person_ids
