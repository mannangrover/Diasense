"""Run the full feature extraction pipeline."""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

import numpy as np
import time
import gc

from src.data.cohort import build_cohort
from src.features.continuous import extract_continuous_features
from src.features.sleep import extract_sleep_features
from src.features.activity import extract_activity_features
from src.features.daily_summary import _merge_daily_features, _to_padded_sequence
from config import SEQUENCE_LENGTH, NUM_DAILY_FEATURES

print("Loading cohort...", flush=True)
cohort = build_cohort(exclude_dead_sensors=True)
n = len(cohort)
print(f"Processing {n} participants...", flush=True)

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
        print(f"  ERROR PID {row['person_id']}: {e}", flush=True)
        skipped.append(row["person_id"])

    if (i + 1) % 100 == 0:
        elapsed = time.time() - start
        rate = (i + 1) / elapsed
        remaining = (n - i - 1) / rate / 60
        print(f"  {i+1}/{n} ({elapsed:.0f}s, ~{remaining:.1f}min left)", flush=True)
        gc.collect()

# Remove skipped
if skipped:
    mask = ~np.isin(person_ids, skipped) & (person_ids > 0)
    X = X[mask]
    y = y[mask]
    lengths = lengths[mask]
    person_ids = person_ids[mask]
    print(f"Skipped {len(skipped)} participants", flush=True)

elapsed = time.time() - start
print(f"\nDone! {len(X)} participants in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
print(f"X shape: {X.shape}", flush=True)
print(f"Lengths: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}", flush=True)

# Save
np.save("outputs/features/X.npy", X)
np.save("outputs/features/y.npy", y)
np.save("outputs/features/lengths.npy", lengths)
np.save("outputs/features/person_ids.npy", person_ids)
print("Saved!", flush=True)
