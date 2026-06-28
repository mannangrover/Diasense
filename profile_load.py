import sys, io, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from src.data.cohort import build_cohort
from src.features.continuous import extract_continuous_features
from src.features.sleep import extract_sleep_features
from src.features.activity import extract_activity_features
from config import DATASET_PATH

cohort = build_cohort(exclude_dead_sensors=True)
row = cohort.iloc[0]

t0 = time.time()
cont = extract_continuous_features(row)
t1 = time.time()
print(f"Continuous features: {t1-t0:.3f}s")

sleep = extract_sleep_features(row)
t2 = time.time()
print(f"Sleep features: {t2-t1:.3f}s")

act = extract_activity_features(row)
t3 = time.time()
print(f"Activity features: {t3-t2:.3f}s")

print(f"Total per participant: {t3-t0:.3f}s")
print(f"Estimated for 1586: {(t3-t0)*1586/60:.1f} min")
