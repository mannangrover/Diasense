# Diasense Project Progress Journal

## Project: Type 2 Diabetes Prediction from Wearable Sensor Data
## Started: 2026-06-22 (Fresh Restart)
## Student: Mannan Grover (B.Tech CSE)
## Mentor: Claude (Senior DL Engineer)

---

## Session 1 — 2026-06-22: Architecture Decisions & Implementation Plan

### Status: ALL DECISIONS FINALIZED

### Decisions Made (all confirmed by student):
- [x] Problem Framing: 4-class with hierarchical evaluation (4/3/2 levels)
- [x] Feature Strategy: Option D — Hybrid (engineered features + LSTM)
- [x] Data Processing: Two-track (continuous modalities + event-based modalities)
- [x] Model: Daily-summary LSTM (14 × 22) as main model + RF baseline for comparison
- [x] Evaluation: ROC-AUC (multi-class OvR macro) + PR-AUC, Sensitivity, Specificity
- [x] Validation: GroupKFold (5-fold) for dev + held-out recommended_split for final

### Key Discussions:
1. Student correctly identified sleep/activity as event-based (not minute-by-minute)
2. Student wanted to skip XGBoost — agreed to quick RF baseline on new features instead
3. LSTM redesigned: daily summaries (14 days × 22 features) instead of raw 60-min windows

### Documents Created:
- PROGRESS.md (this file)
- IMPLEMENTATION_PLAN.md (complete decision history + technical design + roadmap)

---

## Session 2 — 2026-06-22: Step 1 — Project Setup (COMPLETED)

### What was done:
- [x] Created folder structure: src/data, src/features, src/models, src/evaluation
- [x] Created output directories: outputs/features, outputs/models, outputs/figures
- [x] Created config.py — central configuration with ALL paths, constants, feature names
- [x] Created __init__.py for all packages (src, data, features, models, evaluation)
- [x] Created placeholder modules with docstrings (12 .py files)
- [x] Validated: all imports work, all paths resolve, config is correct

### Files Created:
```
config.py                        — Central config (paths, labels, features, model params)
src/__init__.py                  — Package init
src/data/__init__.py             — Package init
src/data/cohort.py               — Placeholder (Step 2)
src/data/loaders.py              — Placeholder (Step 3)
src/data/quality.py              — Placeholder (Step 2)
src/features/__init__.py         — Package init
src/features/continuous.py       — Placeholder (Step 3)
src/features/sleep.py            — Placeholder (Step 4)
src/features/activity.py         — Placeholder (Step 4)
src/features/daily_summary.py    — Placeholder (Step 5)
src/models/__init__.py           — Package init
src/models/baseline.py           — Placeholder (Step 6)
src/models/lstm.py               — Placeholder (Step 7)
src/models/trainer.py            — Placeholder (Step 7)
src/evaluation/__init__.py       — Package init
src/evaluation/metrics.py        — Placeholder (Step 6-7)
outputs/features/                — Empty (will hold feature arrays)
outputs/models/                  — Empty (will hold .pt checkpoints)
outputs/figures/                 — Empty (will hold plots)
```

### Key Design Decisions:
- config.py is the single source of truth for all paths and constants
- set_seed() function handles reproducibility across numpy, random, and torch
- STRESS_HIGH_THRESHOLD left as None — to be set from data during Phase 0
- Old notebooks/*.npy files left untouched (reference from previous work)

### Concepts Taught:
- Single Source of Truth: one place for all paths/constants, everything else imports from it
- Python packages: __init__.py makes a directory importable as a module

### Implementation Roadmap:
1. [x] Step 1: Project setup (folders, config.py) ✅ DONE
2. [ ] Step 2: Phase 0 — Data Audit (notebook 01)
3. [ ] Step 3: Phase 1A — Continuous feature extraction (HR, Stress, Resp, SpO2)
4. [ ] Step 4: Phase 1B — Event-based feature extraction (Sleep, Activity, Calories)
5. [ ] Step 5: Phase 1C — Daily summary assembly → (1655, 14, 22) sequences
6. [ ] Step 6: Phase 2 — RF/LR baseline sanity check
7. [ ] Step 7: Phase 3 — LSTM training & evaluation
8. [ ] Step 8: Phase 4 — Analysis & interpretation

### Next: Step 2 — Phase 0 Data Audit

---

## Session 3 — 2026-06-26: Step 2 — Phase 0 Data Audit (COMPLETED)

### What was done:
- [x] Implemented `src/data/cohort.py` — loads, merges, filters, labels the cohort
- [x] Implemented `src/data/quality.py` — checks file availability across all 1655 participants
- [x] Created `notebooks/01_data_audit.ipynb` — interactive audit notebook with visualizations
- [x] Updated `config.py` — fixed stress invalid values, set STRESS_HIGH_THRESHOLD

### Key Findings:

**Cohort:**
- 1,655 participants (580 healthy, 427 prediabetes, 471 oral_med, 177 insulin)
- Insulin is smallest class at 10.7%

**Dataset Split (surprise!):**
- Dataset provides a 3-way split: train (1156), val (248), test (251)
- Previously assumed 2-way — this is better for evaluation

**Modality Availability:**
- Heart Rate, Sleep, Activity, Calories: 100% available
- Respiratory Rate: 99.6% (6 missing)
- Stress: 98.7% (22 missing)
- SpO2: 79.2% (344 missing) — weakest modality
- SpO2 missingness is uniform across classes (no bias concern)
- 77.7% of participants have all 7 modalities

**Critical Bug Found:**
- Stress values: -2 AND -1 are both invalid (not just -2 as originally assumed)
- -1 values account for 44% of all stress readings!
- Updated config.py to filter both sentinels

**Config Updates:**
- `STRESS_HIGH_THRESHOLD = 50` (top ~23% of valid readings, median=30, 75th=48)
- `stress.invalid_values = [-2, -1]` (was [-2] only)

### Implications for Next Steps:
- SpO2 features will need imputation for ~21% of participants
- Stress feature extraction must filter -1 values
- Can use dataset's recommended_split instead of GroupKFold for final eval
- Still use GroupKFold on train set for model development/tuning

### Concepts Taught:
- Data audit: always understand your data before modeling (40-60% of ML time is data work)
- Sentinel values: device-specific codes that look like real data but aren't
- Missingness bias: checking if missing data correlates with labels (it doesn't here — good)

### Implementation Roadmap:
1. [x] Step 1: Project setup (folders, config.py) ✅ DONE
2. [x] Step 2: Phase 0 — Data Audit (notebook 01) ✅ DONE
3. [x] Step 3: Phase 1A — Continuous feature extraction (HR, Stress, Resp, SpO2) ✅ DONE
4. [x] Step 4: Phase 1B — Event-based feature extraction (Sleep, Activity, Calories) ✅ DONE
5. [x] Step 5: Phase 1C — Daily summary assembly → (1586, 14, 22) sequences ✅ DONE
6. [ ] Step 6: Phase 2 — RF/LR baseline sanity check
7. [ ] Step 7: Phase 3 — LSTM training & evaluation
8. [ ] Step 8: Phase 4 — Analysis & interpretation

### Next: Step 6 — Phase 2 RF/LR Baseline

---

## Session 4 — 2026-06-26: Steps 3-5 — Feature Extraction Pipeline (COMPLETED)

### What was done:
- [x] Implemented `src/data/loaders.py` — generic JSON loader for continuous modalities
- [x] Implemented `src/features/continuous.py` — 13 daily features from HR/Stress/Resp/SpO2
- [x] Implemented `src/features/sleep.py` — 5 daily features from sleep stage intervals
- [x] Implemented `src/features/activity.py` — 4 daily features from activity segments + calories
- [x] Implemented `src/features/daily_summary.py` — merges all 22 features, pads to 14 days
- [x] Updated `src/data/cohort.py` — added dead-sensor filtering (exclude_dead_sensors flag)
- [x] Updated `src/data/quality.py` — added find_dead_sensor_pids() function
- [x] Updated `config.py` — fixed body_keys, added -1 to resp invalid values

### Bugs Found & Fixed:
1. **body_key wrong for resp & spo2**: both use "breathing" not their modality name
2. **Respiratory rate has -1 sentinels** too (16% of readings) — added to invalid_values
3. **69 dead-sensor participants**: all HR=0, all stress/resp invalid — filtered out
4. **Calorie values are cumulative**, not incremental — switched from sum() to max()
5. **56% of participants have <14 days** of continuous data — use padding + masking

### Design Decisions:
- Dead-sensor filtering via flag on build_cohort(exclude_dead_sensors=True)
- Padding: shorter sequences zero-padded to 14 days, actual length stored for masking
- Calorie feature: max per day (cumulative counter)
- Sleep percentages: computed against sleep-only time (excluding awake intervals)

### Cohort After Filtering:
- 1586 participants (was 1655, removed 69 dead sensors)
- Classes: healthy=560, prediabetes=410, oral_med=446, insulin=170

### Output Files:
- outputs/features/X.npy — shape (n, 14, 22) padded sequences
- outputs/features/y.npy — shape (n,) integer labels
- outputs/features/lengths.npy — shape (n,) actual sequence lengths
- outputs/features/person_ids.npy — shape (n,) participant IDs

---

## Session 5 — 2026-06-27: Steps 6-7 — Baseline + LSTM + Tuning (COMPLETED)

### What was done:
- [x] Implemented `src/models/baseline.py` — RF + LR baselines
- [x] Implemented `src/models/lstm.py` — DiasenseLSTM class
- [x] Implemented `src/models/trainer.py` — train_kfold() with StratifiedKFold CV
- [x] Implemented `src/evaluation/metrics.py` — evaluate_hierarchical() at 4/3/2 class levels
- [x] Created `tune_baselines.py` — tunes RF (10 configs), LR (7 configs), GB (4 configs)
- [x] Created `tune_lstm.py` — tunes 19 LSTM configurations
- [x] Created `check_accuracy.py` — accuracy + per-class classification reports

### Tuning Results:

**Baseline Models (best configs, 5-fold StratifiedKFold):**

| Model | Config | Features | 4-class AUC |
|---|---|---|---|
| Random Forest | n_est=500, depth=15, min_leaf=5, balanced | 44 (mean+std) | 0.6534 |
| Logistic Regression | C=0.01, balanced | 44 (mean+std) | 0.6420 |
| Gradient Boosting | n_est=500, depth=5, lr=0.01 | 44 (mean+std) | 0.6320 |

Key finding: adding temporal variability features (std across days) doubled feature count from 22 → 44 and helped RF the most.

**LSTM Models (best config):**

| Config | Hidden | Layers | Dropout | Attention | 4-class AUC |
|---|---|---|---|---|---|
| combo1 (best) | 64 | 2 | 0.4 | Yes | 0.6618 |

Key findings:
- Attention mechanism was the single biggest improvement
- Smaller hidden size (64) + higher dropout (0.4) beat larger models
- Label smoothing (0.1) helped regularization

**Accuracy Analysis:**
- RF: 40.0% | LR: 35.8% | LSTM: 34.0% | Majority baseline: 35.3%
- Accuracy is misleading because class weighting (used for AUC) pushes model to find minority class patterns at the cost of overall accuracy
- AUC is the correct primary metric for this imbalanced multi-class problem

### Files Created:
```
tune_baselines.py    — RF/LR/GB hyperparameter tuning
tune_lstm.py         — 19-config LSTM architecture search
check_accuracy.py    — Accuracy + classification reports
```

---

## Session 6 — 2026-06-27: Enhanced Features — Option C, B, D + Hourly (COMPLETED)

### What was done:
- [x] Created `src/features/enhanced.py` — Option C (19 richer features) + Option B (12 segment features) + Option D (10 hourly-derived features)
- [x] Created `run_enhanced.py` — extraction + evaluation of Option C, B, Combined
- [x] Created `src/features/hourly.py` — hourly resolution extraction (336 timesteps × 10 features)
- [x] Created `run_hourly.py` — flat 336-step LSTM evaluation
- [x] Created `run_option_d.py` — Option D (hourly-derived daily features) evaluation
- [x] Installed PyTorch CUDA 12.8 for RTX 2050 GPU acceleration

### Feature Engineering — Option C (19 richer features per day):
Clinically-motivated features extracted from raw sensor data:
- **HR (6):** HRV (RMSSD), successive diff std, circadian amplitude, resting HR estimate, nocturnal dip %, entropy proxy
- **Stress (3):** sustained high-stress episode count, IQR, low-stress %
- **SpO2 (3):** desaturation events (<90%), minimum, range
- **Activity (4):** sedentary bout count, max bout duration, active bout count, steps per active minute
- **Sleep (3):** sleep efficiency, WASO (wake after sleep onset), fragmentation

### Feature Engineering — Option B (12 segment-level features per day):
Time-of-day features by segment (night/morning/afternoon/evening):
- HR mean × 4 segments, Stress mean × 4 segments, SpO2 night/day means, Steps morning/afternoon

### Feature Engineering — Option D (10 hourly-derived daily features):
Captures WHEN things happen, not just what:
- HR peak/trough hour, hourly CV, post-meal HR (breakfast/lunch/dinner windows)
- Stress peak hour, active hours count, peak active hour, evening-vs-night HR

### Hourly Resolution Experiment:
- Shape: (1586, 336, 10) — 14 days × 24 hours × 10 features per hour
- **Result: 4-AUC = 0.6454** — worse than daily (-0.016)
- Why: 28.8% NaN rate across 336 steps, attention diluted over too many positions, simpler features per step

### Results Summary:

| Configuration | Features/day | Accuracy | 4-AUC | 3-AUC | 2-AUC |
|---|---|---|---|---|---|
| Original LSTM | 22 | 34.0% | 0.6618 | — | — |
| **+ Option C** | **41** | **33.9%** | **0.6725** | **0.6706** | **0.6632** |
| + Option B | 34 | 35.0% | 0.6587 | 0.6578 | 0.6618 |
| + Combined (C+B) | 53 | 35.5% | 0.6638 | 0.6638 | 0.6633 |
| + Option C+D | 51 | 35.1% | 0.6719 | 0.6738 | 0.6735 |
| Hourly 336-step | 10/hr | 31.1% | 0.6454 | 0.6508 | 0.6598 |
| RF + Option C | 41 | 41.5% | 0.6562 | 0.6578 | 0.6615 |
| RF + Combined | 53 | 41.7% | 0.6608 | 0.6605 | 0.6646 |

**Best LSTM: Option C (0.6725 4-AUC)** — HRV and circadian features are strongest contributors.
**Best RF: Combined (0.6608 4-AUC)** — RF benefits from all features; LSTM doesn't.

### Key Takeaways:
1. Option C (HRV, circadian amplitude, sleep quality) is the clear winner for LSTM (+1.07% AUC)
2. Option B segment features add noise for LSTM (already learns temporal patterns)
3. Combined features dilute the Option C gains for LSTM
4. Hourly resolution fails — too sparse, attention gets lost over 336 steps
5. For 3-class and 2-class, Option C+D is marginally better than C alone

### Saved Feature Arrays:
```
outputs/features/X_option_c.npy    — (1586, 14, 19) — 78.5% non-nan
outputs/features/X_option_b.npy    — (1586, 14, 12) — 72.2% non-nan
outputs/features/X_option_d.npy    — (1586, 14, 10)
outputs/features/X_hourly.npy      — (1586, 336, 10) — 28.8% nan
outputs/features/X_combined.npy    — (1586, 14, 53)
outputs/features/X_option_cd.npy   — (1586, 14, 51)
```

### Files Created:
```
src/features/enhanced.py   — Option C + B + D feature extractors
src/features/hourly.py     — Hourly resolution feature extractor
run_enhanced.py            — Enhanced feature eval (C, B, Combined)
run_hourly.py              — Hourly LSTM eval (336-step)
run_option_d.py            — Option D eval (C+D = 51 features)
run_remaining_eval.py      — Remaining evals (B LSTM + Combined)
```

---

## Session 7 — 2026-06-28: CGM Multi-Task + Knowledge Distillation (IN PROGRESS)

### What was done:
- [x] Discovered CGM data already present: `wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6/`
- [x] CGM data audit: 2,245 participants, Dexcom G6, glucose every 5 min, Open mHealth JSON format
- [x] Created `src/features/cgm.py` — CGM feature extraction functions
- [x] Created `cgm_multitask.ipynb` — full CGM multi-task pipeline as Jupyter notebook
- [x] Registered Python 3.12 CUDA kernel for Jupyter (`py312`)
- [x] Ran multi-task BiLSTM lambda sweep (λ = 0.0, 0.3, 0.5, 1.0)
- [x] Created `run_distillation.py` — teacher-student knowledge distillation pipeline
- [ ] Knowledge distillation sweep running (T × α grid search)

### CGM Data Audit:
- **Coverage:** 1,569 / 1,586 cohort participants have CGM (98.9%)
- **Duration:** min=2d, max=15d, mean=10.7d (98.7% have ≥ 7 days)
- **Valid CGM days:** 15,193 / 22,204 (68.4%) — quality gate: ≥48 readings/day
- **Format:** `body.cgm[i].blood_glucose.value` (int, mg/dL), sentinels: "High"→400, "Low"→40
- **Date alignment:** CGM and wearable share same start date, CGM covers first ~11 of 14 days

### Mean Glucose by Class (strong separation):
| Class | Mean glucose (mg/dL) | Std |
|---|---|---|
| Healthy | 117.6 | 14.0 |
| Prediabetes | 126.0 | 23.5 |
| Oral medication | 148.6 | 38.0 |
| Insulin-dependent | 175.8 | 44.0 |

### CGM Features Extracted (8 per day):
| Feature | Healthy | Prediabetes | Oral med | Insulin |
|---|---|---|---|---|
| glucose_mean | 117.5 | 126.1 | 150.1 | 181.2 |
| glucose_std | 18.9 | 21.1 | 28.0 | 39.3 |
| time_in_range (%) | 96.3 | 93.4 | 79.5 | 59.6 |
| time_above_range (%) | 2.5 | 5.9 | 19.9 | 39.4 |
| glucose_cv | 16.1 | 16.6 | 18.7 | 22.1 |
| nocturnal_mean | 121.6 | 130.7 | 156.2 | 193.6 |
| mage | 39.2 | 42.9 | 53.8 | 69.3 |
| peak_count (>200) | 2.6 | 8.9 | 35.5 | 82.7 |

### Approach 1 — Multi-Task BiLSTM:
Architecture: BiLSTM (hidden=128, bidirectional, attention) + two heads
- Classification head: 4-way diabetes class
- Glucose regression head: 8 daily CGM features (masked MSE)
- Loss: L = L_class + λ × L_glucose

**Results:**

| Config | Accuracy | 4-AUC | 3-AUC | 2-AUC |
|---|---|---|---|---|
| BiLSTM+CGM λ=0.0 (no glucose) | 36.1% | 0.6605 | 0.6663 | 0.6641 |
| BiLSTM+CGM λ=0.3 | 36.1% | 0.6614 | 0.6696 | 0.6684 |
| BiLSTM+CGM λ=0.5 | 36.4% | 0.6628 | 0.6694 | 0.6678 |
| BiLSTM+CGM λ=1.0 | 35.7% | 0.6637 | 0.6701 | 0.6682 |

**Diagnosis:** Multi-task didn't beat Option C (0.6725). Two reasons:
1. Architecture change (BiLSTM hidden=128) overfit more than proven hidden=64 attention LSTM
2. CGM does help incrementally (λ=1.0 beats λ=0.0 by +0.003) but not enough to overcome arch regression

### Approach 2 — Knowledge Distillation (COMPLETED):
```
TEACHER: wearable(41) + CGM(8) = 49 features → LSTM → soft predictions
STUDENT: wearable(41) only → LSTM → matches teacher's soft predictions
```
- Same proven architecture (hidden=64, attention) for both teacher and student
- Distillation loss: α × CE(hard_labels) + (1-α) × KL(student/T, teacher/T) × T²
- Sweep: T ∈ {2, 3, 4}, α ∈ {0.3, 0.5, 0.7}

**Teacher: 4-AUC = 0.7472 | Accuracy = 45.4%** (sees glucose directly)

**Student results (wearable only at test time):**

| T | α | Accuracy | 4-AUC | 3-AUC | 2-AUC |
|---|---|---|---|---|---|
| **2** | **0.3** | **37.6%** | **0.6879** | **0.6858** | **0.6846** |
| 2 | 0.5 | 36.6% | 0.6838 | 0.6811 | 0.6772 |
| 2 | 0.7 | 36.4% | 0.6774 | 0.6753 | 0.6705 |
| 3 | 0.3 | 37.0% | 0.6877 | 0.6855 | 0.6849 |
| 3 | 0.5 | 36.8% | 0.6838 | 0.6809 | 0.6769 |
| 3 | 0.7 | 36.6% | 0.6782 | 0.6767 | 0.6733 |
| 4 | 0.3 | 37.6% | 0.6877 | 0.6856 | 0.6850 |
| 4 | 0.5 | 36.7% | 0.6836 | 0.6808 | 0.6773 |
| 4 | 0.7 | 36.3% | 0.6779 | 0.6766 | 0.6729 |

**Best student: T=2, α=0.3 → 4-AUC = 0.6879 (+0.0154 over Option C baseline)**

Key findings:
1. Largest single improvement in the project (+0.015 AUC)
2. α=0.3 wins across all temperatures — heavier teacher reliance (70%) works best
3. Temperature has minimal effect (T=2,3,4 all ~0.6877-0.6879 at α=0.3)
4. Student recovers 20% of teacher's gap from wearable data alone
5. Accuracy improved from 33.9% → 37.6% (+3.7pp)

### CUDA Setup:
- GPU: NVIDIA GeForce RTX 2050, 4GB VRAM
- Driver: 581.83, CUDA 13.0
- PyTorch: 2.11.0+cu128 (installed from local .whl after clearing 4.2GB pip cache)
- Kernel: registered as `py312` for Jupyter

### Saved Feature Arrays (new this session):
```
outputs/features/X_cgm.npy        — (1586, 14, 8) — CGM daily features
outputs/features/cgm_mask.npy     — (1586, 14)    — valid CGM day mask (68.4% True)
outputs/features/teacher_oof_proba.npy — teacher's OOF predictions (pending)
```

### Files Created:
```
src/features/cgm.py          — CGM feature extraction functions
cgm_multitask.ipynb          — Full CGM multi-task pipeline (notebook)
make_cgm_notebook.py         — Notebook generator script
run_distillation.py          — Teacher-student knowledge distillation
```

### Implementation Roadmap (updated):
1. [x] Step 1: Project setup ✅
2. [x] Step 2: Phase 0 — Data Audit ✅
3. [x] Step 3: Phase 1A — Continuous feature extraction ✅
4. [x] Step 4: Phase 1B — Event-based feature extraction ✅
5. [x] Step 5: Phase 1C — Daily summary assembly ✅
6. [x] Step 6: Phase 2 — RF/LR baseline + tuning ✅
7. [x] Step 7: Phase 3 — LSTM training + tuning ✅
8. [x] Step 8: Enhanced features (Option C/B/D) ✅
9. [x] Step 9: Hourly resolution experiment ✅
10. [x] Step 10: CGM multi-task learning ✅
11. [ ] Step 11: Knowledge distillation (running)
12. [ ] Step 12: Final model selection + production code update
13. [ ] Step 13: Analysis & interpretation

### Current Best Model (as of 2026-06-28):
**Knowledge-distilled LSTM (T=2, α=0.3)**
- Architecture: LSTM hidden=64, 2 layers, dropout=0.4, attention
- Input: wearable only (41 features/day = 22 original + 19 Option C)
- Training: distillation from teacher that saw CGM glucose data
- **4-class AUC: 0.6879 | 3-class AUC: 0.6858 | 2-class AUC: 0.6846 | Accuracy: 37.6%**

---
