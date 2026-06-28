# Diasense — Type 2 Diabetes Prediction from Wearable Sensor Data

Diasense predicts Type 2 Diabetes severity (4-class) using **only non-invasive wearable data** from a Garmin Vivosmart 5 smartwatch. During training, it leverages Dexcom G6 CGM (blood glucose) data via **knowledge distillation** — at test time, no invasive sensors are needed.

Built on the [AI-READI dataset](https://aireadi.org/) (v3.0.0) with 1,586 participants.

## Results

| Model | 4-class AUC | 3-class AUC | 2-class AUC | Accuracy |
|---|---|---|---|---|
| Random Forest (baseline) | 0.6534 | — | — | 40.0% |
| LSTM + Attention | 0.6618 | — | — | 34.0% |
| + Option C features (HRV, circadian) | 0.6725 | 0.6706 | 0.6632 | 33.9% |
| **+ Knowledge Distillation (CGM)** | **0.6879** | **0.6858** | **0.6846** | **37.6%** |

**Teacher model** (wearable + CGM input): 4-AUC = 0.7472, Accuracy = 45.4%

Best student config: Temperature=2, Alpha=0.3 (70% teacher soft labels, 30% hard labels)

## Classification Task

4-class hierarchical evaluation:
- **4-class:** Healthy / Prediabetes / Oral Medication / Insulin-Dependent
- **3-class:** Healthy / Prediabetes / Diabetic (oral + insulin merged)
- **2-class:** Healthy / Not Healthy

## Architecture

```
TRAINING:
  Teacher: Wearable(41) + CGM(8) → LSTM+Attention → Soft predictions (AUC 0.747)
  Student: Wearable(41) only     → LSTM+Attention → Learns from teacher's soft labels

DEPLOYMENT:
  Garmin smartwatch → 41 features/day × 14 days → Student model → Diabetes risk
```

- LSTM: hidden=64, 2 layers, dropout=0.4, attention mechanism
- Training: 5-fold Stratified CV, class-weighted loss, label smoothing
- Distillation: KL divergence with temperature scaling

## Features (41 per day)

### Original (22 features)
| Source | Features |
|---|---|
| Heart Rate | mean, std, min, max, range |
| Stress | mean, std, high_pct |
| Respiratory Rate | mean, std |
| SpO2 | mean, std, below_95_pct |
| Sleep | total_hrs, deep_pct, rem_pct, light_pct, awake_count |
| Activity | steps, sedentary_min, active_min |
| Calories | total |

### Enhanced — Option C (19 additional features)
| Source | Features |
|---|---|
| HR Advanced | HRV (RMSSD), successive diff std, circadian amplitude, resting HR, nocturnal dip %, entropy |
| Stress Advanced | sustained high episodes, IQR, low stress % |
| SpO2 Advanced | desaturation events, min, range |
| Activity Advanced | sedentary bout count/max, active bout count, steps per active min |
| Sleep Advanced | efficiency, WASO, fragmentation |

## Project Structure

```
Diasense/
├── config.py                          # Central configuration (paths, constants, features)
├── README.md
├── PROGRESS.md                        # Detailed session-by-session progress journal
│
├── src/                               # Core library
│   ├── data/
│   │   ├── cohort.py                  # Load, merge, filter, label the cohort
│   │   ├── loaders.py                 # Generic JSON loader for sensor files
│   │   └── quality.py                 # Data quality checks, dead sensor detection
│   ├── features/
│   │   ├── continuous.py              # 13 features from HR, Stress, Resp, SpO2
│   │   ├── sleep.py                   # 5 features from sleep stage intervals
│   │   ├── activity.py               # 4 features from activity + calories
│   │   ├── daily_summary.py           # Merge all 22 features → (N, 14, 22) sequences
│   │   ├── enhanced.py                # Option C (19 richer) + Option D (10 hourly-derived)
│   │   ├── hourly.py                  # Hourly resolution extractor (experimental)
│   │   └── cgm.py                     # Dexcom G6 blood glucose feature extraction
│   ├── models/
│   │   ├── baseline.py                # Random Forest + Logistic Regression
│   │   ├── lstm.py                    # DiasenseLSTM class
│   │   └── trainer.py                 # Training loop with StratifiedKFold CV
│   └── evaluation/
│       └── metrics.py                 # Hierarchical evaluation (4/3/2 class)
│
├── 01_extract_features.py             # Step 1: Extract 22 daily features → X.npy
├── 02_tune_baselines.py               # Step 2: Tune RF, LR, GB baselines
├── 03_tune_lstm.py                    # Step 3: Tune LSTM architectures (19 configs)
├── 04_extract_enhanced.py             # Step 4: Extract Option C features + evaluate
├── 05_knowledge_distillation.py       # Step 5: Teacher-student CGM distillation
│
├── notebooks/
│   ├── 01_data_audit.ipynb            # Interactive data exploration
│   └── 02_cgm_multitask.ipynb         # CGM multi-task experiments + visualizations
│
└── outputs/                           # Generated files (not tracked in git)
    ├── features/                      # .npy arrays (X, y, lengths, etc.)
    ├── models/                        # .pt model checkpoints
    └── figures/                       # .png plots and visualizations
```

## Dataset

**AI-READI v3.0.0** — requires separate download from [aireadi.org](https://aireadi.org/)

- 1,586 participants (after dead-sensor filtering)
- Classes: 560 healthy, 410 prediabetes, 446 oral medication, 170 insulin-dependent
- Wearable: Garmin Vivosmart 5 (HR, stress, SpO2, respiratory rate, sleep, activity)
- CGM: Dexcom G6 (blood glucose every 5 min, used only for training via distillation)
- Duration: 14 days of continuous monitoring per participant

## Setup

```bash
# Requires Python 3.12+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas scikit-learn matplotlib jupyter

# Set dataset path in config.py
DATASET_PATH = Path("D:/diabetes_dataset")   # adjust to your location

# Run the pipeline
python 01_extract_features.py        # Extract base features
python 04_extract_enhanced.py        # Extract enhanced features
python 05_knowledge_distillation.py  # Train distilled model (best result)
```

## Key Findings

1. **HRV and circadian features matter most** — RMSSD, circadian HR amplitude, and sleep efficiency are the strongest wearable predictors of diabetes severity
2. **Knowledge distillation works** — CGM glucose data, when used as teacher supervision, transfers glucose-correlated patterns to a wearable-only student model
3. **Daily resolution beats hourly** — 14 meaningful daily summaries outperform 336 sparse hourly readings
4. **Attention mechanism is critical** — the single biggest LSTM improvement; learns which days in the 14-day window are most informative

## Author

**Mannan Grover** — B.Tech CSE
