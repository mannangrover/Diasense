# Diasense — Type 2 Diabetes Prediction from Wearable Sensor Data

Diasense predicts Type 2 Diabetes severity (4-class) using **only non-invasive wearable data** from a Garmin Vivosmart 5 smartwatch and self-reported survey responses. During training, it leverages Dexcom G6 CGM (blood glucose) data via **knowledge distillation** — at test time, no invasive sensors are needed.

Built on the [AI-READI dataset](https://aireadi.org/) (v3.0.0) with 1,586 participants.

## Results

| Model | 4-AUC | 3-AUC | 2-AUC |
|---|---|---|---|
| Random Forest (baseline) | 0.6534 | — | — |
| LSTM + Attention + Option C features | 0.6725 | 0.6706 | 0.6632 |
| + Knowledge Distillation (CGM teacher) | 0.6879 | 0.6858 | 0.6846 |
| + Survey features (Hybrid LSTM) | 0.7167 | 0.7213 | 0.7474 |
| + LightGBM (summary stats + survey) | 0.7024 | 0.7321 | 0.7718 |
| + Optuna hyperparameter tuning | 0.7266 | 0.7481 | 0.7904 |
| **Best ensemble (Optuna LGB + Hybrid LSTM)** | **0.7412** | **0.7490** | **0.7937** |

**Total improvement: +10.9 points on 2-AUC** (0.6846 → 0.7937) using non-invasive data only.

## Classification Task

4-class hierarchical evaluation:
- **4-class:** Healthy / Prediabetes / Oral Medication / Insulin-Dependent
- **3-class:** Healthy / Prediabetes / Diabetic (oral + insulin merged)
- **2-class:** Healthy / Not Healthy (binary screening)

## Architecture

```
TRAINING (knowledge distillation):
  Teacher: Wearable(41) + CGM(8) → LSTM+Attention → Soft predictions (AUC 0.747)
  Student: Wearable(41) + Survey(27) → HybridLSTMAttn → Learns from teacher

INFERENCE (ensemble):
  Wearable(41) + Survey(27) → [Hybrid LSTM × 0.4 + Optuna LGB × 0.6] → Diabetes risk
```

**Models in ensemble:**
- **Hybrid LSTM:** temporal LSTM branch (wearable, 14×41) fused with static MLP branch (survey, 27) — trained via knowledge distillation from CGM teacher
- **Optuna LightGBM:** gradient boosted trees on 315 features (288 wearable summary stats + 27 survey) — hyperparameters tuned with 150 Bayesian optimization trials

## Features (41 wearable + 27 survey per participant)

### Wearable — Original (22 features/day)
| Source | Features |
|---|---|
| Heart Rate | mean, std, min, max, range |
| Stress | mean, std, high_pct |
| Respiratory Rate | mean, std |
| SpO2 | mean, std, below_95_pct |
| Sleep | total_hrs, deep_pct, rem_pct, light_pct, awake_count |
| Activity | steps, sedentary_min, active_min |
| Calories | total |

### Wearable — Enhanced Option C (19 features/day)
| Source | Features |
|---|---|
| HR Advanced | HRV (RMSSD), successive diff std, circadian amplitude, resting HR, nocturnal dip %, entropy |
| Stress Advanced | sustained high episodes, IQR, low stress % |
| SpO2 Advanced | desaturation events, min, range |
| Activity Advanced | sedentary bout count/max, active bout count, steps per active min |
| Sleep Advanced | efficiency, WASO, fragmentation |

### Survey (27 features)
Demographics, smoking/alcohol, CES-D depression score, PAID diabetes distress, diet, family history, comorbidities (hypertension, obesity, cholesterol, heart attack, stroke, kidney, circulation), vision, food insecurity.

## Project Structure

```
Diasense/
├── config.py                          # Central configuration
├── README.md
├── PROGRESS.md                        # Detailed methodology & results
│
├── src/                               # Core library
│   ├── data/                          # Cohort loading, quality checks
│   ├── features/                      # Feature extractors (continuous, sleep, activity, CGM, enhanced)
│   ├── models/                        # DiasenseLSTM, baselines, trainer
│   └── evaluation/                    # Hierarchical metrics (4/3/2 class)
│
├── 06_survey_distillation.py          # Survey extraction + Hybrid LSTM distillation
├── 07_optuna_tuning.py                # Optuna hyperparameter optimization
│
├── notebooks/
│   ├── 01_data_audit.ipynb            # Data exploration
│   ├── 02_cgm_multitask.ipynb         # CGM multi-task experiments
│   ├── 03_survey_distillation.ipynb   # Survey-augmented distillation
│   ├── 04_lightgbm_ensemble.ipynb     # LightGBM + ensembles
│   ├── 05_lgb_perday.ipynb            # Per-day LightGBM
│   └── 06_optuna_tuning.ipynb         # Optuna tuning (notebook version)
│
└── outputs/                           # Generated files (not tracked in git)
    └── features/                      # .npy arrays (X, y, OOF predictions)
```

## Dataset

**AI-READI v3.0.0** — requires separate download from [aireadi.org](https://aireadi.org/)

- 1,586 participants (after dead-sensor filtering)
- Classes: 560 healthy, 410 prediabetes, 446 oral medication, 170 insulin-dependent
- Wearable: Garmin Vivosmart 5 (HR, stress, SpO2, respiratory rate, sleep, activity)
- CGM: Dexcom G6 (used only for training via distillation)
- Survey: OMOP-format observations (demographics, comorbidities, lifestyle)
- Duration: 14 days of continuous monitoring per participant

## Setup

```bash
# Requires Python 3.12+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas scikit-learn lightgbm optuna matplotlib jupyter

# Set dataset path in config.py
DATASET_PATH = Path("D:/diabetes_dataset")

# Run the pipeline
python 06_survey_distillation.py   # Extract survey + train hybrid LSTM
python 07_optuna_tuning.py         # Optuna-tuned LightGBM
```

## Key Findings

1. **Survey data is extremely predictive** — survey-only LGB (27 features) achieves 4-AUC=0.6963, nearly matching the temporal LSTM with 41 wearable features
2. **Knowledge distillation works** — CGM glucose data, used as teacher supervision, transfers glucose-correlated patterns to a wearable-only student model
3. **Neural + tree ensembles outperform either alone** — LSTM captures temporal dynamics; LGB captures feature interactions. Decorrelated errors enable ensemble gains
4. **Strong regularization is critical** — Optuna consistently found high min_split_gain (4.0–4.8) and heavy feature subsampling for this small dataset (n=1586)
5. **HRV and circadian features matter most** — RMSSD, circadian HR amplitude, and sleep efficiency are the strongest wearable predictors

## Author

**Mannan Grover** — B.Tech CSE
