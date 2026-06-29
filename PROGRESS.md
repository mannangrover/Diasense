# Diasense — Research Progress & Methodology

## Type 2 Diabetes Prediction from Non-Invasive Wearable Sensor Data

**Author:** Mannan Grover (B.Tech CSE)
**Dataset:** AI-READI v3.0.0
**Period:** June 22 – June 29, 2026

---

## Abstract

Diasense predicts Type 2 Diabetes severity using **exclusively non-invasive wearable data** from a Garmin Vivosmart 5 smartwatch. The system classifies participants into four categories (healthy, prediabetes, oral medication, insulin-dependent) and evaluates performance hierarchically at 4-class, 3-class, and 2-class levels. Through a pipeline of engineered temporal features, knowledge distillation from CGM teacher models, LightGBM gradient boosting, survey feature integration, and Optuna hyperparameter optimization, we achieve a final **2-class AUC of 0.7937** — a **+10.9 point improvement** over the initial distilled LSTM baseline of 0.6846.

---

## 1. Problem Formulation

### 1.1 Classification Task

We frame diabetes prediction as a **4-class ordinal classification** problem with hierarchical evaluation:

| Level | Classes | Clinical Meaning |
|-------|---------|-----------------|
| **4-class** | Healthy (0), Prediabetes (1), Oral Medication (2), Insulin-Dependent (3) | Full severity spectrum |
| **3-class** | Healthy (0), Prediabetes (1), Diabetic (2) | Merge oral + insulin into one diabetic class |
| **2-class** | Healthy (0), Not Healthy (1) | Binary screening: any diabetes vs. none |

### 1.2 Evaluation Metrics

- **Primary:** ROC-AUC (macro-averaged, One-vs-Rest for multi-class)
- **Secondary:** Accuracy, per-class sensitivity/specificity
- **Validation:** 5-fold Stratified K-Fold cross-validation with out-of-fold (OOF) predictions

### 1.3 Constraint: Non-Invasive Only

At inference time, the model uses **only**:
- Wearable sensor data (heart rate, stress, SpO2, respiratory rate, sleep, activity, calories)
- Self-reported survey data (demographics, comorbidities, lifestyle)

**No blood draws, no CGM, no clinical measurements at test time.** CGM data is used only during training via knowledge distillation.

---

## 2. Dataset

### 2.1 AI-READI v3.0.0

The AI-READI (Artificial Intelligence Ready and Equitable Atlas for Diabetes Insights) dataset provides multi-modal data from 2,280 participants. After quality filtering:

| Property | Value |
|----------|-------|
| Total participants (after filtering) | 1,586 |
| Monitoring duration | 14 days per participant |
| Wearable device | Garmin Vivosmart 5 |
| CGM device | Dexcom G6 (training only) |
| Survey data | 292 observation types from OMOP CDM |

### 2.2 Class Distribution

| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| Healthy | 0 | 560 | 35.3% |
| Prediabetes | 1 | 410 | 25.9% |
| Oral Medication | 2 | 446 | 28.1% |
| Insulin-Dependent | 3 | 170 | 10.7% |

### 2.3 Data Quality Filtering

- **Dead sensor removal:** 69 participants excluded (all HR readings = 0, all stress/resp invalid)
- **Sentinel value handling:** HR (-1), Stress (-2, -1), Respiratory rate (-1), SpO2 — filtered before feature computation
- **Survey sentinel values:** 777, 555, 888, 999, 99 treated as missing (NaN)
- **CGM coverage:** 1,569/1,586 participants (98.9%) have CGM data; used only for teacher training

### 2.4 Modality Availability

| Modality | Coverage |
|----------|----------|
| Heart Rate | 100% |
| Sleep | 100% |
| Activity | 100% |
| Calories | 100% |
| Respiratory Rate | 99.6% |
| Stress | 98.7% |
| SpO2 | 79.2% |
| CGM (training only) | 98.9% |

---

## 3. Feature Engineering

### 3.1 Base Wearable Features (22 per day)

Extracted from raw Garmin Vivosmart 5 JSON files for each of 14 monitoring days:

| Source | Features | Count |
|--------|----------|-------|
| Heart Rate | mean, std, min, max, range | 5 |
| Stress | mean, std, high_pct (>50) | 3 |
| Respiratory Rate | mean, std | 2 |
| SpO2 | mean, std, below_95_pct | 3 |
| Sleep | total_hrs, deep_pct, rem_pct, light_pct, awake_count | 5 |
| Activity | steps, sedentary_min, active_min | 3 |
| Calories | total (daily max of cumulative counter) | 1 |

**Output shape:** `(1586, 14, 22)` — padded to 14 days with actual lengths stored for masking.

### 3.2 Enhanced Features — Option C (19 per day)

Clinically-motivated features designed to capture diabetes-relevant physiological patterns:

| Source | Features | Rationale |
|--------|----------|-----------|
| HR Advanced | HRV (RMSSD), successive diff std, circadian amplitude, resting HR estimate, nocturnal dip %, entropy proxy | HRV reduction is a hallmark of diabetic autonomic neuropathy |
| Stress Advanced | sustained high episodes, IQR, low stress % | Chronic stress correlates with insulin resistance |
| SpO2 Advanced | desaturation events (<90%), min, range | Sleep apnea (desaturation) is comorbid with T2D |
| Activity Advanced | sedentary bout count/max, active bout count, steps per active min | Sedentary behavior patterns predict metabolic risk |
| Sleep Advanced | efficiency, WASO (wake after sleep onset), fragmentation | Poor sleep quality is bidirectionally linked with diabetes |

**Combined wearable shape:** `(1586, 14, 41)` — 22 base + 19 enhanced features per day.

### 3.3 Alternative Feature Sets Explored

| Feature Set | Shape | Result | Outcome |
|-------------|-------|--------|---------|
| Option B (segment-level) | (1586, 14, 12) | 4-AUC=0.6587 | Dropped — added noise for LSTM |
| Option D (hourly-derived) | (1586, 14, 10) | 4-AUC=0.6719 | Marginal; C alone preferred |
| Hourly resolution | (1586, 336, 10) | 4-AUC=0.6454 | Failed — too sparse (28.8% NaN) |
| Combined C+B | (1586, 14, 53) | 4-AUC=0.6638 | Diluted Option C gains |

### 3.4 CGM Features (8 per day, training only)

Extracted from Dexcom G6 continuous glucose monitoring data:

| Feature | Healthy | Prediabetes | Oral Med | Insulin |
|---------|---------|-------------|----------|---------|
| glucose_mean (mg/dL) | 117.5 | 126.1 | 150.1 | 181.2 |
| glucose_std | 18.9 | 21.1 | 28.0 | 39.3 |
| time_in_range (70–180 mg/dL, %) | 96.3 | 93.4 | 79.5 | 59.6 |
| time_above_range (%) | 2.5 | 5.9 | 19.9 | 39.4 |
| glucose_cv (%) | 16.1 | 16.6 | 18.7 | 22.1 |
| nocturnal_mean | 121.6 | 130.7 | 156.2 | 193.6 |
| MAGE | 39.2 | 42.9 | 53.8 | 69.3 |
| peak_count (>200 mg/dL) | 2.6 | 8.9 | 35.5 | 82.7 |

These features show strong class separation and are used exclusively to train the teacher model.

### 3.5 Survey Features (27 non-invasive)

Extracted from the OMOP-format `observation.csv` (707,126 rows, 2,280 participants, 292 observation types). Features selected to be non-invasive and non-leaking (excluding diabetes diagnosis labels, A1C, and diabetes-specific medications):

| Category | Features | Source Variables |
|----------|----------|-----------------|
| Demographics | age, education_years | cage, years_of_education |
| Smoking | ever smoked, current smoker | susmkncf, susmkcdur |
| Alcohol | ever consumed | sualckncf |
| Mental Health | CES-D total score, restless sleep | cestl, ces7 |
| Diabetes Distress | PAID score (0–100) | paidscore |
| Medications | sleeping pills frequency | cm_slp |
| Diet | composite score, fast food, beans, regular food, desserts, fats | dietscore, diet1/5/6/7/8 |
| Family History | parent with T2D, sibling with T2D | fh_dm2pt, fh_dm2sb |
| Comorbidities | hypertension, obesity, high cholesterol, heart attack, stroke, kidney problems, circulation | mhoccur_hbp/obs/clsh/mi/strk/rnl/circ |
| Vision | difficulty seeing | via1 |
| Food Insecurity | two items | pxfi1, pxfi2 |

**Key finding:** Survey-only LightGBM (27 features, no wearable data) achieved 4-AUC = 0.6963, nearly matching the LSTM with 41 wearable features (0.6725). Top predictors: PAID score, hypertension, family history, age, education level.

### 3.6 LightGBM Feature Flattening

For tree-based models, temporal wearable data was flattened into summary statistics:

**Per-feature aggregates (7 stats × 41 features = 287):**
- mean, std, min, max, range across valid days
- slope (linear trend over time)
- first-to-last difference

Plus `seq_length` (number of valid days) = **288 wearable summary features**.
Combined with 27 survey features = **315 total features** for LightGBM.

### 3.7 Per-Day Features (experimental)

Raw 14 × 41 = 574 individual day values fed directly to LightGBM (which handles NaN natively), combined with aggregates and survey features = **725 total features**. This gave marginal improvement over summary stats (4-AUC: 0.7067 vs 0.7024).

---

## 4. Models & Methods

### 4.1 Baseline Models

**Random Forest:**
- Best config: n_estimators=500, max_depth=15, min_samples_leaf=5, class_weight=balanced
- Features: 44 (mean + std across days for each of 22 base features)
- **4-AUC: 0.6534** | Accuracy: 40.0%

**Logistic Regression:**
- Best config: C=0.01, class_weight=balanced
- **4-AUC: 0.6420** | Accuracy: 35.8%

**Gradient Boosting:**
- Best config: n_estimators=500, max_depth=5, learning_rate=0.01
- **4-AUC: 0.6320** | Accuracy: 32.4%

### 4.2 LSTM with Attention

**Architecture (DiasenseLSTM):**
```
Input (batch, 14, F) → LSTM(hidden=64, layers=2, dropout=0.4, bidirectional=False)
    → Attention(64 → 1 weight per timestep, length-masked softmax)
    → Dropout(0.4)
    → Linear(64 → 4)
```

**Training:**
- 5-fold StratifiedKFold, 60 epochs, early stopping (patience=10)
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Loss: CrossEntropy with class weights (inversely proportional to frequency) + label smoothing (0.1)
- Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=5)

**Key architectural finding:** Attention mechanism was the single biggest improvement. It learns which of the 14 monitoring days are most informative, effectively downweighting padded or noisy days.

**Results progression:**

| Configuration | Input Features | 4-AUC |
|--------------|----------------|-------|
| Base LSTM (no attention) | 22/day | ~0.64 |
| + Attention | 22/day | 0.6618 |
| + Option C features | 41/day | 0.6725 |

### 4.3 CGM Multi-Task Learning (Experiment)

**Architecture:** BiLSTM (hidden=128, bidirectional) + two heads:
- Classification head: 4-way softmax
- Glucose regression head: predict 8 daily CGM features (masked MSE)
- Combined loss: L = L_class + λ × L_glucose

| λ | 4-AUC |
|---|-------|
| 0.0 (no glucose) | 0.6605 |
| 0.3 | 0.6614 |
| 0.5 | 0.6628 |
| 1.0 | 0.6637 |

**Conclusion:** Multi-task approach underperformed Option C LSTM (0.6725) due to architecture change (BiLSTM hidden=128 overfit). CGM signal helped incrementally (+0.003) but not enough. Abandoned in favor of knowledge distillation.

### 4.4 Knowledge Distillation (Teacher → Student)

**Approach:**
```
TEACHER: wearable(41) + CGM(8) = 49 features/day → LSTM+Attention → soft predictions
STUDENT: wearable(41) only                       → LSTM+Attention → learns from teacher
```

Both teacher and student use the proven architecture (hidden=64, 2-layer LSTM, attention, dropout=0.4). The teacher is trained on wearable + CGM features jointly.

**Distillation loss:**
```
L_student = α × CE(student, hard_labels) + (1-α) × KL(student/T, teacher/T) × T²
```

Where T = temperature (softens probability distributions) and α = hard label weight.

**Teacher performance:** 4-AUC = 0.7472 | Accuracy = 45.4%

**Student sweep (9 configurations: T ∈ {2,3,4} × α ∈ {0.3,0.5,0.7}):**

| T | α | 4-AUC | 3-AUC | 2-AUC |
|---|---|-------|-------|-------|
| **2** | **0.3** | **0.6879** | **0.6858** | **0.6846** |
| 2 | 0.5 | 0.6838 | 0.6811 | 0.6772 |
| 2 | 0.7 | 0.6774 | 0.6753 | 0.6705 |
| 3 | 0.3 | 0.6877 | 0.6855 | 0.6849 |
| 3 | 0.5 | 0.6838 | 0.6809 | 0.6769 |
| 3 | 0.7 | 0.6782 | 0.6767 | 0.6733 |
| 4 | 0.3 | 0.6877 | 0.6856 | 0.6850 |
| 4 | 0.5 | 0.6836 | 0.6808 | 0.6773 |
| 4 | 0.7 | 0.6779 | 0.6766 | 0.6729 |

**Best student: T=2, α=0.3** — heavier teacher reliance (70% soft labels) works best. The student recovers ~20% of the teacher's performance gap using only wearable data.

### 4.5 Hybrid LSTM with Survey Features

**Architecture (HybridLSTMAttn):**
```
Temporal branch:  Input(batch, 14, 41) → LSTM(64, 2 layers, attention) → h_temporal(64)
Static branch:    Input(batch, 27)     → MLP(27→64→32, ReLU, Dropout) → h_static(32)
Fusion:           concat(h_temporal, h_static) → Linear(96→4)
```

This architecture fuses temporal wearable patterns with static survey features before the classification head.

**Training:** Same distillation setup as 4.4 but with survey features added to the student.

**Best config sweep (T ∈ {2,3,4} × α ∈ {0.3,0.5,0.7}):**

| T | α | 4-AUC | 3-AUC | 2-AUC |
|---|---|-------|-------|-------|
| **2** | **0.5** | **0.7167** | **0.7213** | **0.7474** |
| 2 | 0.3 | 0.7089 | 0.7118 | 0.7393 |
| 3 | 0.5 | 0.7131 | 0.7180 | 0.7431 |

**Improvement:** +0.0288 4-AUC over distilled LSTM without survey (0.6879 → 0.7167).

### 4.6 LightGBM with Summary Statistics

LightGBM trained on flattened wearable summary stats (288 features) + survey (27 features) = 315 total. Three dedicated models trained for each evaluation level:

**Hyperparameters (default):**
- n_estimators=2000, learning_rate=0.05, max_depth=6
- num_leaves=31, min_child_samples=20, is_unbalance=True
- Early stopping: patience=50 on validation logloss

| Model | 4-AUC | 3-AUC | 2-AUC |
|-------|-------|-------|-------|
| LGB 4-class | 0.7024 | — | 0.7658 |
| LGB 3-class (dedicated) | — | 0.7321 | — |
| LGB 2-class (dedicated) | — | — | 0.7718 |

**LGB + Hybrid LSTM Ensemble (best weights):**
- LGB×0.4 + Hybrid×0.6: 4-AUC=0.7333, 3-AUC=0.7394, 2-AUC=0.7823

**Top LightGBM features:** PAID score (diabetes distress), hypertension, family history (parent), age, education years, high cholesterol, diet score — survey features dominated the importance rankings.

**Survey-only ablation:** LGB with only 27 survey features achieved 4-AUC = 0.6963 — nearly matching the LSTM with 41 temporal wearable features (0.6725), demonstrating the extreme predictive power of self-reported health data.

### 4.7 LightGBM with Per-Day Features

Instead of summary statistics, raw per-day values (14 × 41 = 574) were fed directly to LightGBM alongside aggregates (123) and survey (27) = **725 total features**.

| Model | 4-AUC |
|-------|-------|
| Per-day LGB 4-class | 0.7067 |

Marginal improvement over summary stats (0.7024). LightGBM already captured most temporal patterns from aggregates.

**3-way ensemble (PerDay×0.3 + SumLGB×0.3 + Hybrid×0.4):** 4-AUC=0.7338, 3-AUC=0.7407, 2-AUC=0.7852

### 4.8 Optuna Hyperparameter Optimization

Bayesian optimization via Optuna (150 trials per task) on the LightGBM summary features model. Key hyperparameters tuned:

| Parameter | Search Range | Best (4-class) | Best (3-class) | Best (2-class) |
|-----------|-------------|----------------|----------------|----------------|
| learning_rate | 0.005–0.1 (log) | 0.0114 | 0.0089 | 0.0507 |
| max_depth | 3–10 | 4 | 9 | 10 |
| num_leaves | 15–127 | 113 | 97 | 56 |
| min_child_samples | 5–60 | 58 | 60 | 36 |
| colsample_bytree | 0.3–1.0 | 0.30 | 0.32 | 0.37 |
| reg_alpha | 1e-3–10 (log) | 0.001 | 0.68 | 7.74 |
| min_split_gain | 0–5 | 4.68 | 4.82 | 4.01 |

**Key insight:** High `min_split_gain` (4.0–4.8) preferred across all tasks — strong regularization is critical for this small dataset (n=1586).

**Optuna Results:**

| Model | 4-AUC | 3-AUC | 2-AUC |
|-------|-------|-------|-------|
| Optuna LGB 4-class | 0.7266 | 0.7392 | 0.7888 |
| Optuna LGB 3-class (dedicated) | — | 0.7481 | — |
| Optuna LGB 2-class (dedicated) | — | — | 0.7904 |

### 4.9 Augmented Survey Feature Mining

Expanded from 27 to 60 survey features by adding:
- CES-D depression items (ces1–ces10): individual items beyond total score
- Food insecurity (pxfi3–5): additional items
- Vision difficulty (via2, via3): reading, driving
- Additional comorbidities: arthritis, lung problems, urinary, cognitive, cataracts, digestive, hearing, dry eye, other heart issues, falls, low BP
- Lifestyle: activity level (dmlact), fruit/veg consumption (dmlfrveg)
- Fasting hours (paate), racial discrimination items (pxrd1/4/7/10)

Univariate screening showed **activity level** (AUC=0.5887) and **arthritis** (AUC=0.5624) as strongest new features. Optuna tuning (200 trials) on 348 augmented features yielded 2-AUC = 0.7928 (marginal over 0.7904 with 27 features).

---

## 5. Ensemble Methods

### 5.1 Weighted Averaging

All models produce out-of-fold (OOF) probability predictions via 5-fold StratifiedKFold. For multi-class models, 2-class probability is computed by summing P(prediabetes) + P(oral_med) + P(insulin).

Ensemble weights determined by grid search over OOF predictions:

### 5.2 Stacking (Meta-Learner)

Attempted LogisticRegression and LightGBM as second-level learners on 22 meta-features (all OOF predictions). Best stacked result: 2-AUC = 0.7940 (LogisticRegression, C=1.0). Stacking did not significantly improve over weighted averaging due to high correlation between base models.

### 5.3 Best Ensemble Configurations

| Ensemble | 4-AUC | 3-AUC | 2-AUC |
|----------|-------|-------|-------|
| LGB×0.4 + Hybrid×0.6 | 0.7333 | 0.7394 | 0.7823 |
| PerDay×0.3 + SumLGB×0.3 + Hybrid×0.4 | 0.7338 | 0.7407 | 0.7852 |
| **OptLGB×0.6 + Hybrid×0.4** | **0.7412** | **0.7490** | **0.7937** |
| OptLGB×0.7 + Hybrid×0.3 | 0.7397 | 0.7484 | 0.7941 |
| Opt4×0.3 + Opt2(ded) + LGB + Hybrid (4-way) | — | — | 0.7978 |

---

## 6. Final Results

### 6.1 Model Progression

| Model | 4-AUC | 3-AUC | 2-AUC | Key Innovation |
|-------|-------|-------|-------|----------------|
| Random Forest baseline | 0.6534 | — | — | Flattened daily features |
| LSTM + Attention | 0.6618 | — | — | Temporal modeling with attention |
| + Option C features | 0.6725 | 0.6706 | 0.6632 | HRV, circadian, sleep quality features |
| + Knowledge Distillation | 0.6879 | 0.6858 | 0.6846 | CGM teacher → wearable student |
| + Survey features (Hybrid LSTM) | 0.7167 | 0.7213 | 0.7474 | Fused temporal + static branches |
| + LightGBM (summary stats) | 0.7024 | — | 0.7658 | Tree model on flattened features |
| + LGB + Hybrid ensemble | 0.7333 | 0.7394 | 0.7823 | Neural + tree ensemble |
| + Optuna tuning | 0.7266 | 0.7481 | 0.7904 | Bayesian hyperparameter search |
| **+ Best ensemble (Optuna+Hybrid)** | **0.7412** | **0.7490** | **0.7937** | **Optimized multi-model blend** |

### 6.2 Improvement from Baseline

| Metric | Baseline (Distilled LSTM) | Best Ensemble | Improvement |
|--------|--------------------------|---------------|-------------|
| 4-AUC | 0.6879 | 0.7412 | **+0.0533** |
| 3-AUC | 0.6858 | 0.7490 | **+0.0632** |
| 2-AUC | 0.6846 | 0.7937 | **+0.1091** |

### 6.3 Improvement from Original RF Baseline

| Metric | RF Baseline | Best Ensemble | Improvement |
|--------|-------------|---------------|-------------|
| 4-AUC | 0.6534 | 0.7412 | **+0.0878** |

---

## 7. Key Findings & Insights

1. **Survey data is extremely predictive.** Survey-only LightGBM (27 features) achieved 4-AUC = 0.6963, nearly matching the temporal LSTM with 41 wearable features per day (0.6725). Self-reported comorbidities (hypertension, obesity), family history, and diabetes distress (PAID score) are among the strongest individual predictors.

2. **Knowledge distillation successfully transfers CGM signal.** The distilled student model (wearable-only) recovers ~20% of the teacher's performance gap. Low α (0.3 = 70% teacher soft labels) works best, suggesting the soft probability distributions contain richer information than hard class labels.

3. **Neural + tree ensembles outperform either alone.** The LSTM captures temporal dynamics and inter-day patterns; LightGBM captures feature interactions and handles survey data natively. Their errors are sufficiently decorrelated for ensemble gains.

4. **Strong regularization is critical.** Optuna consistently found high `min_split_gain` (4.0–4.8) and heavy `colsample_bytree` regularization (0.30–0.37) — expected for n=1586 with 315+ features.

5. **HRV and circadian features matter most** among wearable signals. RMSSD, circadian HR amplitude, and sleep efficiency are the strongest wearable predictors of diabetes severity, consistent with known diabetic autonomic neuropathy pathophysiology.

6. **Daily resolution beats hourly.** 14 meaningful daily summaries outperform 336 sparse hourly readings (4-AUC: 0.6725 vs 0.6454). The attention mechanism effectively learns which days are most informative.

7. **Diminishing returns near 0.80.** Multiple approaches (stacking, augmented features, multi-seed ensembles, fine-grained weight search) were explored to push 2-AUC from 0.7937 to 0.80+, reaching 0.7978 at best. The remaining gap likely requires fundamentally new data sources or substantially larger cohorts.

---

## 8. Technical Infrastructure

### 8.1 Hardware
- **GPU:** NVIDIA GeForce RTX 2050, 4GB VRAM
- **CUDA:** 13.0, PyTorch 2.11.0+cu128

### 8.2 Software Stack
- Python 3.12, PyTorch, LightGBM, Optuna, scikit-learn
- Jupyter with `python312-cuda` kernel for GPU-accelerated notebooks

### 8.3 Reproducibility
- Random seed: 42 (numpy, random, torch, CUDA)
- All OOF predictions saved as `.npy` arrays in `outputs/features/`
- 5-fold StratifiedKFold with fixed random_state throughout

---

## 9. Saved Artifacts

### 9.1 Feature Arrays (`outputs/features/`)

| File | Shape | Description |
|------|-------|-------------|
| person_ids.npy | (1586,) | Participant IDs |
| X.npy | (1586, 14, 22) | Base wearable features |
| X_option_c.npy | (1586, 14, 19) | Enhanced wearable features |
| X_cgm.npy | (1586, 14, 8) | CGM features (training only) |
| X_survey.npy | (1586, 27) | Survey features |
| y.npy | (1586,) | 4-class labels |
| lengths.npy | (1586,) | Valid day counts |

### 9.2 OOF Predictions (`outputs/features/`)

| File | Shape | Model | 2-AUC |
|------|-------|-------|-------|
| teacher_oof_proba.npy | (1586, 4) | Teacher (wearable+CGM) | — |
| oof_T2_a0.3.npy | (1586, 4) | Distilled LSTM (wearable) | 0.6846 |
| oof_hybrid_T2_a0.5.npy | (1586, 4) | Hybrid LSTM+Survey | 0.7474 |
| oof_lgb_4class.npy | (1586, 4) | LGB summary 4-class | 0.7658 |
| oof_lgb_3class.npy | (1586, 3) | LGB summary 3-class | — |
| oof_lgb_2class.npy | (1586,) | LGB dedicated binary | 0.7718 |
| oof_lgb_perday_4class.npy | (1586, 4) | LGB per-day 4-class | 0.7690 |
| oof_optuna_4class.npy | (1586, 4) | Optuna LGB 4-class | 0.7888 |
| oof_optuna_3class.npy | (1586, 3) | Optuna LGB 3-class | — |
| oof_optuna_2class.npy | (1586,) | Optuna LGB binary | 0.7904 |

---

## 10. File Index

### Scripts
| File | Purpose |
|------|---------|
| config.py | Central configuration (paths, constants, feature definitions) |
| 06_survey_distillation.py | Survey feature extraction + Hybrid LSTM distillation |
| 07_optuna_tuning.py | Optuna hyperparameter optimization (150 trials × 3 tasks) |

### Notebooks
| File | Purpose |
|------|---------|
| notebooks/01_data_audit.ipynb | Interactive data exploration and quality checks |
| notebooks/02_cgm_multitask.ipynb | CGM multi-task experiments and visualizations |
| notebooks/03_survey_distillation.ipynb | Survey-augmented knowledge distillation |
| notebooks/04_lightgbm_ensemble.ipynb | LightGBM with summary stats + ensembles |
| notebooks/05_lgb_perday.ipynb | LightGBM with per-day features |
| notebooks/06_optuna_tuning.ipynb | Optuna tuning (notebook version, superseded by script) |
| notebooks/lstm.ipynb | LSTM architecture exploration |

### Source Library (`src/`)
| File | Purpose |
|------|---------|
| src/data/cohort.py | Load, merge, filter, label the cohort |
| src/data/loaders.py | Generic JSON loader for sensor files |
| src/data/quality.py | Data quality checks, dead sensor detection |
| src/features/continuous.py | 13 daily features from HR, Stress, Resp, SpO2 |
| src/features/sleep.py | 5 daily features from sleep stage intervals |
| src/features/activity.py | 4 daily features from activity + calories |
| src/features/daily_summary.py | Merge all 22 features → (N, 14, 22) sequences |
| src/features/enhanced.py | Option C (19) + Option D (10) feature extractors |
| src/features/cgm.py | Dexcom G6 CGM feature extraction |
| src/models/baseline.py | Random Forest + Logistic Regression |
| src/models/lstm.py | DiasenseLSTM with attention |
| src/models/trainer.py | Training loop with StratifiedKFold CV |
| src/evaluation/metrics.py | Hierarchical evaluation (4/3/2 class) |
