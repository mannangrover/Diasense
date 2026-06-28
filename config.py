"""
Diasense Configuration
======================
Central configuration file. Every path, constant, and feature definition
lives here. All other files import from this module.

Change a path or add a feature in ONE place — never hardcode values elsewhere.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Project root (this file's parent directory)
PROJECT_ROOT = Path(__file__).parent

# Raw dataset (AI-READI v3.0.0)
DATASET_PATH = Path("D:/diabetes_dataset")

# Key dataset files
PARTICIPANTS_TSV = DATASET_PATH / "participants.tsv"
WEARABLE_MANIFEST_TSV = DATASET_PATH / "wearable_activity_monitor" / "manifest.tsv"

# Wearable data subdirectories
WEARABLE_BASE = DATASET_PATH / "wearable_activity_monitor"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FEATURES_DIR = OUTPUTS_DIR / "features"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# =============================================================================
# COHORT PARAMETERS
# =============================================================================

# Minimum sensor duration to include a participant
MIN_SENSOR_DAYS = 14

# Fixed sequence length for LSTM (number of days)
SEQUENCE_LENGTH = 14

# =============================================================================
# LABEL DEFINITIONS
# =============================================================================

# 4-class mapping: study_group string → integer label
LABEL_MAP_4CLASS = {
    "healthy": 0,
    "pre_diabetes_lifestyle_controlled": 1,
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": 2,
    "insulin_dependent": 3,
}

# Human-readable class names (for plots and reports)
CLASS_NAMES_4 = ["healthy", "prediabetes", "oral_med", "insulin"]
CLASS_NAMES_3 = ["healthy", "prediabetes", "diabetic"]
CLASS_NAMES_2 = ["healthy", "not_healthy"]

# Hierarchical groupings for evaluation
# 3-class: merge oral_med (2) and insulin (3) into diabetic (2)
LABEL_MAP_4_TO_3 = {0: 0, 1: 1, 2: 2, 3: 2}

# 2-class: merge everything non-healthy into 1
LABEL_MAP_4_TO_2 = {0: 0, 1: 1, 2: 1, 3: 1}

# =============================================================================
# MODALITY CONFIGURATION
# =============================================================================

# Continuous modalities: point-in-time measurements (~1 min intervals)
# Each entry: (manifest_filepath_column, json_body_key, json_value_key, invalid_values)
CONTINUOUS_MODALITIES = {
    "heart_rate": {
        "filepath_col": "heartrate_filepath",
        "body_key": "heart_rate",
        "value_key": "heart_rate",
        "invalid_values": [0],        # HR=0 is sensor dropout, not real
    },
    "stress": {
        "filepath_col": "stress_level_filepath",
        "body_key": "stress",
        "value_key": "stress",
        "invalid_values": [-2, -1],   # -2 and -1 are device sentinels (44% of readings are -1)
    },
    "respiratory_rate": {
        "filepath_col": "respiratory_rate_filepath",
        "body_key": "breathing",
        "value_key": "respiratory_rate",
        "invalid_values": [-2, -1],   # -2 and -1 are device sentinels (16% are -1)
    },
    "spo2": {
        "filepath_col": "oxygen_saturation_filepath",
        "body_key": "breathing",
        "value_key": "oxygen_saturation",
        "invalid_values": [0],        # SpO2=0 is sensor dropout
    },
}

# Event-based modalities: interval data (start_time → end_time)
EVENT_MODALITIES = {
    "sleep": {
        "filepath_col": "sleep_filepath",
        "body_key": "sleep",
    },
    "activity": {
        "filepath_col": "physical_activity_filepath",
        "body_key": "activity",
    },
}

# Sparse point modality
CALORIE_CONFIG = {
    "filepath_col": "active_calories_filepath",
    "body_key": "activity",
}

# Valid sleep stages (from data inspection)
SLEEP_STAGES = ["deep", "light", "rem", "awake"]

# Valid activity types (from data inspection)
ACTIVITY_TYPES = ["sedentary", "walking", "running", "generic"]

# =============================================================================
# DAILY FEATURE NAMES
# =============================================================================

# These are the 22 features extracted per participant per day.
# This list defines the column order in the final feature arrays.

DAILY_FEATURE_NAMES = [
    # Continuous — Heart Rate (5 features)
    "hr_mean",
    "hr_std",
    "hr_min",
    "hr_max",
    "hr_range",
    # Continuous — Stress (3 features)
    "stress_mean",
    "stress_std",
    "stress_high_pct",
    # Continuous — Respiratory Rate (2 features)
    "resp_mean",
    "resp_std",
    # Continuous — SpO2 (3 features)
    "spo2_mean",
    "spo2_std",
    "spo2_below95_pct",
    # Event-based — Sleep (5 features)
    "sleep_total_hrs",
    "sleep_deep_pct",
    "sleep_rem_pct",
    "sleep_light_pct",
    "sleep_awake_count",
    # Event-based — Activity (3 features)
    "steps_total",
    "sedentary_minutes",
    "active_minutes",
    # Sparse — Calories (1 feature)
    "calories_total",
]

NUM_DAILY_FEATURES = len(DAILY_FEATURE_NAMES)  # Should be 22

# Stress threshold for "high stress" percentage calculation
# Set from data audit: positive stress values have median=30, 75th=48.
# 50 captures the top ~23% of valid readings — clinically meaningful.
STRESS_HIGH_THRESHOLD = 50

# SpO2 clinical threshold
SPO2_LOW_THRESHOLD = 95.0  # Below 95% is clinically significant

# Participants with completely dead sensors (all HR=0, all stress/resp invalid).
# Precomputed from data audit to avoid re-scanning 1655 JSON files each run.
DEAD_SENSOR_PIDS = {
    1059, 1078, 1082, 1107, 1108, 1127, 1130, 1162, 1265, 1279,
    1319, 1342, 1375, 1459, 1460, 1560, 1601, 1633, 1638, 1671,
    1672, 1673, 1700, 1711, 1724, 1755, 1761, 1781, 1796, 4137,
    4173, 4176, 4197, 4217, 4218, 4238, 4243, 4277, 4288, 4300,
    4303, 4324, 4325, 4342, 4355, 4363, 4375, 4390, 4393, 4408,
    4414, 4415, 4475, 4552, 4580, 4598, 4607, 4641, 4675, 4679,
    7289, 7324, 7331, 7370, 7380, 7410, 7441, 7560, 7565,
}

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

# LSTM
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_LEARNING_RATE = 0.001
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS = 30

# Validation
N_FOLDS = 5
RANDOM_STATE = 42

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

import numpy as np
import random

def set_seed(seed=RANDOM_STATE):
    """Set random seeds for reproducibility across numpy, random, and torch."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
