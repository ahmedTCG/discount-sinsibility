"""
config.py
---------
Central configuration for the Discount Sensitivity Pipeline.
"""

# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================

ID_COLS = ["externalcustomerkey"]

DATE_COLS = ["as_of_date", "first_order_date", "last_order_date"]

LEAKAGE_COLS = [
    "discount_abs_lifetime_eur",
    "discount_rate_lifetime",
    "share_of_orders_with_discount",
    "share_of_items_discounted",
    "avg_discount_per_order",
    "max_discount_single_order",
]

DROP_COLS = ["gender", "shops_included", "registration_flag"]

WINDOW_TOKENS = ["_15d", "_30d", "_3m", "_6m", "_12m"]

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

COUNTRY_MIN_SHARE = 0.01

TEST_SIZE = 0.2
RANDOM_SEED = 42

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}

NUM_BOOST_ROUND = 300
EARLY_STOPPING_ROUNDS = 30

# =============================================================================
# SEGMENTATION THRESHOLDS
# =============================================================================

SEGMENT_THRESHOLDS = [0.0, 0.2, 0.6, 1.0]
SEGMENT_LABELS = ["full_price", "conditional", "discount_driven"]

# =============================================================================
# FILE PATHS
# =============================================================================

DEFAULT_PATHS = {
    "data_clean": "data/df_model_clean.parquet",
    "metadata": "artifacts/metadata.json",
    "model": "artifacts/model.txt",
    "feature_importance": "artifacts/feature_importance.csv",
    "metrics": "artifacts/train_metrics.json",
    "scores": "data/scores.csv",
    "scores_bucketized": "data/scores_bucketized.csv",
}

# =============================================================================
# TARGET DEFINITION
# =============================================================================

TARGET_COLUMN = "y_discount_sensitive"
TARGET_SOURCE_COLS = ["share_of_orders_with_discount", "discount_abs_lifetime_eur"]
