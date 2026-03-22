# src/infer.py

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd

from catboost import Pool
from src.preprocessing import preprocess_train_test, PreprocessConfig
from src.features import engineer_train_test_features, FeatureConfig


# =============================
# CONFIG
# =============================

TRAIN_PATH = "D:\\dataorg-financial-health-prediction-challenge20251204-19827-m2tn1n\\Train.csv"
TEST_PATH = "D:\\dataorg-financial-health-prediction-challenge20251204-19827-m2tn1n\\Test.csv"  # needed for consistent preprocessing

OUTPUT_PATH = "outputs/submission.csv"

CAT_WEIGHT = 0.35
LGB_WEIGHT = 0.65


# =============================
# THRESHOLD FUNCTION
# =============================

def predict_with_thresholds(proba, thresholds, class_names):
    preds = np.zeros(proba.shape[0], dtype=int)

    idx_map = {cls: i for i, cls in enumerate(class_names)}

    for i in range(proba.shape[0]):
        assigned = False

        for cls in class_names:
            cls_idx = idx_map[cls]

            if proba[i, cls_idx] >= thresholds[cls]:
                preds[i] = cls_idx
                assigned = True
                break

        if not assigned:
            preds[i] = np.argmax(proba[i])

    return preds


# =============================
# MAIN
# =============================

def main():

    print("🔄 Loading data...")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    pre_cfg = PreprocessConfig()
    feat_cfg = FeatureConfig()

    # =============================
    # PREPROCESS + FEATURES
    # =============================

    print("⚙️ Applying preprocessing and feature engineering...")

    # CatBoost pipeline
    train_cb, test_cb = preprocess_train_test(train, test, pre_cfg, for_model="catboost")
    train_cb, test_cb = engineer_train_test_features(train_cb, test_cb, feat_cfg)

    # LightGBM pipeline
    train_lgb, test_lgb = preprocess_train_test(train, test, pre_cfg, for_model="lightgbm")
    train_lgb, test_lgb = engineer_train_test_features(train_lgb, test_lgb, feat_cfg)

    TARGET = pre_cfg.target_col
    ID_COL = pre_cfg.id_col

    X_cb_test = test_cb.copy()
    X_lgb_test = test_lgb.copy()

    # =============================
    # LOAD MODELS
    # =============================

    print("📦 Loading models...")

    cat_models = joblib.load("models/catboost_models.pkl")
    lgb_models = joblib.load("models/lgbm_models.pkl") git 

    label_encoder = joblib.load("models/label_encoder.pkl")
    class_names = list(label_encoder.classes_)

    n_classes = len(class_names)

    # =============================
    # CATBOOST PREDICTIONS
    # =============================

    print("🐱 CatBoost inference...")

    cat_cols = [c for c in X_cb_test.columns if not pd.api.types.is_numeric_dtype(X_cb_test[c])]
    cat_idx = [X_cb_test.columns.get_loc(c) for c in cat_cols]

    for c in cat_cols:
        X_cb_test[c] = X_cb_test[c].astype("string").fillna("missing")

    cat_proba = np.zeros((len(X_cb_test), n_classes))

    for model in cat_models:
        pool = Pool(X_cb_test, cat_features=cat_idx)
        cat_proba += model.predict_proba(pool) / len(cat_models)

    # =============================
    # LIGHTGBM PREDICTIONS
    # =============================

    print("🌳 LightGBM inference...")

    X_lgb_test = X_lgb_test.copy()

    for col in X_lgb_test.columns:
        if not pd.api.types.is_numeric_dtype(X_lgb_test[col]):
            X_lgb_test[col] = X_lgb_test[col].astype("category").cat.codes

    lgb_proba = np.zeros((len(X_lgb_test), n_classes))

    for model in lgb_models:
        lgb_proba += model.predict_proba(X_lgb_test) / len(lgb_models)

    # =============================
    # ENSEMBLE BLEND
    # =============================

    print("🔀 Blending predictions...")

    blend_proba = CAT_WEIGHT * cat_proba + LGB_WEIGHT * lgb_proba

    # =============================
    # THRESHOLDS (FROM NOTEBOOK)
    # =============================

    print("🎯 Applying thresholds...")

    # You can tune these later if needed
    thresholds = {
        "High": 0.45,
        "Low": 0.40,
        "Medium": 0.50
    }

    preds = predict_with_thresholds(blend_proba, thresholds, class_names)

    # =============================
    # FINAL LABELS
    # =============================

    final_labels = label_encoder.inverse_transform(preds)

    # =============================
    # SAVE SUBMISSION
    # =============================

    submission = pd.DataFrame({
        ID_COL: test[ID_COL],
        TARGET: final_labels
    })

    os.makedirs("outputs", exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)

    print("✅ Submission saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()