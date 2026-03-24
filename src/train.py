# src/train.py

from __future__ import annotations

import os
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb

from src.preprocessing import preprocess_train_test, PreprocessConfig
from src.features import engineer_train_test_features, FeatureConfig


# =============================
# CONFIG
# =============================

RANDOM_STATE = 42
TRAIN_PATH = r"D:\dataorg-financial-health-prediction-challenge20251204-19827-m2tn1n\Train.csv"
TEST_PATH = r"D:\dataorg-financial-health-prediction-challenge20251204-19827-m2tn1n\Test.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# =============================
# SEED
# =============================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# =============================
# METRIC
# =============================

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


# =============================
# CATBOOST
# =============================

def run_catboost_cv(X, y, X_test, params, cv):

    X = X.copy()
    X_test = X_test.copy()

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("missing")
        X_test[c] = X_test[c].astype("string").fillna("missing")

    n_classes = len(np.unique(y))

    oof_proba = np.zeros((len(X), n_classes))
    test_proba = np.zeros((len(X_test), n_classes))
    models = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        print(f"CatBoost Fold {fold}")

        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        tr_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
        va_pool = Pool(X_va, y_va, cat_features=cat_idx)
        te_pool = Pool(X_test, cat_features=cat_idx)

        model = CatBoostClassifier(**params)

        model.fit(
            tr_pool,
            eval_set=va_pool,
            use_best_model=True,
            verbose=0
        )

        proba_va = model.predict_proba(va_pool)
        proba_te = model.predict_proba(te_pool)

        oof_proba[va_idx] = proba_va
        test_proba += proba_te / cv.n_splits

        models.append(model)

    print(f"CatBoost OOF F1: {macro_f1(y, np.argmax(oof_proba, axis=1)):.4f}")

    return {
        "models": models,
        "oof_proba": oof_proba,
        "test_proba": test_proba
    }


# =============================
# LIGHTGBM
# =============================

def run_lgb_cv(X, y, X_test, params, cv):

    X = X.copy()
    X_test = X_test.copy()

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    for col in cat_cols:
        X[col] = X[col].astype("category").cat.codes
        X_test[col] = X_test[col].astype("category").cat.codes

    n_classes = len(np.unique(y))

    oof_proba = np.zeros((len(X), n_classes))
    test_proba = np.zeros((len(X_test), n_classes))
    models = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        print(f"LightGBM Fold {fold}")

        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )

        models.append(model)

        proba_va = model.predict_proba(X_va)
        proba_te = model.predict_proba(X_test)

        oof_proba[va_idx] = proba_va
        test_proba += proba_te / cv.n_splits

    print(f"LightGBM OOF F1: {macro_f1(y, np.argmax(oof_proba, axis=1)):.4f}")

    return {
        "models": models,
        "oof_proba": oof_proba,
        "test_proba": test_proba
    }


# =============================
# MAIN
# =============================

def main():

    set_seed(RANDOM_STATE)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    pre_cfg = PreprocessConfig()
    feat_cfg = FeatureConfig()

    # =============================
    # FEATURE ENGINEERING
    # =============================

    train_cb, test_cb = preprocess_train_test(train, test, pre_cfg, for_model="catboost")
    train_cb, test_cb = engineer_train_test_features(train_cb, test_cb, feat_cfg)

    train_lgb, test_lgb = preprocess_train_test(train, test, pre_cfg, for_model="lightgbm")
    train_lgb, test_lgb = engineer_train_test_features(train_lgb, test_lgb, feat_cfg)

    TARGET = pre_cfg.target_col

    y = train_cb[TARGET].copy()

    # =============================
    # LABEL ENCODER
    # =============================

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y))

    joblib.dump(le, "models/label_encoder.pkl")

    X_cb = train_cb.drop(columns=[TARGET])
    X_lgb = train_lgb.drop(columns=[TARGET])

    # =============================
    # 🚨 SAVE PIPELINE (CRITICAL FIX)
    # =============================

    pipeline_artifacts = {
        "feature_columns": X_cb.columns.tolist(),
        "preprocess_cfg": pre_cfg,
        "feature_cfg": feat_cfg
    }

    joblib.dump(pipeline_artifacts, "models/pipeline.pkl")
    print("✅ Pipeline saved")

    # =============================
    # TRAINING
    # =============================

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cat_params = {
        "loss_function": "MultiClass",
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 7,
        "random_seed": RANDOM_STATE,
        "verbose": 0
    }

    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.03,
        "n_estimators": 2000,
        "num_leaves": 63,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }

    cat_res = run_catboost_cv(X_cb, y, test_cb.copy(), cat_params, cv)
    lgb_res = run_lgb_cv(X_lgb, y, test_lgb.copy(), lgb_params, cv)

    # =============================
    # ENSEMBLE
    # =============================

    best_cat_w = 0.35
    best_lgb_w = 0.65

    blend_oof = best_cat_w * cat_res["oof_proba"] + best_lgb_w * lgb_res["oof_proba"]
    blend_pred = np.argmax(blend_oof, axis=1)

    score = macro_f1(y, blend_pred)
    print(f"\nFinal Ensemble F1: {score:.4f}")

    # =============================
    # SAVE MODELS
    # =============================

    joblib.dump(cat_res["models"], "models/catboost_models.pkl")
    joblib.dump(lgb_res["models"], "models/lgbm_models.pkl")
    np.save("models/oof_preds.npy", blend_oof)

    print("✅ Training complete")


if __name__ == "__main__":
    main()