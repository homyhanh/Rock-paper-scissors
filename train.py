# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# import joblib  # để lưu model
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report

# # 1. Đọc dữ liệu từ CSV
# df = pd.read_csv("data.csv")

# # 2. Chia input (X) và label (y)
# X = df.drop(columns=["Label"])
# y = df["Label"]

# # 3. Chia train/test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# # 4. Huấn luyện model (Random Forest)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # 5. Đánh giá
# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))

# # 6. Lưu model để dùng sau
# joblib.dump(model, "model.pkl")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data.csv")
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_iter", type=int, default=40, help="số tổ hợp thử RandomizedSearch")
    args = parser.parse_args()

    # 1) Load
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"[ERROR] Không đọc được CSV: {e}")
        sys.exit(1)

    if "Label" not in df.columns:
        print("[ERROR] Thiếu cột Label trong CSV")
        sys.exit(1)

    X = df.drop(columns=["Label"]).copy()
    y = df["Label"].astype(int)

    # ép numeric & loại NaN/Inf nếu có
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1) & y.notna()
    X, y = X[valid], y[valid].astype(int)

    # 2) Split train/val/test: 60/20/20 (stratify)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=args.seed, stratify=y_trainval
    )
    # (0.25 của 0.8 = 0.2 → tổng = 0.6/0.2/0.2)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print("Class distribution (train):")
    print(y_train.value_counts().sort_index())

    # 3) Tuning bằng RandomizedSearchCV trên TRAIN (đánh giá CV nội bộ)
    base = RandomForestClassifier(random_state=args.seed, n_jobs=-1, class_weight="balanced_subsample")
    param_dist = {
        "n_estimators":        np.linspace(150, 500, 8, dtype=int),
        "max_depth":           [None, 6, 8, 10, 12, 16, 20],
        "min_samples_split":   [2, 3, 4, 5, 6, 8, 10],
        "min_samples_leaf":    [1, 2, 3, 4, 5],
        "max_features":        ["sqrt", "log2", None, 0.4, 0.6, 0.8],
        "bootstrap":           [True, False],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    search = RandomizedSearchCV(
        base, param_distributions=param_dist, n_iter=args.n_iter,
        scoring="f1_macro", n_jobs=-1, cv=cv, random_state=args.seed, verbose=1
    )
    search.fit(X_train, y_train)
    print("\n=== Best CV params ===")
    print(search.best_params_)
    print("Best CV score (f1_macro):", search.best_score_)

    # 4) Đánh giá trên VALIDATION (giữ-out) với best estimator
    best = search.best_estimator_
    y_val_pred = best.predict(X_val)
    print("\n=== Validation report ===")
    print(classification_report(y_val, y_val_pred, digits=4))
    print("Confusion matrix (val):\n", confusion_matrix(y_val, y_val_pred))

    # 5) Refit trên TRAIN+VAL với best params
    final = RandomForestClassifier(
        **{k: search.best_params_.get(k, getattr(best, k)) for k in search.best_params_},
        random_state=args.seed, n_jobs=-1, class_weight="balanced_subsample"
    )
    final.fit(pd.concat([X_train, X_val], axis=0), pd.concat([y_train, y_val], axis=0))

    # 6) Đánh giá cuối trên TEST
    y_test_pred = final.predict(X_test)
    print("\n=== Test report (final) ===")
    print(classification_report(y_test, y_test_pred, digits=4))
    print("Confusion matrix (test):\n", confusion_matrix(y_test, y_test_pred))

    # 7) Lưu model
    joblib.dump(final, args.model)
    print(f"\n✅ Saved final model to {args.model}")

if __name__ == "__main__":
    main()
