import os

os.makedirs("static", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =========================================================
# 1. LOAD DATASET
# =========================================================

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Loaded Successfully")

print("Dataset Shape:", X.shape)

print("Number of Features:", X.shape[1])

print("Number of Samples:", X.shape[0])


# =========================================================
# 2. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# 3. STANDARDIZATION
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# =========================================================
# 4. DECISION TREE
# =========================================================

tree = DecisionTreeClassifier(
    random_state=42
)

tree.fit(X_train, y_train)

tree_train = accuracy_score(
    y_train,
    tree.predict(X_train)
)

tree_test = accuracy_score(
    y_test,
    tree.predict(X_test)
)


# =========================================================
# 5. RANDOM FOREST - BAGGING
# =========================================================

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(
    X_train,
    y_train
)

rf_train = accuracy_score(
    y_train,
    random_forest.predict(X_train)
)

rf_test = accuracy_score(
    y_test,
    random_forest.predict(X_test)
)


# =========================================================
# 6. XGBOOST - BOOSTING
# =========================================================

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(
    X_train,
    y_train
)

xgb_train = accuracy_score(
    y_train,
    xgb.predict(X_train)
)

xgb_test = accuracy_score(
    y_test,
    xgb.predict(X_test)
)


# =========================================================
# 7. LIGHTGBM - BOOSTING
# =========================================================

lgbm = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    verbosity=-1
)

lgbm.fit(
    X_train,
    y_train
)

lgb_train = accuracy_score(
    y_train,
    lgbm.predict(X_train)
)

lgb_test = accuracy_score(
    y_test,
    lgbm.predict(X_test)
)


# =========================================================
# 8. L1 REGULARIZATION
# =========================================================

l1_model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=1.0,
    max_iter=5000
)

l1_model.fit(
    X_train_scaled,
    y_train
)

l1_test = accuracy_score(
    y_test,
    l1_model.predict(X_test_scaled)
)


# =========================================================
# 9. L2 REGULARIZATION
# =========================================================

l2_model = LogisticRegression(
    penalty="l2",
    solver="liblinear",
    C=1.0,
    max_iter=5000
)

l2_model.fit(
    X_train_scaled,
    y_train
)

l2_test = accuracy_score(
    y_test,
    l2_model.predict(X_test_scaled)
)


# =========================================================
# 10. STORE RESULTS
# =========================================================

results = pd.DataFrame({

    "Model": [
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "LightGBM"
    ],

    "Training Accuracy": [
        tree_train,
        rf_train,
        xgb_train,
        lgb_train
    ],

    "Testing Accuracy": [
        tree_test,
        rf_test,
        xgb_test,
        lgb_test
    ]

})


print("\n================ MODEL RESULTS ================\n")

print(results)


# =========================================================
# 11. MODEL COMPARISON GRAPH
# =========================================================

x = np.arange(len(results))

width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    results["Training Accuracy"],
    width,
    label="Training Accuracy"
)

plt.bar(
    x + width / 2,
    results["Testing Accuracy"],
    width,
    label="Testing Accuracy"
)

plt.xticks(
    x,
    results["Model"]
)

plt.ylabel("Accuracy")

plt.title(
    "Training vs Testing Accuracy"
)

plt.ylim(0.7, 1.05)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "static/model_comparison.png",
    dpi=150
)

plt.close()


# =========================================================
# 12. REGULARIZATION GRAPH
# =========================================================

regularization_models = [
    "L1 Regularization",
    "L2 Regularization"
]

regularization_scores = [
    l1_test,
    l2_test
]

plt.figure(figsize=(8, 5))

plt.bar(
    regularization_models,
    regularization_scores
)

plt.ylabel("Testing Accuracy")

plt.title(
    "L1 vs L2 Regularization"
)

plt.ylim(0.8, 1.0)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "static/regularization.png",
    dpi=150
)

plt.close()


# =========================================================
# 13. TRAINING-TESTING GAP
# =========================================================

gap = (
    results["Training Accuracy"]
    -
    results["Testing Accuracy"]
)

plt.figure(figsize=(9, 5))

plt.bar(
    results["Model"],
    gap
)

plt.axhline(
    0,
    linewidth=1
)

plt.ylabel(
    "Training - Testing Accuracy"
)

plt.title(
    "Overfitting Gap"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "static/train_test.png",
    dpi=150
)

plt.close()


print("\nL1 Testing Accuracy:", l1_test)

print("L2 Testing Accuracy:", l2_test)

print("\nAll graphs generated successfully!")

print("\nProject completed successfully!")
