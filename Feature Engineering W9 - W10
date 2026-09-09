# ============================================
# FEATURE ENGINEERING + SCALING + MISSING DATA
# ============================================
print("PROGRAM STARTED")
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================
# 1. CREATE SAMPLE DATASET
# ============================================

data = {
    "Age": [25, 30, np.nan, 40, 35, 28, np.nan, 50, 45, 32],
    
    "Salary": [
        30000,
        50000,
        40000,
        np.nan,
        70000,
        45000,
        60000,
        90000,
        np.nan,
        55000
    ],
    
    "Experience": [1, 5, 3, 10, 7, 2, 4, 20, 15, 6],
    
    "City": [
        "Nagpur",
        "Pune",
        "Mumbai",
        np.nan,
        "Nagpur",
        "Pune",
        "Mumbai",
        "Nagpur",
        "Pune",
        "Mumbai"
    ],
    
    "Purchased": [0, 1, 0, 1, 1, 0, 0, 1, 1, 0]
}


df = pd.DataFrame(data)

print("Original Dataset:")
print(df)


# ============================================
# 2. FEATURE ENGINEERING
# ============================================

# Salary per year of experience
df["Salary_Per_Experience"] = (
    df["Salary"] / df["Experience"]
)

print("\nAfter Feature Engineering:")
print(df)


# ============================================
# 3. DEFINE FEATURES AND TARGET
# ============================================

X = df.drop("Purchased", axis=1)

y = df["Purchased"]


# Numerical Columns
numerical_features = [
    "Age",
    "Salary",
    "Experience",
    "Salary_Per_Experience"
]


# Categorical Columns
categorical_features = [
    "City"
]


# ============================================
# 4. NUMERICAL PREPROCESSING
# ============================================

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ============================================
# 5. CATEGORICAL PREPROCESSING
# ============================================

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# ============================================
# 6. COMBINE PREPROCESSING
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================
# 7. CREATE COMPLETE MODEL PIPELINE
# ============================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression())
    ]
)


# ============================================
# 8. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================
# 9. TRAIN MODEL
# ============================================

model.fit(X_train, y_train)


# ============================================
# 10. PREDICTION
# ============================================

y_pred = model.predict(X_test)


# ============================================
# 11. EVALUATION
# ============================================

print("\nPredictions:")
print(y_pred)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
