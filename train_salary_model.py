"""
Trains a salary prediction model for Data Analyst roles using salary_data_analyst.csv.

Run this once to produce models/salary_model.joblib, which app.py loads
to power the "Estimate Your Salary" widget in the dashboard.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. Load data ---
df = pd.read_csv("data/salary_data_analyst.csv")

CATEGORICAL_FEATURES = [
    "country", "experience_level", "education_required",
    "industry", "company_size", "work_mode",
]
NUMERIC_FEATURES = ["experience_years"]
TARGET = "salary_usd"

X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
y = df[TARGET]

# --- 2. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 3. Preprocessing + model pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ],
    remainder="passthrough",  # keeps experience_years as-is
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )),
])

model.fit(X_train, y_train)

# --- 4. Evaluate honestly (this is what you cite in your README/interview) ---
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"Test MAE:  ${mae:,.0f}")
print(f"Test R^2:  {r2:.3f}")
print(f"Baseline (predict mean) MAE: ${mean_absolute_error(y_test, [y_train.mean()]*len(y_test)):,.0f}")

# --- 5. Save model ---
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/salary_model.joblib")
print("Saved model to models/salary_model.joblib")
