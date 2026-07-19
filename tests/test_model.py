import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def test_model_meets_r2_threshold():
    df = pd.read_csv("data/salary_data_analyst.csv")

    categorical = ["country", "experience_level", "education_required",
                   "industry", "company_size", "work_mode"]
    numeric = ["experience_years"]
    X = df[categorical + numeric]
    y = df["salary_usd"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)

    assert r2 > 0.8, "R^2 too low: " + str(r2)