"""Train and evaluate five classifiers on the UCI 'Predict Students' Dropout
and Academic Success' dataset, then save them for the Streamlit app.

Run:  python train_models.py
Outputs:
    test_data.csv                  held-out test split (features + Target)
    model/<name>.joblib            fitted pipelines (scaler + classifier)
    model/feature_columns.json     input columns and class labels the app expects
    model/best_params.json         hyper-parameters chosen by cross-validation
    model/metrics_summary.csv      test-set metrics for every model
"""

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from eval_utils import METRIC_NAMES, compute_metrics

ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "data" / "dropout_raw.csv"
MODEL_DIR = ROOT / "model"
TARGET = "Target"
SEED = 1108
TEST_SIZE = 0.20

# name shown in app/README -> (file stem, estimator, small search grid)
CANDIDATES = {
    "Logistic Regression": (
        "logistic_regression",
        LogisticRegression(max_iter=3000, random_state=SEED),
        {"clf__C": [0.05, 0.1, 0.5, 1.0, 5.0], "clf__class_weight": [None, "balanced"]},
    ),
    "Decision Tree": (
        "decision_tree",
        DecisionTreeClassifier(random_state=SEED),
        {
            "clf__max_depth": [4, 6, 8, 10, None],
            "clf__min_samples_leaf": [1, 5, 10, 20],
            "clf__criterion": ["gini", "entropy"],
        },
    ),
    "KNN": (
        "knn",
        KNeighborsClassifier(),
        {"clf__n_neighbors": [5, 9, 15, 21, 31], "clf__weights": ["uniform", "distance"]},
    ),
    "Naive Bayes": (
        "naive_bayes",
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
    ),
    "Support Vector Machine": (
        "svm",
        # SVC gives no probabilities by itself; calibration adds them (needed for AUC)
        CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=SEED), method="isotonic", ensemble=False
        ),
        {"clf__estimator__C": [1.0, 3.0, 10.0], "clf__estimator__gamma": ["scale", 0.01, 0.003]},
    ),
}


def load_data():
    df = pd.read_csv(RAW_DATA)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Loaded {df.shape[0]} rows x {X.shape[1]} features")
    print("Class balance:\n" + y.value_counts().to_string() + "\n")
    return X, y


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)} (saved to test_data.csv)\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, best_params = [], {}

    for display_name, (stem, estimator, grid) in CANDIDATES.items():
        start = time.time()
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
        search = GridSearchCV(pipe, grid, scoring="f1_weighted", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        scores = compute_metrics(y_test, y_pred, y_proba, model.classes_)

        joblib.dump(model, MODEL_DIR / f"{stem}.joblib")
        best_params[display_name] = {k.replace("clf__", "").replace("estimator__", ""): v for k, v in search.best_params_.items()}
        results.append({"Model": display_name, **scores})
        print(
            f"{display_name:<24} CV-F1={search.best_score_:.4f}  test-acc={scores['Accuracy']:.4f}  "
            f"({time.time() - start:.1f}s)  best={best_params[display_name]}"
        )

    summary = pd.DataFrame(results)[["Model"] + METRIC_NAMES]
    summary.to_csv(MODEL_DIR / "metrics_summary.csv", index=False)

    meta = {
        "target": TARGET,
        "features": X.columns.tolist(),
        "classes": sorted(y.unique().tolist()),
        "models": {name: f"{stem}.joblib" for name, (stem, _, _) in CANDIDATES.items()},
    }
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(meta, indent=2))
    (MODEL_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))

    print("\nTest-set comparison:")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
