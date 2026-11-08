# Student Outcome Classifier – Dropout, Enrolled or Graduate

ML Assignment 2 · M.Tech (AIML) · BITS Pilani WILP

- **GitHub repository:** https://github.com/gauthamRohan/student-dropout-ml
- **Live Streamlit app:** https://student-dropout-classifier.streamlit.app/
  (open https://student-dropout-classifier.streamlit.app/?data=sample to start with the bundled test data loaded)

---

## a. Problem statement

Universities lose a large share of students before graduation, and the earlier an at-risk student
is identified, the more time there is to help them. The goal of this project is to predict, from
information known at enrolment and at the end of the first academic year, whether a student will
**drop out**, **still be enrolled** at the end of the normal course duration, or **graduate**.

This is a **multi-class (3-class) classification** problem. Five classifiers are trained on the same
data and compared on six metrics, and the trained models are served through an interactive
Streamlit app that scores an uploaded test file without retraining.

## b. Dataset description

| Item | Details |
|---|---|
| Name | Predict Students' Dropout and Academic Success |
| Source | UCI Machine Learning Repository, dataset #697 – https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |
| Origin | Polytechnic Institute of Portalegre (Portugal), students enrolled 2008/09 – 2018/19 across 17 undergraduate degrees |
| Instances | 4,424 students |
| Features | 36 (all numeric; categorical fields are already integer-coded) |
| Target | `Target` – Dropout / Enrolled / Graduate |
| Missing values | None |
| Class balance | Graduate 2,209 (49.9%), Dropout 1,421 (32.1%), Enrolled 794 (17.9%) |

Feature groups:

- **Demographic** – marital status, nationality, gender, age at enrolment, displaced, international.
- **Socio-economic** – mother's / father's qualification and occupation, scholarship holder, debtor, tuition fees up to date, educational special needs.
- **Academic path at entry** – application mode and order, course, day/evening attendance, previous qualification and its grade, admission grade.
- **Academic performance** – curricular units credited, enrolled, evaluated, approved, average grade and units without evaluation, for the 1st and 2nd semester (12 features).
- **Macro-economic** – unemployment rate, inflation rate and GDP in the year of enrolment.

**Preparation:** column names were trimmed of stray whitespace and the file was saved as
`data/dropout_raw.csv`. The data was split **80 / 20, stratified by class** (seed 1108):
3,539 rows for training and 885 rows for testing. The test split is saved as `test_data.csv`
(36 features + `Target`) and is the file to upload in the app. Every model is a scikit-learn
`Pipeline` of `StandardScaler` + classifier, so scaling is saved inside each model file.

## c. GitHub repository

https://github.com/gauthamRohan/student-dropout-ml

```
student-dropout-ml/
├── app.py                 # Streamlit app (loads saved models, no retraining)
├── train_models.py        # trains, tunes, evaluates and saves the 5 models
├── eval_utils.py          # metric function shared by training and the app
├── requirements.txt
├── README.md
├── test_data.csv          # held-out test split (885 rows)
├── data/
│   └── dropout_raw.csv    # full UCI dataset
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── feature_columns.json   # expected input columns and class labels
    ├── best_params.json       # hyper-parameters chosen by cross-validation
    └── metrics_summary.csv    # test-set metrics below
```

## d. Models used

Hyper-parameters were chosen with a 5-fold stratified `GridSearchCV` on the **training split only**
(scoring: weighted F1). The test split was used once, for the final numbers.

| Model | Implementation | Selected hyper-parameters |
|---|---|---|
| Logistic Regression | `LogisticRegression` | C = 5.0, class_weight = balanced |
| Decision Tree | `DecisionTreeClassifier` | criterion = gini, max_depth = 10, min_samples_leaf = 20 |
| KNN | `KNeighborsClassifier` | k = 5, weights = uniform |
| Naive Bayes | `GaussianNB` | var_smoothing = 1e-9 |
| Support Vector Machine | `SVC` (RBF kernel) + `CalibratedClassifierCV` (isotonic) for probabilities | C = 10, gamma = 0.01 |

**Metric definitions (3-class problem):** Precision, Recall and F1 are support-weighted averages
over the three classes. AUC is the one-vs-rest ROC AUC, weighted by class support, computed from
predicted probabilities. MCC is the multi-class Matthews Correlation Coefficient.

### Comparison table (test set, 885 students)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7412 | **0.8963** | **0.7775** | 0.7412 | 0.7534 | 0.5959 |
| Decision Tree | 0.7412 | 0.8737 | 0.7256 | 0.7412 | 0.7310 | 0.5711 |
| KNN | 0.7051 | 0.8223 | 0.6912 | 0.7051 | 0.6948 | 0.5093 |
| Naive Bayes | 0.6870 | 0.8237 | 0.6705 | 0.6870 | 0.6762 | 0.4795 |
| Support Vector Machine | **0.7751** | 0.8911 | 0.7643 | **0.7751** | **0.7646** | **0.6273** |

Per-class recall on the test set (Dropout / Enrolled / Graduate):
LR 0.72 / 0.62 / 0.80 · DT 0.75 / 0.33 / 0.88 · KNN 0.71 / 0.30 / 0.85 · NB 0.70 / 0.28 / 0.82 · SVM 0.76 / 0.39 / 0.92.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline: highest AUC (0.896) and precision (0.778). With `class_weight='balanced'` it has by far the best recall on the hard "Enrolled" class (0.62 vs ≤0.39 for the others), at the cost of lower overall accuracy (0.741) because some Graduates are pushed into Enrolled. Best choice if catching "still enrolled / at-risk" students matters more than raw accuracy. |
| Decision Tree | Ties LR on accuracy (0.741) but has lower AUC (0.874) and MCC (0.571). Depth was limited to 10 with min 20 samples per leaf to stop overfitting. It relies heavily on one feature — 2nd-semester approved units (≈65% of importance) — so it captures the main signal but misses most Enrolled students (recall 0.33). Easy to interpret. |
| KNN | Weaker (accuracy 0.705, MCC 0.509). With 36 dimensions, many of them integer-coded categories, Euclidean distance after scaling is a poor similarity measure, and the minority "Enrolled" class is outvoted by neighbours (recall 0.30). |
| Naive Bayes | Lowest scores (accuracy 0.687, MCC 0.480). The independence assumption is clearly broken: the 1st- and 2nd-semester curricular features are strongly correlated, and many coded categorical features are not Gaussian. Still a fast baseline with a reasonable AUC (0.824). |
| Support Vector Machine | Best overall: highest accuracy (0.775), weighted F1 (0.765) and MCC (0.627), plus a near-best AUC (0.891). The RBF kernel captures non-linear interactions between academic-performance features, and it has the best Graduate (0.92) and Dropout (0.76) recall. Needed probability calibration for AUC and is the slowest to train. |
| **Overall winner** | **Support Vector Machine.** It leads on accuracy, F1 and MCC (MCC is the most reliable single metric under class imbalance). Logistic Regression is a close second and is preferable if the priority is detecting "Enrolled" students. All models find the "Enrolled" class hardest, because it sits between Dropout and Graduate and is the smallest class. |

---

## Streamlit app

Features:

1. **CSV upload** of test data (or one click to use the bundled `test_data.csv`; a sample file can be downloaded from the sidebar).
2. **Model selection dropdown** for the five saved models, with their tuned hyper-parameters.
3. **Evaluation metrics** – Accuracy, AUC, Precision, Recall, F1, MCC for the selected model.
4. **Confusion matrix** (heatmap with counts and row share) and **classification report**.
5. **Compare all models** tab – all six metrics for the five models on the uploaded file, with a bar chart for any metric.
6. **Predictions** tab – per-row predicted class and class probabilities, filter for misclassified rows, CSV download.

The uploaded file must contain the 36 feature columns and the `Target` column. The app checks the
columns and shows a clear error if something is missing. Models are only loaded with `joblib` and
used for prediction; nothing is retrained in the app.

## How to run locally

```bash
git clone https://github.com/gauthamRohan/student-dropout-ml.git
cd student-dropout-ml
pip install -r requirements.txt

python train_models.py      # optional: retrains and rewrites model/ and test_data.csv
streamlit run app.py
```

Tested with Python 3.11.
