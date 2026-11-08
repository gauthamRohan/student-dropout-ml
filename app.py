"""Streamlit app: evaluate the five saved student-outcome classifiers on an
uploaded test CSV. Models are only loaded and scored here - never retrained."""

import json
from io import BytesIO
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from eval_utils import METRIC_NAMES, compute_metrics

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
SAMPLE_CSV = APP_DIR / "test_data.csv"

# single blue hue: light step for "other", dark step for the highlighted item
BLUE_LIGHT, BLUE_MAIN, BLUE_DARK = "#9ec5f4", "#2a78d6", "#0d366b"
HEAT_RANGE = ["#eef4fc", "#0d366b"]

st.set_page_config(page_title="Student Outcome Classifier", page_icon="🎓", layout="wide")


# ---------- loading ----------
@st.cache_resource
def load_meta():
    meta = json.loads((MODEL_DIR / "feature_columns.json").read_text())
    params = json.loads((MODEL_DIR / "best_params.json").read_text())
    return meta, params


@st.cache_resource
def load_model(file_name):
    return joblib.load(MODEL_DIR / file_name)


@st.cache_data
def read_csv(raw_bytes):
    return pd.read_csv(BytesIO(raw_bytes))


def check_columns(df, meta):
    """Return a list of problems that stop evaluation (empty list = OK)."""
    problems = []
    missing = [c for c in meta["features"] if c not in df.columns]
    if missing:
        problems.append(f"Missing {len(missing)} feature column(s): " + ", ".join(missing))
    if meta["target"] not in df.columns:
        problems.append(f"Missing target column '{meta['target']}'.")
    else:
        unknown = sorted(set(df[meta["target"]].astype(str)) - set(meta["classes"]))
        if unknown:
            problems.append("Unknown target labels: " + ", ".join(unknown))
    if not problems:
        bad = [c for c in meta["features"] if not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            problems.append("These columns must be numeric: " + ", ".join(bad))
    return problems


def score_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return y_pred, y_proba, compute_metrics(y, y_pred, y_proba, model.classes_)


@st.cache_data(show_spinner="Scoring all five models...")
def score_all(raw_bytes, model_files):
    df = read_csv(raw_bytes)
    meta, _ = load_meta()
    X, y = df[meta["features"]], df[meta["target"]].astype(str)
    rows = []
    for name, file_name in model_files.items():
        _, _, scores = score_model(load_model(file_name), X, y)
        rows.append({"Model": name, **scores})
    return pd.DataFrame(rows)


# ---------- charts ----------
def confusion_chart(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, actual in enumerate(labels):
        row_total = cm[i].sum() or 1
        for j, predicted in enumerate(labels):
            rows.append({
                "Actual": actual,
                "Predicted": predicted,
                "Count": int(cm[i, j]),
                "Share of actual": cm[i, j] / row_total,
            })
    data = pd.DataFrame(rows)
    base = alt.Chart(data).encode(
        x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(labelAngle=0, title="Predicted class")),
        y=alt.Y("Actual:N", sort=labels, title="Actual class"),
    )
    cells = base.mark_rect(stroke="white", strokeWidth=2, cornerRadius=4).encode(
        color=alt.Color(
            "Share of actual:Q",
            scale=alt.Scale(domain=[0, 1], range=HEAT_RANGE),
            legend=alt.Legend(title="Row share", format="%"),
        ),
        tooltip=[
            "Actual", "Predicted", "Count",
            alt.Tooltip("Share of actual:Q", format=".1%"),
        ],
    )
    text = base.mark_text(fontSize=15, fontWeight="bold").encode(
        text="Count:Q",
        color=alt.condition(
            alt.datum["Share of actual"] > 0.5, alt.value("white"), alt.value("#1f1f1e")
        ),
    )
    return (cells + text).properties(height=320)


def comparison_chart(table, metric, highlight):
    data = table[["Model", metric]].copy()
    data["Selected"] = data["Model"] == highlight
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, height=26).encode(
        x=alt.X(f"{metric}:Q", scale=alt.Scale(domain=[0, 1]), title=metric),
        y=alt.Y("Model:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
        color=alt.condition(alt.datum.Selected, alt.value(BLUE_MAIN), alt.value(BLUE_LIGHT)),
        tooltip=["Model", alt.Tooltip(f"{metric}:Q", format=".4f")],
    )
    labels = bars.mark_text(align="left", dx=6, color="#5b5b57").encode(
        text=alt.Text(f"{metric}:Q", format=".3f"), color=alt.value("#5b5b57")
    )
    return (bars + labels).properties(height=260)


# ---------- sidebar ----------
meta, best_params = load_meta()
model_files = meta["models"]
target = meta["target"]

with st.sidebar:
    st.header("1 · Test data")
    source = st.radio(
        "Data source",
        ["Upload a CSV", "Use bundled test_data.csv"],
        # a link ending in ?data=sample opens the app with the bundled test split
        index=1 if st.query_params.get("data") == "sample" else 0,
        help="Upload only test data. It must contain all 36 feature columns and the "
        f"'{target}' column.",
    )
    raw_bytes = None
    if source == "Upload a CSV":
        uploaded = st.file_uploader("Test CSV", type=["csv"])
        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
    else:
        raw_bytes = SAMPLE_CSV.read_bytes()

    st.download_button(
        "Download sample test CSV",
        SAMPLE_CSV.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        width="stretch",
    )

    st.header("2 · Model")
    model_name = st.selectbox("Classifier", list(model_files))
    with st.expander("Tuned hyper-parameters"):
        st.json(best_params[model_name])

    st.caption(
        "Models were trained once on an 80% stratified split of the UCI dataset "
        "(5-fold grid search). This app only loads and scores them."
    )

# ---------- header ----------
st.title("🎓 Student Outcome Classifier")
st.markdown(
    "Predicts whether a higher-education student will **Drop out**, stay **Enrolled**, or "
    "**Graduate**, using enrolment, socio-economic and first-year academic data "
    "([UCI dataset #697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)). "
    "Upload a labelled test file to see how each saved model performs on it."
)

if raw_bytes is None:
    st.info(
        "⬅️ Upload a test CSV in the sidebar, or switch the data source to the bundled "
        "test split. The sample file can be downloaded from the sidebar."
    )
    st.stop()

try:
    data = read_csv(raw_bytes)
except Exception as err:  # malformed file
    st.error(f"Could not read the CSV: {err}")
    st.stop()

problems = check_columns(data, meta)
if problems:
    st.error("This file can't be evaluated:\n\n- " + "\n- ".join(problems))
    st.stop()

X = data[meta["features"]]
y = data[target].astype(str)

counts = y.value_counts().reindex(meta["classes"], fill_value=0)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(data):,}")
for col, cls in zip((c2, c3, c4), meta["classes"]):
    col.metric(cls, f"{counts[cls]:,}  ({counts[cls] / len(data):.0%})")

with st.expander("Preview uploaded data"):
    st.dataframe(data.head(20), width="stretch")

model = load_model(model_files[model_name])
y_pred, y_proba, scores = score_model(model, X, y)

tab_eval, tab_compare, tab_pred = st.tabs(
    [f"Evaluation · {model_name}", "Compare all models", "Predictions"]
)

with tab_eval:
    cols = st.columns(len(METRIC_NAMES))
    for col, name in zip(cols, METRIC_NAMES):
        value = scores[name]
        col.metric(name, "n/a" if pd.isna(value) else f"{value:.4f}")
    st.caption(
        "Precision, Recall and F1 are support-weighted averages over the three classes; "
        "AUC is one-vs-rest (weighted)."
    )

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Confusion matrix")
        st.altair_chart(confusion_chart(y, y_pred, meta["classes"]), width="stretch")
    with right:
        st.subheader("Classification report")
        report = classification_report(
            y, y_pred, labels=meta["classes"], output_dict=True, zero_division=0
        )
        # accuracy is already shown above; its row has no meaningful support value
        report_df = pd.DataFrame(report).T.drop(index="accuracy", errors="ignore")
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(
            report_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}),
            width="stretch",
        )

with tab_compare:
    table = score_all(raw_bytes, model_files)
    metric = st.radio("Metric to plot", METRIC_NAMES, horizontal=True, index=METRIC_NAMES.index("F1"))
    left, right = st.columns([1, 1.2])
    with left:
        st.altair_chart(comparison_chart(table, metric, model_name), width="stretch")
        st.caption(f"Darker bar = the model selected in the sidebar ({model_name}).")
    with right:
        st.dataframe(
            table.set_index("Model").style.format("{:.4f}").highlight_max(
                axis=0, props="font-weight:bold; background-color:rgba(42,120,214,0.18)"
            ),
            width="stretch",
        )
        best = table.loc[table["F1"].idxmax(), "Model"]
        st.markdown(f"Best weighted F1 on this file: **{best}**. Bold cells mark the top score per metric.")

with tab_pred:
    class_order = list(model.classes_)
    result = pd.DataFrame({
        "Actual": y.values,
        "Predicted": y_pred,
        "Confidence": y_proba.max(axis=1),
    })
    result["Correct"] = result["Actual"] == result["Predicted"]
    for i, cls in enumerate(class_order):
        result[f"P({cls})"] = y_proba[:, i]

    only_wrong = st.toggle("Show only misclassified rows")
    shown = result[~result["Correct"]] if only_wrong else result
    st.write(f"{(~result['Correct']).sum():,} of {len(result):,} rows misclassified by {model_name}.")
    st.dataframe(
        shown.style.format({c: "{:.3f}" for c in shown.columns if c.startswith("P(") or c == "Confidence"}),
        width="stretch",
        height=380,
    )
    st.download_button(
        "Download predictions (CSV)",
        result.to_csv(index=False).encode(),
        file_name=f"predictions_{model_files[model_name].replace('.joblib', '')}.csv",
        mime="text/csv",
    )
