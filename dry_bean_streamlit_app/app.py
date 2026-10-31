from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
TARGET_COLUMN = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "k-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")
st.title("🫘 Dry Bean Classifier Lab")
st.write(
    "Explore five machine-learning classifiers trained to predict dry bean variety "
    "from 16 numerical measurements. Upload a labelled CSV test file to compare "
    "predictions with the known class labels."
)

with st.sidebar:
    st.header("Model settings")
    selected_model_name = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()))
    st.divider()
    st.caption("Expected target column: `Class`")
    st.caption("Upload test data only; do not upload training data.")

@st.cache_resource
def load_model(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_data
def load_sample_data():
    path = APP_DIR / "test_data.csv"
    return pd.read_csv(path) if path.exists() else None

def calculate_metrics(model, X, y_true):
    y_pred = model.predict(X)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    try:
        y_scores = model.predict_proba(X)
        estimator = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
        metrics["AUC (macro OVR)"] = roc_auc_score(
            y_true, y_scores, multi_class="ovr", average="macro", labels=estimator.classes_
        )
    except (ValueError, TypeError, AttributeError):
        metrics["AUC (macro OVR)"] = np.nan
    return y_pred, metrics

model = load_model(MODEL_FILES[selected_model_name])
if model is None:
    st.error(
        f"Missing `model/{MODEL_FILES[selected_model_name]}`. Run the training notebook, "
        "then copy the exported model files into the repository's `model/` folder."
    )
    st.stop()

try:
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
except Exception:
    st.error("Missing `model/label_encoder.joblib`. Copy it from the notebook export.")
    st.stop()

class_names = list(label_encoder.classes_)
label_to_encoded = {name: index for index, name in enumerate(class_names)}
encoded_to_label = {index: name for index, name in enumerate(class_names)}

if hasattr(model, "feature_names_in_"):
    feature_columns = list(model.feature_names_in_)
elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("classifier"), "feature_names_in_"):
    feature_columns = list(model.named_steps["classifier"].feature_names_in_)
else:
    st.error("Could not identify the trained model's feature names. Train using the provided notebook.")
    st.stop()

with st.expander("CSV requirements", expanded=False):
    st.markdown(
        "- Include all 16 numeric feature columns used during training.\n"
        "- Include a `Class` column with the actual bean variety labels.\n"
        "- Evaluation metrics require known labels, so an unlabelled CSV cannot be scored."
    )

sample_data = load_sample_data()
uploaded_file = st.file_uploader("Upload labelled test data (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        source_label = uploaded_file.name
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()
elif sample_data is not None:
    st.info("No file uploaded. Showing the bundled example test data.")
    data = sample_data.copy()
    source_label = "Bundled example test data"
    st.download_button(
        "Download example test_data.csv",
        data=sample_data.to_csv(index=False).encode("utf-8"),
        file_name="test_data.csv",
        mime="text/csv",
    )
else:
    st.warning("Upload a labelled test CSV to begin. Add the notebook-generated `test_data.csv` to the project root to enable the example.")
    st.stop()

st.subheader("Test data preview")
st.caption(f"Source: {source_label} · {len(data):,} rows · {len(data.columns):,} columns")
st.dataframe(data.head(10), use_container_width=True)

if "Class" not in data.columns:
    st.error("Required target column `Class` was not found.")
    st.stop()

missing_features = [column for column in feature_columns if column not in data.columns]
if missing_features:
    st.error("Missing required feature columns: " + ", ".join(missing_features))
    st.stop()

if data.empty:
    st.error("The CSV has no data rows.")
    st.stop()

X_test = data[feature_columns].copy()
try:
    X_test = X_test.apply(pd.to_numeric, errors="raise")
except Exception:
    st.error("All feature columns must contain numeric values.")
    st.stop()
if X_test.isna().any().any():
    st.error("Missing values were found in feature columns. Clean the test data and upload again.")
    st.stop()

actual_labels = data["Class"].astype(str).str.strip()
unknown_labels = sorted(set(actual_labels) - set(class_names))
if unknown_labels:
    st.error("Unknown class labels found: " + ", ".join(unknown_labels))
    st.stop()

y_true = actual_labels.map(label_to_encoded).to_numpy()
try:
    y_pred, metrics = calculate_metrics(model, X_test, y_true)
except Exception as exc:
    st.error(f"Prediction or metric calculation failed: {exc}")
    st.stop()

st.subheader("Evaluation results")
st.caption(f"Selected model: {selected_model_name} · {len(data):,} test rows")
metric_order = ["Accuracy", "AUC (macro OVR)", "Precision (macro)", "Recall (macro)", "F1 (macro)", "MCC"]
cols = st.columns(3)
for index, metric_name in enumerate(metric_order):
    value = metrics.get(metric_name, np.nan)
    cols[index % 3].metric(metric_name, f"{value:.4f}" if pd.notna(value) else "N/A")

st.caption("Precision, Recall and F1 use macro averaging. AUC is macro one-vs-rest from class probabilities. MCC is multiclass.")

left, right = st.columns([1.05, 1])
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(
        y_true, y_pred, labels=np.arange(len(class_names)),
        target_names=class_names, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

st.subheader("Predictions")
results_df = data.copy()
results_df["Actual Class"] = actual_labels
results_df["Predicted Class"] = [encoded_to_label[int(value)] for value in y_pred]
results_df["Correct Prediction"] = results_df["Actual Class"] == results_df["Predicted Class"]
st.dataframe(results_df, use_container_width=True)
st.download_button(
    "Download predictions as CSV",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="dry_bean_predictions.csv",
    mime="text/csv",
)

st.divider()
st.caption("For a fair evaluation, upload a held-out test set that was not used to train the selected model.")
