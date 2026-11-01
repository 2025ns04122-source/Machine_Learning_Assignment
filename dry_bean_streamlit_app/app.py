from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


# --------------------------------------------------
# 1. PROJECT FOLDERS AND MODEL FILES
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "k-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


# --------------------------------------------------
# 2. PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classifier Lab",
    page_icon="🫘",
    layout="wide",
)

st.title("🫘 Dry Bean Classifier Lab")

st.write(
    "Explore five machine-learning classifiers trained to predict "
    "dry bean variety from 16 numerical measurements. Upload a "
    "labelled CSV test file to compare predictions with the known "
    "class labels."
)


# --------------------------------------------------
# 3. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("Model settings")

    selected_model_name = st.selectbox(
        "Choose a classifier",
        list(MODEL_FILES.keys()),
    )

    st.divider()

    st.caption("Expected target column: Class")
    st.caption("Upload test data only; do not upload training data.")


# --------------------------------------------------
# 4. LOAD THE SELECTED MODEL
# --------------------------------------------------

def load_model(filename):
    model_path = MODEL_DIR / filename

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


try:
    model = load_model(
        MODEL_FILES[selected_model_name]
    )

except Exception as exc:
    st.error(f"Could not load the selected model: {exc}")

    st.write("The app is checking this model folder:")

    st.code(str(MODEL_DIR))

    st.write("Files currently available in the model folder:")

    if MODEL_DIR.is_dir():
        available_files = [
            file.name for file in MODEL_DIR.iterdir()
            if file.is_file()
        ]

        st.write(available_files)
    else:
        st.warning("The model folder does not exist.")

    st.info(
        "Check that the required model file is uploaded to "
        "the model folder in your GitHub repository."
    )

    st.stop()


# --------------------------------------------------
# 5. LOAD THE LABEL ENCODER
# --------------------------------------------------

label_encoder_path = MODEL_DIR / "label_encoder.joblib"

try:
    if not label_encoder_path.is_file():
        raise FileNotFoundError(
            f"Label encoder not found: {label_encoder_path}"
        )

    label_encoder = joblib.load(label_encoder_path)

except Exception as exc:
    st.error(f"Could not load the label encoder: {exc}")
    st.stop()


class_names = [
    str(name) for name in label_encoder.classes_
]

label_to_encoded = {
    name: index
    for index, name in enumerate(class_names)
}

encoded_to_label = {
    index: name
    for index, name in enumerate(class_names)
}


# --------------------------------------------------
# 6. IDENTIFY THE FEATURE COLUMNS
# --------------------------------------------------

if hasattr(model, "feature_names_in_"):

    feature_columns = list(model.feature_names_in_)

elif (
    hasattr(model, "named_steps")
    and hasattr(
        model.named_steps.get("classifier"),
        "feature_names_in_",
    )
):

    feature_columns = list(
        model.named_steps["classifier"].feature_names_in_
    )

else:
    st.error(
        "Could not identify the trained model's feature names. "
        "Please check the training notebook."
    )
    st.stop()


# --------------------------------------------------
# 7. LOAD THE EXAMPLE TEST DATA
# --------------------------------------------------

@st.cache_data
def load_sample_data():
    sample_path = APP_DIR / "test_data.csv"

    if sample_path.is_file():
        return pd.read_csv(sample_path)

    return None


sample_data = load_sample_data()


# --------------------------------------------------
# 8. UPLOAD TEST DATA
# --------------------------------------------------

with st.expander("CSV requirements", expanded=False):

    st.markdown(
        """
        - Include all 16 numerical feature columns used during training.
        - Include a `Class` column with the actual bean variety labels.
        - Evaluation metrics require known labels.
        - Upload a separate test dataset, not the training dataset.
        """
    )


uploaded_file = st.file_uploader(
    "Upload labelled test data (CSV)",
    type=["csv"],
)


if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)
        source_label = uploaded_file.name

    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()

elif sample_data is not None:

    st.info(
        "No file uploaded. Showing the bundled example test data."
    )

    data = sample_data.copy()
    source_label = "Bundled example test data"

    st.download_button(
        "Download example test_data.csv",
        data=sample_data.to_csv(index=False).encode("utf-8"),
        file_name="test_data.csv",
        mime="text/csv",
    )

else:

    st.warning(
        "Upload a labelled test CSV to begin. "
        "Add the notebook-generated test_data.csv "
        "to the project root to enable the example."
    )

    st.stop()


# --------------------------------------------------
# 9. SHOW THE TEST DATA
# --------------------------------------------------

st.subheader("Test data preview")

st.caption(
    f"Source: {source_label} | "
    f"{len(data):,} rows | "
    f"{len(data.columns):,} columns"
)

st.dataframe(
    data.head(10),
    use_container_width=True,
)


# --------------------------------------------------
# 10. CHECK THE CSV DATA
# --------------------------------------------------

if "Class" not in data.columns:

    st.error("Required target column 'Class' was not found.")
    st.stop()


missing_features = [
    column
    for column in feature_columns
    if column not in data.columns
]

if missing_features:

    st.error(
        "Missing required feature columns: "
        + ", ".join(missing_features)
    )

    st.stop()


if data.empty:

    st.error("The CSV has no data rows.")
    st.stop()


# --------------------------------------------------
# 11. PREPARE THE FEATURES
# --------------------------------------------------

X_test = data[feature_columns].copy()

try:

    X_test = X_test.apply(
        pd.to_numeric,
        errors="raise",
    )

except Exception:

    st.error(
        "All feature columns must contain numeric values."
    )

    st.stop()


if X_test.isna().any().any():

    st.error(
        "Missing values were found in feature columns. "
        "Clean the test data and upload again."
    )

    st.stop()


# --------------------------------------------------
# 12. PREPARE THE ACTUAL CLASS LABELS
# --------------------------------------------------

actual_labels = (
    data["Class"]
    .astype(str)
    .str.strip()
)

unknown_labels = sorted(
    set(actual_labels) - set(class_names)
)

if unknown_labels:

    st.error(
        "Unknown class labels found: "
        + ", ".join(unknown_labels)
    )

    st.stop()


y_true = actual_labels.map(
    label_to_encoded
).to_numpy()


# --------------------------------------------------
# 13. MAKE PREDICTIONS AND CALCULATE METRICS
# --------------------------------------------------

def calculate_metrics(model, X, y_true):

    y_pred = model.predict(X)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision (macro)": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "Recall (macro)": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "F1 (macro)": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "MCC": matthews_corrcoef(
            y_true,
            y_pred,
        ),
    }

    try:

        y_scores = model.predict_proba(X)

        estimator = (
            model.named_steps["classifier"]
            if hasattr(model, "named_steps")
            else model
        )

        metrics["AUC (macro OVR)"] = roc_auc_score(
            y_true,
            y_scores,
            multi_class="ovr",
            average="macro",
            labels=estimator.classes_,
        )

    except (
        ValueError,
        TypeError,
        AttributeError,
    ):

        metrics["AUC (macro OVR)"] = np.nan

    return y_pred, metrics


try:

    y_pred, metrics = calculate_metrics(
        model,
        X_test,
        y_true,
    )

except Exception as exc:

    st.error(
        f"Prediction or metric calculation failed: {exc}"
    )

    st.stop()


# --------------------------------------------------
# 14. DISPLAY EVALUATION RESULTS
# --------------------------------------------------

st.subheader("Evaluation results")

st.caption(
    f"Selected model: {selected_model_name} | "
    f"{len(data):,} test rows"
)

metric_order = [
    "Accuracy",
    "AUC (macro OVR)",
    "Precision (macro)",
    "Recall (macro)",
    "F1 (macro)",
    "MCC",
]

cols = st.columns(3)

for index, metric_name in enumerate(metric_order):

    value = metrics.get(metric_name, np.nan)

    cols[index % 3].metric(
        metric_name,
        f"{value:.4f}" if pd.notna(value) else "N/A",
    )


st.caption(
    "Precision, Recall and F1 use macro averaging. "
    "AUC is macro one-vs-rest from class probabilities. "
    "MCC is multiclass."
)


# --------------------------------------------------
# 15. CONFUSION MATRIX
# --------------------------------------------------

left, right = st.columns([1.05, 1])

with left:

    st.subheader("Confusion matrix")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    ax.tick_params(
        axis="y",
        rotation=0,
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# --------------------------------------------------
# 16. CLASSIFICATION REPORT
# --------------------------------------------------

with right:

    st.subheader("Classification report")

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose().round(4)

    st.dataframe(
        report_df,
        use_container_width=True,
    )


# --------------------------------------------------
# 17. SHOW AND DOWNLOAD PREDICTIONS
# --------------------------------------------------

st.subheader("Predictions")

results_df = data.copy()

results_df["Actual Class"] = actual_labels

results_df["Predicted Class"] = [
    encoded_to_label[int(value)]
    for value in y_pred
]

results_df["Correct Prediction"] = (
    results_df["Actual Class"]
    == results_df["Predicted Class"]
)

st.dataframe(
    results_df,
    use_container_width=True,
)


st.download_button(
    "Download predictions as CSV",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="dry_bean_predictions.csv",
    mime="text/csv",
)


# --------------------------------------------------
# 18. FINAL MESSAGE
# --------------------------------------------------

st.divider()

st.caption(
    "For a fair evaluation, upload a held-out test set "
    "that was not used to train the selected model."
)
