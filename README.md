# Dry Bean Classifier Lab

Interactive Streamlit application for demonstrating five classification models on the UCI Dry Bean dataset.

## Features
- Upload a labelled CSV test set.
- Select Logistic Regression, Decision Tree, k-Nearest Neighbors, Gaussian Naive Bayes, or Support Vector Machine.
- Display Accuracy, multiclass macro one-vs-rest AUC, macro Precision, macro Recall, macro F1, and MCC.
- View a confusion matrix and classification report.
- Inspect predictions and download them as CSV.

## Dataset
UCI Dry Bean dataset: 13,611 instances, 16 numerical input features, and 7 bean classes.
Source: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

## Project structure
```text
dry-bean-streamlit-app/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── label_encoder.joblib
```

## Prepare model files
1. Run `Dry_Bean_Classification_Assignment.ipynb` from top to bottom.
2. Copy the five generated model files and `label_encoder.joblib` into `model/`.
3. Copy the generated `test_data.csv` into the project root.
4. Confirm all six model files and the test CSV are present before running or deploying.

The starter package does not contain fabricated model files. Generate them by running the notebook.

## Run locally
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push the project to your GitHub repository.
2. Ensure the model files and `test_data.csv` are committed.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Create a new app and select your repository, branch, and `app.py`.
5. Deploy and test the resulting app URL.

## CSV format
The uploaded file must contain all feature columns used in training and a `Class` column containing actual bean variety labels. Metrics cannot be computed for an unlabelled dataset.

## Metric definitions
- Accuracy: fraction of correct predictions.
- AUC: macro one-vs-rest ROC-AUC using class probabilities.
- Precision, Recall, F1: macro averages across classes.
- MCC: multiclass Matthews Correlation Coefficient.

## Academic integrity
Run the training notebook, inspect the actual results, write your own observations, maintain your own GitHub commit history, and capture your own BITS Virtual Lab screenshot.
