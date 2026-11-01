# Machine Learning Assignment — Dry Bean Classification

## 1. Problem Statement

The objective of this project is to implement and compare five machine learning classification models to predict the variety of dry beans using their numerical geometric measurements.

The models are evaluated using six performance metrics: Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

This project also includes an interactive Streamlit web application that allows users to upload test data, select a classification model, and view prediction results and evaluation metrics.

## 2. Dataset Description

- **Dataset Name:** UCI Dry Bean Dataset
- **Dataset Source:** [UCI Machine Learning Repository — Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Problem Type:** Multiclass Classification
- **Total Instances:** 13,611
- **Input Features:** 16 numerical features
- **Target Variable:** Dry bean variety
- **Number of Classes:** 7

The dataset contains numerical measurements describing the geometric characteristics of dry beans. These features are used to classify the beans into their respective varieties.

## 3. GitHub Repository Link

[Machine Learning Assignment — GitHub Repository](https://github.com/2025ns04122-source/Machine_Learning_Assignment)

The repository contains the application source code, required Python dependencies, test dataset, saved machine learning models, and project documentation.

## 4. Models Used and Evaluation Metrics

The following five classification algorithms are implemented and evaluated using the same Dry Bean dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The following six evaluation metrics are used to compare their performance:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC Score:** Measures the model's ability to distinguish between different classes.
- **Precision:** Measures how many predicted positive classifications are correct.
- **Recall:** Measures how many actual positive classifications are correctly identified.
- **F1 Score:** Represents the harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC):** Measures the quality of classification predictions using a correlation-based score.

## 5. Live Streamlit Application

[Open the Live Dry Bean Classification Application](https://machinelearningassignment1-cl8iaoxzmrscz427jygq55.streamlit.app/)

The project is deployed using Streamlit Community Cloud.

The application provides an interactive interface for Dry Bean Classification using five trained machine learning models. Users can upload test data, select a model, generate predictions, and view the model's evaluation results.

### Application Features

- Upload test data in CSV format.
- Select one of the five available classification models.
- Generate predictions using the selected model.
- Display evaluation metrics.
- View the confusion matrix or classification report.

## 6. GitHub Repository Structure

The repository contains the following project files:

```text
Machine_Learning_Assignment/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── label_encoder.joblib
```

### Description of Important Files

| File | Description |
|---|---|
| `app.py` | Contains the Streamlit application code. |
| `requirements.txt` | Contains the Python libraries required to run the application. |
| `README.md` | Provides project details, model comparison, and performance observations. |
| `test_data.csv` | Contains the test data used for model evaluation and application testing. |
| `model/` | Contains the saved machine learning models and label encoder. |

## 7. Model Performance Comparison

All five classification models were evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The following table presents the evaluation results obtained from the experiments.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9948 | 0.9349 | 0.9314 | 0.9329 | 0.9042 |
| Decision Tree | 0.8920 | 0.9450 | 0.9075 | 0.9090 | 0.9081 | 0.8696 |
| kNN | 0.9166 | 0.9833 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Gaussian Naive Bayes | 0.7653 | 0.9631 | 0.7811 | 0.7684 | 0.7712 | 0.7154 |
| **Support Vector Machine** | **0.9221** | **0.9949** | **0.9359** | **0.9330** | **0.9343** | **0.9059** |

## 8. Model-Wise Observations

The performance of each classification model was analyzed based on the evaluation metrics obtained from the experiments.

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved 92.07% accuracy and an F1 score of 0.9329. It performed well as a linear classification model when features were scaled. |
| Decision Tree | Achieved 89.20% accuracy and an F1 score of 0.9081. It can capture non-linear relationships but showed lower test performance compared with the leading models. |
| kNN | Achieved 91.66% accuracy and an F1 score of 0.9293. It provided strong classification performance when distance calculations were performed on scaled features. |
| Gaussian Naive Bayes | Achieved 76.53% accuracy and an F1 score of 0.7712. It recorded the lowest overall performance among the five models. Correlations between geometric features may affect its feature-independence assumption. |
| Support Vector Machine | Achieved the highest accuracy of 92.21% and an F1 score of 0.9343. The RBF kernel helps the model capture complex, non-linear relationships between the input features. |

## 9. Overall Winner

**Support Vector Machine (SVM)** achieved the best overall performance among the five classification models.

Its evaluation results were:

- **Accuracy:** 0.9221 (92.21%)
- **AUC:** 0.9949
- **Precision:** 0.9359
- **Recall:** 0.9330
- **F1 Score:** 0.9343
- **MCC:** 0.9059

SVM achieved the highest value across all six evaluation metrics in the comparison table.

The RBF kernel allows SVM to capture non-linear relationships between the geometric features of dry beans, helping it achieve better classification performance than the other evaluated models.

## 10. Conclusion

In this project, five machine learning classification models were implemented and evaluated using the UCI Dry Bean Dataset.

The models were compared using six evaluation metrics to understand their classification performance.

Based on the experimental results:

- Support Vector Machine achieved the best overall performance.
- Logistic Regression and kNN also showed strong classification results.
- Decision Tree provided comparatively lower performance than the leading models.
- Gaussian Naive Bayes recorded the lowest performance among the five models.

Overall, Support Vector Machine with an RBF kernel was the best-performing model for the Dry Bean Classification task based on the reported evaluation results.

The Streamlit application provides an interactive way to upload test data, select a classification model, and view the corresponding prediction and evaluation results.
