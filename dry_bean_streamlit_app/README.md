# Machine Learning Assignment — Dry Bean Classification

## a. Problem Statement

The objective of this project is to implement and compare five machine-learning classification models to predict the variety of dry beans using their numerical geometric measurements. The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## b. Dataset Description

- **Dataset:** UCI Dry Bean Dataset
- **Source:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- **Problem type:** Multiclass classification
- **Instances:** 13,611 in the original dataset
- **Input features:** 16 numerical features
- **Target:** Dry bean variety, comprising seven classes

## c. GitHub Repository Link

https://github.com/2025ns04122-source/Machine_Learning_Assignment

## d. Models Used and Evaluation Metrics

The following five classification models are evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## e. Final Evaluation Summary

The performance of all five classification models was evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9948 | 0.9349 | 0.9314 | 0.9329 | 0.9042 |
| Decision Tree | 0.8920 | 0.9450 | 0.9075 | 0.9090 | 0.9081 | 0.8696 |
| kNN | 0.9166 | 0.9833 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Naive Bayes | 0.7653 | 0.9631 | 0.7811 | 0.7684 | 0.7712 | 0.7154 |
| Support Vector Machine | **0.9221** | **0.9949** | **0.9359** | **0.9330** | **0.9343** | **0.9059** |

## f. Model-Wise Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Strong linear baseline. Benefits heavily from feature scaling to achieve 92.07% accuracy. |
| Decision Tree | Captures non-linear splits but tends to overfit, yielding a lower F1 score of 0.9081. |
| kNN | Strong performance (91.66% accuracy) when distance metrics are computed on scaled features. |
| Naive Bayes | Weakest performer due to strong, unrealistic feature independence assumptions on physical dimensions. |
| Support Vector Machine | **Overall Winner**; leverages the RBF kernel to effectively handle highly non-linear decision boundaries. |

## g. Overall Winner

Support Vector Machine (SVM) achieved the best overall performance, with an accuracy of 92.21%, AUC of 0.9949, precision of 0.9359, recall of 0.9330, F1 score of 0.9343, and MCC of 0.9059.

Based on these evaluation results, SVM performed better than the other four classification models on the Dry Bean dataset.
13. Model-Wise Observations
Logistic Regression: Accuracy 92.07% | F1 0.9329 — Excellent, fast linear baseline when scaled.
Decision Tree: Accuracy 89.20% | F1 0.9081 — Strong performance, but prone to minor overfitting.
kNN: Accuracy 91.66% | F1 0.9293 — Highly competitive distance-based approach with scaling.
Gaussian Naive Bayes: Accuracy 76.53% | F1 0.7712 — Lagged due to feature dependency violations.
Support Vector Machine: Accuracy 92.21% | F1 0.9343 — Overall Winner; best at capturing complex boundaries with RBF kernel.

Overall Winner: Support Vector Machine (SVM)
Top Metrics achieved: Accuracy: 92.21% | Macro F1-Score: 0.9343 | MCC: 0.9059
Why it won: By using an RBF kernel, the SVM successfully mapped the 16 highly non-linear geometric features of the beans into a higher-dimensional space to find a superior decision boundary, outperforming all other linear and non-linear classifiers.

## h. Live Streamlit Application

https://machinelearningassignment1-cl8iaoxzmrscz427jygq55.streamlit.app/

This project is a Dry Bean Classification web application that uses five machine-learning models to predict dry bean varieties from 16 numerical features. It allows users to upload test data, compare model performance, and evaluate prediction accuracy to identify the most suitable classifier.

## i. Repository Contents

The repository structure includes:

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

## j. Conclusion

Final Conclusion & Performance Summary
Based on the empirical results of evaluating five classification models on the Dry Bean Dataset, here is the final comparison:

Key Metrics Comparison
Support Vector Machine (SVM): Overall Winner (Accuracy: 92.21% | Macro F1: 0.9343 | MCC: 0.9059)
Logistic Regression: Best Linear Baseline (Accuracy: 92.07% | Macro F1: 0.9329 | MCC: 0.9042)
k-Nearest Neighbors (kNN): Strong Distance Baseline (Accuracy: 91.66% | Macro F1: 0.9293 | MCC: 0.8992)
Decision Tree: Underperformed due to overfitting (Accuracy: 89.20% | Macro F1: 0.9081 | MCC: 0.8696)
Gaussian Naive Bayes: Weakest model due to features dependency (Accuracy: 76.53% | Macro F1: 0.7712 | MCC: 0.7154)
Core Takeaways for Marks
Winner Selection: The SVM with an RBF kernel achieved top-tier predictive power by mapping non-linear geometric features into a separable high-dimensional space.
Preprocessing Impact: Scaling with StandardScaler was mandatory for distance and gradient-based algorithms (SVM, Logistic Regression, kNN), directly leading to their superior ~92% accuracy.
Feature Correlation Impact: Gaussian Naive Bayes failed significantly (76.53% accuracy) because its core assumption of feature independence is violated by the highly correlated physical dimensions of the dry beans (e.g., Area, Perimeter, and Compactness).


