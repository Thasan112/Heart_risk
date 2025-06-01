Heart Attack Prediction Model Improvement
This repository contains a Python script demonstrating techniques to improve the performance of a machine learning model, specifically addressing challenges posed by imbalanced datasets in the context of heart attack risk prediction.

Project Description
The provided dataset offers a comprehensive overview of various factors associated with heart attack risks, including patient medical history, lifestyle habits, and physiological measurements. The data is collected from multiple reputable medical studies and hospital records, ensuring a diverse and accurate representation.

Initial model evaluation revealed low precision, recall, and F1-score for Class 0 (the minority class, which often represents the positive case like 'heart attack' in such datasets). This project aims to demonstrate how to improve these metrics by employing strategies to handle class imbalance.

Key Findings (from initial analysis)
Multimodal Distribution: The dataset exhibits a multimodal distribution.

Data Split: The data is split into training (70%) and testing (30%) sets using train_test_split with random_state=42 for reproducibility.

Baseline Model Performance (for Class 0):

Precision: 0.24 (24%)

Recall: 0.25 (25%)

F1-score: 0.24 (24%)
These low scores indicate that the initial model struggles to correctly identify instances of Class 0.

How to Run the Script
Prerequisites
numpy
matplotlib
scikit-learn


