# Loan-Default-Risk-and-Expected-Loss-Model
This repository contains a Python script for building, evaluating, and deploying a credit risk model using XGBoost. The model predicts the probability of loan default and calculates the Expected Loss (EL) for a given customer, which is a key metric in finance for risk management.

The project follows a standard machine learning workflow:
1.	Data Preparation: Load and inspect the loan dataset.
2.	Model Training: Train an XGBoost classifier to predict the binary outcome (default).
3.	Model Evaluation: Assess performance using standard metrics (Accuracy, ROC-AUC, F1 Score) and visualize model calibration.
4.	Risk Quantification: Calculate the Expected Loss (EL) using the formula: 
EL = P(Default) \times EAD \times LGD
5.	Deployment: Deploy the trained model as a REST API using Flask for real-time predictions.

