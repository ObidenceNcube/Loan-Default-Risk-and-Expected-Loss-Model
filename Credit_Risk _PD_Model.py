
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from flask import Flask, request, jsonify


# Step 1: Data Preparation

#Load dataset into a DataFrame
df = pd.read_csv(r"C:\Users\_Loan_Data.csv")

#Dataset Exploration
df.head(5)

df.describe()

df.isnull()

df.info()


#  Step 2: Feature Selection and Engineering

#Define features and target variable
features = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
target = 'default'

#Split data into train and test sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 3: Model Selection
#XGBoost to be used as the main model

# Step 4: Model Training and Tuning

#Train an XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',  #Other evaluation metrics can be used as well
    use_label_encoder=False,  #To avoid warnings
    random_state=42)

xgb_model.fit(X_train, y_train)

#Predict probability of default for the test set
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]  # Probability of default


# Step 5: Model Evaluation

#Evaluate model performance
accuracy = accuracy_score(y_test, (y_pred_prob > 0.5).astype(int))
precision = precision_score(y_test, (y_pred_prob > 0.5).astype(int))
recall = recall_score(y_test, (y_pred_prob > 0.5).astype(int))
f1 = f1_score(y_test, (y_pred_prob > 0.5).astype(int))
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")

#Calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)

#Plot the calibration curve
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve', linestyle='--', color='b')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.show()

# Step 6: Expected Loss Calculation

#Define a function to calculate expected loss
def calculate_expected_loss(prob_default, outstanding_loan_amount, lgd=0.1):
    expected_loss = prob_default * outstanding_loan_amount * lgd
    return expected_loss

#Example usage of the function
customer_data = {
    'credit_lines_outstanding': 3,
    'loan_amt_outstanding': 150000,
    'total_debt_outstanding': 10000,
    'income': 60000,
    'years_employed': 5,
    'fico_score': 100
}

#Predict probability of default for a new customer
new_customer_data = pd.DataFrame([customer_data])
prob_default_new_customer = xgb_model.predict_proba(new_customer_data)[:, 1]

#Calculate expected loss for the new customer
outstanding_loan_amount = customer_data['loan_amt_outstanding']
expected_loss_new_customer = calculate_expected_loss(prob_default_new_customer, outstanding_loan_amount)

print(f"Predicted Probability of Default for New Customer: {prob_default_new_customer[0]:.2f}")
print(f"Expected Loss for New Customer: ${expected_loss_new_customer[0]:.2f}")


# Step 7: Model Deployment

#Initialize Flask app
app = Flask(__name__)

#Define a route to predict default probability and expected loss
@app.route('/predict', methods=['POST'])
def predict_default():
    try:
        #Get loan details from the request
        data = request.get_json()
        
        # Extract features for the new customer
        new_customer_data = pd.DataFrame([data])
        prob_default = xgb_model.predict_proba(new_customer_data)[:, 1][0]
        
        # Calculate expected loss
        outstanding_loan_amount = data['loan_amt_outstanding']
        lgd = 0.1  # Loss Given Default
        expected_loss = prob_default * outstanding_loan_amount * lgd

        # Prepare the response
        response = {
            'prob_default': prob_default,
            'expected_loss': expected_loss
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    #Run the Flask app on a specified host and port
    app.run(host='0.0.0.0', port=5000)





