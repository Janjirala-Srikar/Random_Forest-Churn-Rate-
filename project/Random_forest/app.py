from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

# ===============================
# Flask App
# ===============================
app = Flask(__name__)

# ===============================
# PATH SETUP (🔥 CRITICAL FIX)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)   # ensure static/ exists

# ===============================
# LOAD & TRAIN MODEL (ONCE)
# ===============================
train_path = r"C:\Users\srika\Downloads\bank-customer-churn-prediction-2026\bank-customer-churn-prediction-2026\train.csv"

df = pd.read_csv(train_path)
df = df.drop(columns=["CustomerId", "Surname"])

X = df.drop("Exited", axis=1)
y = df["Exited"]

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)
FEATURE_COLUMNS = X.columns

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ===============================
# THRESHOLD
# ===============================
THRESHOLD = 0.65  # high precision focus

# ===============================
# MODEL METRICS (ONCE)
# ===============================
y_val_prob = rf.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_prob >= THRESHOLD).astype(int)

ACCURACY  = round(accuracy_score(y_val, y_val_pred), 4)
PRECISION = round(precision_score(y_val, y_val_pred), 4)
RECALL    = round(recall_score(y_val, y_val_pred), 4)
ROC_AUC   = round(roc_auc_score(y_val, y_val_prob), 4)

# ===============================
# ROC CURVE IMAGE (SAVE SAFELY)
# ===============================
fpr, tpr, _ = roc_curve(y_val, y_val_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {ROC_AUC}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Churn Prediction")
plt.legend()

roc_path = os.path.join(STATIC_DIR, "roc_curve.png")
plt.savefig(roc_path)
plt.close()

print("ROC curve saved at:", roc_path)

# ===============================
# ROUTES
# ===============================
@app.route("/")
def home():
    return render_template(
        "index.html",
        accuracy=ACCURACY,
        precision=PRECISION,
        recall=RECALL,
        roc_auc=ROC_AUC
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    customer = {
        "CreditScore": int(data["CreditScore"]),
        "Age": int(data["Age"]),
        "Tenure": int(data["Tenure"]),
        "Balance": float(data["Balance"]),
        "NumOfProducts": int(data["NumOfProducts"]),
        "HasCrCard": int(data["HasCrCard"]),
        "IsActiveMember": int(data["IsActiveMember"]),
        "EstimatedSalary": float(data["EstimatedSalary"]),
        "Gender_Male": 1 if data["Gender"] == "Male" else 0,
        "Geography_Germany": 1 if data["Geography"] == "Germany" else 0,
        "Geography_Spain": 1 if data["Geography"] == "Spain" else 0
    }

    input_df = pd.DataFrame([customer])

    for col in FEATURE_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[FEATURE_COLUMNS]

    prob = rf.predict_proba(input_df)[0][1]
    prediction = "Churn" if prob >= THRESHOLD else "No Churn"

    return jsonify({
        "churn_probability": round(float(prob), 4),
        "prediction": prediction,
        "threshold": THRESHOLD
    })


# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
