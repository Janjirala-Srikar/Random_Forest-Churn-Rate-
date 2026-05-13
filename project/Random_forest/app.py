from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

# ==========================================
# FLASK APP
# ==========================================
app = Flask(__name__)

# ==========================================
# DIRECTORY SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# project root folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(STATIC_DIR, exist_ok=True)

# ==========================================
# DATASET PATHS
# ==========================================
train_path = os.path.join(PROJECT_ROOT, "train.csv")

test_path = os.path.join(PROJECT_ROOT, "test.csv")

# ==========================================
# LOAD TRAIN DATA
# ==========================================
train_df = pd.read_csv(train_path)

# remove unwanted columns
train_df = train_df.drop(columns=["CustomerId", "Surname"])

# features and target
X_train = train_df.drop("Exited", axis=1)

y_train = train_df["Exited"]

# ==========================================
# ENCODING
# ==========================================
X_train = pd.get_dummies(X_train, drop_first=True)

# save columns
FEATURE_COLUMNS = X_train.columns

# ==========================================
# RANDOM FOREST MODEL
# ==========================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# train model
rf.fit(X_train, y_train)

# ==========================================
# THRESHOLD
# ==========================================
THRESHOLD = 0.65

# ==========================================
# DEFAULT METRICS
# ==========================================
ACCURACY = "N/A"
PRECISION = "N/A"
RECALL = "N/A"
ROC_AUC = "N/A"

# ==========================================
# LOAD TEST DATA
# ==========================================
test_df = pd.read_csv(test_path)

# remove unwanted columns
test_df = test_df.drop(columns=["CustomerId", "Surname"])

# ==========================================
# CHECK IF TEST HAS TARGET COLUMN
# ==========================================
if "Exited" in test_df.columns:

    X_test = test_df.drop("Exited", axis=1)

    y_test = test_df["Exited"]

    # encode test data
    X_test = pd.get_dummies(X_test, drop_first=True)

    # match training columns
    for col in FEATURE_COLUMNS:
        if col not in X_test.columns:
            X_test[col] = 0

    X_test = X_test[FEATURE_COLUMNS]

    # ==========================================
    # PREDICTIONS
    # ==========================================
    y_prob = rf.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= THRESHOLD).astype(int)

    # ==========================================
    # METRICS
    # ==========================================
    ACCURACY = round(accuracy_score(y_test, y_pred), 4)

    PRECISION = round(precision_score(y_test, y_pred), 4)

    RECALL = round(recall_score(y_test, y_pred), 4)

    ROC_AUC = round(roc_auc_score(y_test, y_prob), 4)

    # ==========================================
    # DYNAMIC ROC CURVE
    # ==========================================
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC AUC = {ROC_AUC}"
    )

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve - Customer Churn Prediction")

    plt.legend()

    roc_path = os.path.join(STATIC_DIR, "roc_curve.png")

    plt.savefig(roc_path, bbox_inches="tight")

    plt.close()

    print("ROC curve saved at:", roc_path)

else:

    print("test.csv has no Exited column")

    print("ROC curve cannot be generated")

# ==========================================
# HOME ROUTE
# ==========================================
@app.route("/")
def home():

    return render_template(
        "index.html",
        accuracy=ACCURACY,
        precision=PRECISION,
        recall=RECALL,
        roc_auc=ROC_AUC
    )

# ==========================================
# PREDICTION ROUTE
# ==========================================
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    customer = {

        "CreditScore":
            int(data["CreditScore"]),

        "Age":
            int(data["Age"]),

        "Tenure":
            int(data["Tenure"]),

        "Balance":
            float(data["Balance"]),

        "NumOfProducts":
            int(data["NumOfProducts"]),

        "HasCrCard":
            int(data["HasCrCard"]),

        "IsActiveMember":
            int(data["IsActiveMember"]),

        "EstimatedSalary":
            float(data["EstimatedSalary"]),

        "Gender_Male":
            1 if data["Gender"] == "Male" else 0,

        "Geography_Germany":
            1 if data["Geography"] == "Germany" else 0,

        "Geography_Spain":
            1 if data["Geography"] == "Spain" else 0
    }

    # convert to dataframe
    input_df = pd.DataFrame([customer])

    # add missing columns
    for col in FEATURE_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = 0

    # arrange columns
    input_df = input_df[FEATURE_COLUMNS]

    # predict probability
    prob = rf.predict_proba(input_df)[0][1]

    prediction = (
        "Churn"
        if prob >= THRESHOLD
        else "No Churn"
    )

    return jsonify({
        "churn_probability": round(float(prob), 4),
        "prediction": prediction,
        "threshold": THRESHOLD
    })

# ==========================================
# RUN APP
# ==========================================
if __name__ == "__main__":

    app.run(debug=True)