from flask import Flask, jsonify, render_template, request
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "train.csv"))

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Could not find training data at: {TRAIN_PATH}")

raw_df = pd.read_csv(TRAIN_PATH)
model_df = raw_df.drop(columns=["id", "CustomerId", "Surname"], errors="ignore")

X = model_df.drop("Exited", axis=1)
y = model_df["Exited"]

X = pd.get_dummies(X, drop_first=True)
FEATURE_COLUMNS = X.columns

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

THRESHOLD = 0.65

y_val_prob = rf.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_prob >= THRESHOLD).astype(int)

ACCURACY = round(accuracy_score(y_val, y_val_pred), 4)
PRECISION = round(precision_score(y_val, y_val_pred), 4)
RECALL = round(recall_score(y_val, y_val_pred), 4)
ROC_AUC = round(roc_auc_score(y_val, y_val_prob), 4)
CONFUSION = confusion_matrix(y_val, y_val_pred).tolist()

_fpr, _tpr, _ = roc_curve(y_val, y_val_prob)


def percent(value):
    return round(float(value) * 100, 1)


def series_counts(column):
    counts = raw_df[column].value_counts().sort_index()
    return {
        "labels": [str(label) for label in counts.index.tolist()],
        "values": [int(value) for value in counts.tolist()],
    }


def churn_by_segment(column):
    grouped = raw_df.groupby(column)["Exited"].mean().sort_values(ascending=False)
    return {
        "labels": [str(label) for label in grouped.index.tolist()],
        "values": [percent(value) for value in grouped.tolist()],
    }


def histogram(column, bins):
    counts, edges = np.histogram(raw_df[column].dropna(), bins=bins)
    labels = [f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(len(edges) - 1)]
    return {"labels": labels, "values": [int(value) for value in counts.tolist()]}


def build_dashboard_data():
    feature_importance = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": rf.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(8)
    )

    validation_with_scores = raw_df.loc[y_val.index].copy()
    validation_with_scores["RiskScore"] = pd.Series(y_val_prob, index=y_val.index)

    risk_bands = pd.cut(
        validation_with_scores["RiskScore"],
        bins=[0, 0.25, 0.5, 0.65, 0.8, 1],
        labels=["0-25%", "25-50%", "50-65%", "65-80%", "80-100%"],
        include_lowest=True,
    ).value_counts().sort_index()

    return {
        "kpis": {
            "customers": int(len(raw_df)),
            "churn_rate": percent(raw_df["Exited"].mean()),
            "active_rate": percent(raw_df["IsActiveMember"].mean()),
            "avg_balance": round(float(raw_df["Balance"].mean()), 2),
            "avg_age": round(float(raw_df["Age"].mean()), 1),
            "avg_salary": round(float(raw_df["EstimatedSalary"].mean()), 2),
        },
        "model": {
            "accuracy": ACCURACY,
            "precision": PRECISION,
            "recall": RECALL,
            "roc_auc": ROC_AUC,
            "threshold": THRESHOLD,
            "confusion_matrix": CONFUSION,
        },
        "segments": {
            "geography_counts": series_counts("Geography"),
            "gender_counts": series_counts("Gender"),
            "product_counts": series_counts("NumOfProducts"),
            "churn_by_geography": churn_by_segment("Geography"),
            "churn_by_gender": churn_by_segment("Gender"),
            "churn_by_products": churn_by_segment("NumOfProducts"),
        },
        "distributions": {
            "age": histogram("Age", [18, 25, 35, 45, 55, 65, 75, 95]),
            "credit_score": histogram("CreditScore", [300, 450, 550, 650, 750, 850, 950]),
            "balance": histogram("Balance", [0, 25000, 75000, 125000, 175000, 225000, 275000]),
        },
        "feature_importance": {
            "labels": feature_importance["feature"].tolist(),
            "values": [
                round(float(value), 4)
                for value in feature_importance["importance"].tolist()
            ],
        },
        "risk_bands": {
            "labels": [str(label) for label in risk_bands.index.tolist()],
            "values": [int(value) for value in risk_bands.tolist()],
        },
    }


DASHBOARD_DATA = build_dashboard_data()

print(
    f"Model ready - Accuracy: {ACCURACY} | Precision: {PRECISION} | "
    f"Recall: {RECALL} | AUC: {ROC_AUC}"
)


@app.route("/")
def home():
    return render_template(
        "index.html",
        accuracy=ACCURACY,
        precision=PRECISION,
        recall=RECALL,
        roc_auc=ROC_AUC,
    )


@app.route("/dashboard_data")
def dashboard_data():
    return jsonify(DASHBOARD_DATA)


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
        "Geography_Spain": 1 if data["Geography"] == "Spain" else 0,
    }

    input_df = pd.DataFrame([customer])
    for col in FEATURE_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[FEATURE_COLUMNS]

    prob = rf.predict_proba(input_df)[0][1]
    prediction = "Churn" if prob >= THRESHOLD else "No Churn"

    return jsonify(
        {
            "churn_probability": round(float(prob), 4),
            "prediction": prediction,
            "threshold": THRESHOLD,
        }
    )


@app.route("/roc_data")
def roc_data():
    fpr = _fpr.tolist()
    tpr = _tpr.tolist()

    max_pts = 400
    if len(fpr) > max_pts:
        step = len(fpr) // max_pts
        fpr = fpr[::step]
        tpr = tpr[::step]
        if fpr[-1] != 1.0:
            fpr.append(1.0)
            tpr.append(1.0)

    return jsonify({"fpr": fpr, "tpr": tpr, "auc": ROC_AUC})


if __name__ == "__main__":
    app.run(debug=True)
