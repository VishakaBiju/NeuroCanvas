#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Emotion Model Testing / Validation Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize

# ---------------------------------------------------
# 1 LOAD TRAINED MODEL
# ---------------------------------------------------

print("Loading trained model...")

model = joblib.load("eeg_xgboost_model.pkl")
scaler = joblib.load("eeg_scaler.pkl")
selector = joblib.load("feature_selector.pkl")
selected_features = joblib.load("selected_features.pkl")

print("Model loaded successfully")

# ---------------------------------------------------
# 2 LOAD DATASET
# ---------------------------------------------------

path = "/serverdata/ccshome/anjanasinha/NAS/DreamData/DSU/preprocessed_v2.csv"

print("Loading test dataset...")
df = pd.read_csv(path)

print("Dataset shape:", df.shape)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# ---------------------------------------------------
# 3 SPLIT FEATURES / LABELS
# ---------------------------------------------------

X = df.drop("Emotion", axis=1)
y = df["Emotion"]

# ---------------------------------------------------
# 4 CLEAN DATA
# ---------------------------------------------------

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

mask = ~X.isna().any(axis=1)

X = X.loc[mask]
y = y.loc[mask]

print("Dataset after cleaning:", X.shape)

# ---------------------------------------------------
# 5 APPLY TRAINED SCALER
# ---------------------------------------------------

print("Applying scaler...")

X_scaled = scaler.transform(X)

# ---------------------------------------------------
# 6 APPLY FEATURE SELECTION
# ---------------------------------------------------

print("Selecting features...")

X_selected = selector.transform(X_scaled)

# ---------------------------------------------------
# 7 PREDICTIONS
# ---------------------------------------------------

print("Running predictions...")

preds = model.predict(X_selected)
probs = model.predict_proba(X_selected)

# ---------------------------------------------------
# 8 ACCURACY METRICS
# ---------------------------------------------------

accuracy = accuracy_score(y, preds)
balanced_acc = balanced_accuracy_score(y, preds)

print("\nTest Accuracy:", accuracy)
print("Balanced Accuracy:", balanced_acc)

print("\nClassification Report\n")
print(classification_report(y, preds))

# ---------------------------------------------------
# 9 CONFUSION MATRIX
# ---------------------------------------------------

cm = confusion_matrix(y, preds)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("test_confusion_matrix.png", dpi=300)

plt.show()

# ---------------------------------------------------
# 10 ROC CURVES
# ---------------------------------------------------

classes = np.unique(y)

y_bin = label_binarize(y, classes=classes)

plt.figure(figsize=(7,6))

for i in range(len(classes)):

    fpr, tpr, _ = roc_curve(y_bin[:,i], probs[:,i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"Class {classes[i]} AUC={roc_auc:.2f}")

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curves")

plt.legend()

plt.savefig("test_roc_curves.png", dpi=300)

plt.show()

# ---------------------------------------------------
# 11 SAVE PREDICTIONS
# ---------------------------------------------------

results = pd.DataFrame({
    "Actual": y,
    "Predicted": preds,
    "Prob_Confusion": probs[:,0],
    "Prob_Happiness": probs[:,1],
    "Prob_Sadness": probs[:,2]
})

results.to_csv("emotion_predictions.csv", index=False)

print("Predictions saved to emotion_predictions.csv")

# ---------------------------------------------------
# 12 SAVE MISCLASSIFIED SAMPLES
# ---------------------------------------------------

misclassified = results[results["Actual"] != results["Predicted"]]

misclassified.to_csv("misclassified_samples.csv", index=False)

print("Misclassified samples saved:", len(misclassified))

# ---------------------------------------------------
# 13 SAMPLE OUTPUT
# ---------------------------------------------------

print("\nSample predictions:")

print(results.head())