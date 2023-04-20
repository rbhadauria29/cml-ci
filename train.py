import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import json


dtypes_features = [
    ("feature_1", "float"),
    ("feature_2", "float"),
    ("feature_3", "float"),
    ("feature_4", "float"),
]
dtypes_labels = [("class_labels", "<i8")]
# Read in data
X_train = np.loadtxt(
    "processed_dataset/train_features.csv",
    skiprows=1,
    delimiter=",",
).astype(float)
y_train = np.loadtxt(
    "processed_dataset/train_labels.csv",
    skiprows=1,
    delimiter=",",
)
X_test = np.loadtxt(
    "processed_dataset/test_features.csv",
    skiprows=1,
    delimiter=",",
).astype(float)
y_test = np.loadtxt("processed_dataset/test_labels.csv", skiprows=1, delimiter=",")

# Fit a model
depth = 2
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy:{accuracy}, Precision: {precision}, Recall: {recall}, f1_score: {f1}")
with open("metrics.json", "w") as outfile:
    json.dump(
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        outfile,
    )

# Plot it
disp = ConfusionMatrixDisplay.from_estimator(
    clf, X_test, y_test, normalize="true", cmap=plt.cm.Blues
)
plt.savefig("confusion_matrix.png")
