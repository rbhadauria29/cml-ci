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


RFC_FOREST_DEPTH=5

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
clf = RandomForestClassifier(max_depth=RFC_FOREST_DEPTH)
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
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, normalize="true", cmap=plt.cm.Blues
)
plt.savefig("confusion_matrix.png")


header = "true_label,predicted_label"
np.savetxt(
    "predictions.csv",
    np.column_stack([y_test, y_pred]),
    delimiter=",",
    header=header,
    comments="",
    fmt="%i",
)
