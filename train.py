import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay


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

acc = clf.score(X_test, y_test)
print(f"Accuracy:{acc}")
with open("metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy: {acc}\n")

# Plot it
disp = ConfusionMatrixDisplay.from_estimator(
    clf, X_test, y_test, normalize="true", cmap=plt.cm.Blues
)
plt.savefig("plot.png")
