import os

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


from utils import delete_and_recreate_dir

# Generate data
seed = 1993
X, y = make_classification(
    n_samples=1000,
    random_state=seed,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_repeated=1,
)

# Make a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

# Create fresh directory
delete_and_recreate_dir("raw_dataset")

header = "feature_1,feature_2,feature_3,feature_4,feature_5"
np.savetxt(
    "raw_dataset/train_features.csv", X_train, delimiter=",", header=header, comments=""
)
np.savetxt(
    "raw_dataset/test_features.csv", X_test, delimiter=",", header=header, comments=""
)
np.savetxt(
    "raw_dataset/train_labels.csv", y_train, fmt="%i", header="class_label", comments=""
)
np.savetxt(
    "raw_dataset/test_labels.csv", y_test, fmt="%i", header="class_label", comments=""
)
