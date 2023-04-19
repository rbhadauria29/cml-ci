import shutil

import pandas as pd

from utils import delete_and_recreate_dir

DATASET_TYPES = ["test", "train"]
DROP_COLNAMES = ["feature_5"]

# Create a clean directory
delete_and_recreate_dir("processed_dataset")

for dataset_type in DATASET_TYPES:
    df = pd.read_csv(f"raw_dataset/{dataset_type}_features.csv")

    for col in DROP_COLNAMES:
        if col in df.columns:
            df = df.drop(columns=col)

    df.to_csv(f"processed_dataset/{dataset_type}_features.csv", index=None)
    shutil.copyfile(
        f"raw_dataset/{dataset_type}_labels.csv",
        f"processed_dataset/{dataset_type}_labels.csv",
    )
