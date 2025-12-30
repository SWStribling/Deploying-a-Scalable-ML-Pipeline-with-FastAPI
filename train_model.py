import os

import pandas as pd
from sklearn.model_selection import train_test_split, KFold

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
# TODO: load the cencus.csv data
project_path = "C:\\Users\\sstri\\OneDrive\\Documents\\School\\WGU\\D501-Machine_Learning_DevOps\\FastAPI"
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path) # your code here

# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# train, test = None, None# Your code here

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

metrics = []
fold = 1

for train_index, test_index in kf.split(data):
    train = data.iloc[train_index]
    test = data.iloc[test_index]

# TODO: use the process_data function provided to process the data.
    X_train, y_train, encoder, lb = process_data(
        # your code here
        # use the train dataset 
        # use training=True
        # do not need to pass encoder and lb as input
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
        )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # TODO: use the train_model function to train the model on the training dataset
    model = train_model(X_train, y_train) # your code here

    # TODO: use the inference function to run the model inferences on the test dataset.
    preds = inference(model, X_test) # your code here

    # Calculate and print the metrics
    p, r, fb = compute_model_metrics(y_test, preds)
    metrics.append((p, r, fb))

    print(f"Fold {fold}: Precision={p:.4f}, Recall={r:.4f}, F1={fb:.4f}")

    fold += 1

# average metrics across folds
avg_p = sum([m[0] for m in metrics]) / len(metrics)
avg_r = sum([m[1] for m in metrics]) / len(metrics)
avg_fb = sum([m[2] for m in metrics]) / len(metrics)

print(f"Average across folds: Precision={avg_p:.4f}, Recall={avg_r:.4f}, F1={avg_fb:.4f}")

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            # your code here
            # use test, col and slicevalue as part of the input
            test,
            column_name=col,
            slice_value=slicevalue,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
