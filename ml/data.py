import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler

# Education mapping dictionary
edu_map = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-acdm": 11,
    "Assoc-voc": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Doctorate": 15,
    "Prof-school": 16
}

def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None, scaler=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """
    # Apply ordinal mapping to education
    if "education" in X.columns:
        X = X.copy()
        X["education"] = X["education"].map(edu_map)

    # Separate categorical and numeric features
    X_cat = X[categorical_features]
    drop_cols = categorical_features + ([label] if label else [])
    X_num = X.drop(columns=drop_cols)


    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        scaler = StandardScaler()

        X_cat_enc = encoder.fit_transform(X_cat)
        X_num_scaled = scaler.fit_transform(X_num)

        y = lb.fit_transform(X[label]) if label else None
    else:
        X_cat_enc = encoder.transform(X_cat)
        X_num_scaled = scaler.transform(X_num)
        y = lb.transform(X[label]) if label else None

    # Concatenate numeric + categorical features
    X_out = np.concatenate([X_num_scaled, X_cat_enc], axis=1)

    return X_out, y, encoder, lb, scaler

def apply_label(inference):
    """ Convert the binary label in a single inference sample into string output."""
    if inference[0] == 1:
        return ">50K"
    elif inference[0] == 0:
        return "<=50K"
