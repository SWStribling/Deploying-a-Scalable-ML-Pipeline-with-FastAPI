import pytest
# TODO: add necessary import
import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference, performance_on_categorical_slice


# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    # add description for the first test
     - returns 5 objects: X, y, encoder, lb, scaler
     - encoded categorical features expand correctly
     - numeric features are scaled
    """
    # Your code here
    sample = pd.DataFrame({
        "age": [25, 40],
        "workclass": ['Private', 'Self-emp'],
        "education": ['Bachelors', 'Masters'],
        "salary": ['<=50K', '>50K']
    })
    cat_features = [
        "workclass",
        "education"
    ]

    X, y, encoder, lb, scaler = process_data(
        sample, categorical_features=cat_features, label="salary", training=True
    )

    assert X.shape[0] == 2 # x check number of rows
    assert y.shape[0] == 2 # y check number of rows

    # Check that categorical features are one-hot encoded
    expected_cat_dims = sum(
        len(sample[feature].unique()) for feature in cat_features
    )
    assert X.shape[1] == expected_cat_dims + 1  # +1 for the numeric age feature

    assert encoder is not None
    assert scaler is not None


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    - train_model returns a fitted model
    - inference returns predictions of correct shape
    - predictions are binary

    """
    # Your code here
    sample = pd.DataFrame({
        "age": [30, 50, 22],
        "workclass": ["Private", "Private", "Self-emp"],
        "education": ["Bachelors", "Masters", "HS-grad"],
        "salary": ["<=50K", ">50K", "<=50K"]
    })

    X, y, encoder, lb, scaler = process_data(
        sample,
        categorical_features=["workclass", "education"],
        label="salary",
        training=True
    )

    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})



# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    - performance_on_categorical_features works end to end
    - returns valid metric values (floats)
    """
    # Your code here
    sample = pd.DataFrame({
        "age": [25, 45, 33, 60],
        "workclass": ["Private", "Private", "Self-emp", "Private"],
        "education": ["Bachelors", "HS-grad", "HS-grad", "Masters"],
        "salary": ["<=50K", ">50K", "<=50K", ">50K"]
    })

    X, y, encoder, lb, scaler = process_data(
        sample,
        categorical_features=["workclass", "education"],
        label="salary",
        training=True
    )

    model = train_model(X, y)

    p, r, f1 = performance_on_categorical_slice(
        sample,
        column_name="workclass",
        slice_value="Private",
        categorical_features=["workclass", "education"],
        label="salary",
        encoder=encoder,
        lb=lb,
        model=model,
        scaler=scaler
    )

    assert isinstance(p, float)
    assert isinstance(r, float)
    assert isinstance(f1, float)

