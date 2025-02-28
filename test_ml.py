# test_ml.py

import pytest
import numpy as np
import pandas as pd

from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data


@pytest.fixture
def sample_data():
    """
    Create a small DataFrame that resembles your real data.
    Adjust columns to match your actual feature names.
    """
    data_dict = {
        "age": [25, 38],
        "workclass": ["Private", "Private"],
        "fnlgt": [123456, 654321],
        "education": ["Bachelors", "HS-grad"],
        "education-num": [13, 9],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Prof-specialty", "Handlers-cleaners"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "Black"],
        "sex": ["Male", "Male"],
        "capital-gain": [0, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 35],
        "native-country": ["United-States", "United-States"],
        "salary": [">50K", "<=50K"],
    }
    df = pd.DataFrame(data_dict)
    return df


def test_train_model(sample_data):
    """
    Test that train_model returns a trained model and that the model
    can predict without errors.
    """
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
    # Use the entire sample_data for training so that both classes are present.
    X_train, y_train, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model = train_model(X_train, y_train)
    assert model is not None, "train_model() returned None"
    preds = inference(model, X_train)
    # Ensure predictions have the same number of samples as the training set.
    assert len(preds) == X_train.shape[0], (
        "Predictions do not match the number of training samples"
    )


def test_inference_output_shape(sample_data):
    """
    Test that the inference function returns an array of the correct shape.
    """
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
    X_train, y_train, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert preds.shape[0] == X_train.shape[0], (
        "Number of predictions does not match number of samples"
    )


def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns correct precision, recall,
    and fbeta for perfect predictions.
    """
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 1.0, "Precision should be 1.0 for perfect predictions"
    assert recall == 1.0, "Recall should be 1.0 for perfect predictions"
    assert fbeta == 1.0, "F-beta should be 1.0 for perfect predictions"
