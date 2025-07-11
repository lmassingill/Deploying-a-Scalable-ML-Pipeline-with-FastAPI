import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics

def test_train_model_type():
    """
    # Checks that the train_model() function returns a RandomForestClassifier 
    # instance — verifying the correct ML algorithm is used.
    """
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size = 10)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "Model is not RandomForestClassifier"

def test_inference_output_shape():
    """
    # Verifies that the shape of the predictions from inference() matches the 
    # shape of the input labels — ensuring the model outputs the correct number of predictions.
    """
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size = 10)
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == y.shape, "Predictions shape does not match label shape"

def test_compute_model_metrics_output():
    """
    # Tests that compute_model_metrics() returns precision, recall, and F1 values all 
    # within the valid range [0, 1] — confirming the metric function behaves as expected.
    """
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
