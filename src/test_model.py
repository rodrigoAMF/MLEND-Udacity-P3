import lightgbm as lgb
import numpy as np

from lib.ml.model import train_model, inference, compute_model_metrics


def test_train_model():
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_train = np.array([0, 1, 0])

    model = train_model(X_train, y_train)

    assert model is not None
    assert isinstance(model, lgb.LGBMClassifier)


def test_inference():
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_train = np.array([0, 1, 0])

    model = train_model(X_train, y_train)

    preds = inference(model, X_test)

    assert preds is not None
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (3,)


def test_compute_metrics():
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_train = np.array([0, 1, 0])
    y_test = np.array([0, 0, 1])

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    metrics = compute_model_metrics(y_test, preds)

    assert metrics is not None
    assert len(metrics) == 3
    assert isinstance(metrics[0], float)    # precision
    assert isinstance(metrics[1], float)    # recall
    assert isinstance(metrics[2], float)    # fbeta