import lightgbm as lgb
import numpy as np

from lib.ml.model import train_model


def test_train_model():
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_train = np.array([0, 1, 0])

    model = train_model(X_train, y_train)

    assert model is not None
    assert isinstance(model, lgb.LGBMClassifier)
