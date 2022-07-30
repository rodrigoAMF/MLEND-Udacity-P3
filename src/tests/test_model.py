import os
import sys
import inspect
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(1, "../src")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from lib.ml.model import train_model  # noqa: E402


def test_train_model():
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_train = np.array([0, 1, 0])

    model = train_model(X_train, y_train)

    assert model is not None
    assert isinstance(model, lgb.LGBMClassifier)
