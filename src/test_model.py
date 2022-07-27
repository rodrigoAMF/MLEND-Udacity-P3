from joblib import load

import numpy as np

from lib.ml.model import inference, compute_model_metrics

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

model = load('model/model.joblib')
encoder = load('model/encoder.joblib')
lb = load('model/lb.joblib')

y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F-beta:    {fbeta:.4f}")
