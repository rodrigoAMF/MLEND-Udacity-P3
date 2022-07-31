import pandas as pd
from joblib import load

from lib.ml.data import process_data
from lib.ml.model import (inference, compute_model_metrics,
                          compute_model_metrics_slide)

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

encoder = load('model/encoder.joblib')
lb = load('model/lb.joblib')
model = load('model/model.joblib')

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
    train_data, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test_data, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

y_pred = inference(model, X_train)
precision, recall, fbeta = compute_model_metrics(y_train, y_pred)

print("Train set Metrics")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F-beta:    {fbeta:.4f}")

print()

y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

print("Test set Metrics")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F-beta:    {fbeta:.4f}")
