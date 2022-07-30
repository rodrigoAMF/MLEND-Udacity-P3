# Script to train machine learning model.
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

from lib.ml.data import process_data
from lib.ml.model import (train_model, inference, compute_model_metrics,
                          compute_model_metrics_slide)


# Add code to load in the data.
data = pd.read_csv("data/census.csv")

data = data.drop_duplicates()
data.columns = [column.strip() for column in data.columns]

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

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
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train a model.
model = train_model(X_train, y_train)

# Save model.
dump(model, 'model/model.joblib')
dump(encoder, 'model/encoder.joblib')
dump(lb, 'model/lb.joblib')

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

compute_model_metrics_slide(model, encoder, lb, data, cat_features,
                            column_slice="education")
