# Script to train machine learning model.
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

from lib.ml.data import process_data
from lib.ml.model import train_model


# Add code to load in the data.
data = pd.read_csv("data/census.csv")

data = data.drop_duplicates()
data.columns = [column.strip() for column in data.columns]

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)

# Train a model.
model = train_model(X_train, y_train)

# Save model.
dump(model, 'model/model.joblib')
dump(encoder, 'model/encoder.joblib')
dump(lb, 'model/lb.joblib')
