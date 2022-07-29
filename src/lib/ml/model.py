import numpy as np
import lightgbm as lgb

from sklearn.metrics import fbeta_score, precision_score, recall_score

from lib.ml.data import process_data

# Optional: implement hyperparameter tuning.


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)

    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : lightgbm.LGBMClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def compute_model_metrics_slide(model, encoder, lb, data, cat_features,
                                column_slice="education"):

    unique_values = np.unique(data["education"].values)

    with open("slide_output.txt", 'w') as f:
        for value in unique_values:
            data_w_slide = data[data[column_slice].str.contains(value)]

            X, y, encoder, lb = process_data(
                data_w_slide, categorical_features=cat_features, label="salary", training=False,
                encoder=encoder, lb=lb
            )

            y_pred = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y, y_pred)

            print(f"Metrics using slice {column_slice} == {value}", file=f)
            print(f"Precision: {precision:.4f}", file=f)
            print(f"Recall:    {recall:.4f}", file=f)
            print(f"F-beta:    {fbeta:.4f}", file=f)
            print(file=f)
