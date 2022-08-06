import os

from fastapi import FastAPI
import pandas as pd
from joblib import load
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field

from lib.ml.data import process_data
from lib.ml.model import inference


# Instantiate the FastAPI object
app = FastAPI()

absolute_path = os.path.dirname(os.path.abspath(__file__))
app.encoder = load(os.path.join(absolute_path, 'model/encoder.joblib'))
app.lb = load(os.path.join(absolute_path, 'model/lb.joblib'))
app.model = load(os.path.join(absolute_path, 'model/model.joblib'))

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


class Features(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age": 26,
                "workclass": "Local-gov",
                "fnlgt": 283217,
                "education": "Some-college",
                "education-num": 10,
                "marital-status": "Never-married",
                "occupation": "Protective-serv",
                "relationship": "Own-child",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }


# Define a route
@app.get("/")
async def say_hello():
    return {"message": "Hi there!"}


@app.post("/inference")
async def create_item(features: Features):
    features = features.dict(by_alias=True)
    features["salary"] = ">50K"
    data = pd.DataFrame(features, index=[0])

    X, _, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=False,
        encoder=app.encoder, lb=app.lb
    )

    prediction = inference(app.model, X)
    prediction = app.lb.inverse_transform(prediction)[0].strip()

    return {"predicted_salary": prediction}
