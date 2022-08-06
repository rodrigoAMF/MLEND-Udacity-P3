from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


# Write tests using the same syntax as with the requests module.
def test_greeting():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Hi there!"}


def test_prediction_below_or_equal_50k():
    response = client.post("/inference", json={
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
    })

    assert response.status_code == 200
    assert response.json() == {"predicted_salary": "<=50K"}


def test_prediction_above_50k():
    response = client.post("/inference", json={
        "age": 42,
        "workclass": "State-gov",
        "fnlgt": 205499,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 77,
        "native-country": "United-States"
    })

    assert response.status_code == 200
    assert response.json() == {"predicted_salary": ">50K"}
