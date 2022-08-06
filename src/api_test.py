import requests
import json

host_heroku = "https://mlend-udacity-p3.herokuapp.com/"
inference_endpoint = "inference/"

features = {
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

url = host_heroku + inference_endpoint
response = requests.post(url, data=json.dumps(features))

print(f"Request status code: {response.status_code}")
print("Request response:")
print(response.json())
