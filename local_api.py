import requests

# Send a GET request to the base URL
response_get = requests.get("http://127.0.0.1:8000")

# Print the status code and welcome message from the GET request
print("GET request:")
print("Status Code:", response_get.status_code)
print("Welcome Message:", response_get.text)

data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST request to the /predict endpoint using the data above
response_post = requests.post("http://127.0.0.1:8000/predict", json=data)

# Print the status code and result from the POST request
print("\nPOST request:")
print("Status Code:", response_post.status_code)
print("Result:", response_post.text)
