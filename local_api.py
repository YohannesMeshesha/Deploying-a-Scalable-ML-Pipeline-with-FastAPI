import requests

# Test the GET endpoint
response_get = requests.get("http://127.0.0.1:8000")
print("GET request:")
print("Status Code:", response_get.status_code)
print("Welcome Message:", response_get.text)

# Define sample input data for the POST endpoint
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
    "native-country": "United-States"
}

# Test the POST endpoint (note the trailing slash)
response_post = requests.post("http://127.0.0.1:8000/data/", json=data)
print("\nPOST request:")
print("Status Code:", response_post.status_code)
print("Result:", response_post.text)
