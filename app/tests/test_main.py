from fastapi.testclient import TestClient
from api import main
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
# Get the model version from the environment
model_version = os.getenv('MODEL_VERSION', 'v1')
# Import the correct model version dynamically
model = __import__(f'model.{model_version}.model', fromlist=[''])

client = TestClient(main.app)

API_KEY = os.getenv('API_KEY')

def test_health_check():
    headers = {"Authorization": API_KEY}
    response = client.get("/", headers=headers)
    print(response.json())
    print('\n'*9)
    assert response.status_code == 200
    assert response.json() == {"health_check": "OK", "model_version": model.__version__}

def test_prediction_endpoint():
    headers = {"Authorization": API_KEY}
    params = {
        "Age": [20],
        "Sex": ["male"],
        "Bmi": [30],
        "Children": [0],
        "Smoker": ["no"],
        "Region": ["northwest"]
    }
    response = client.post("/prediction/", params=params, headers=headers)
    assert response.status_code == 200
    assert "predicted_medical_insurance_cost" in response.json()
    assert "details" in response.json()
