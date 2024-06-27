from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security.api_key import APIKeyHeader, APIKey
from pydantic import BaseModel, ValidationError, conint, confloat
from enum import Enum
import logging
from dotenv import load_dotenv
import pandas as pd
import os
from data.data import preprocessor

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app/api/app.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()
# Get the model version from the environment
model_version = os.getenv('MODEL_VERSION', 'v1')

# Import the correct model version dynamically
try:
    model = __import__(f'model.{model_version}.model', fromlist=[''])

except ImportError as e:
    logging.error(f'Failed to import model version {model_version}: {e}', exc_info=True)
    raise RuntimeError(f'Model version {model_version} not found')

app = FastAPI()

API_KEY = os.getenv('API_KEY')
api_key_header = APIKeyHeader(name='Authorization')

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

class Sex_type(str, Enum):
    male = "male"
    female = "female"

class Smoker_type(str, Enum):
    yes = "yes"
    no = "no"

class Region_type(str, Enum):
    northeast = "northeast"
    northwest = "northwest"
    southeast = "southeast"
    southwest = "southwest"

@app.get("/")
async def home(request: Request, api_key: APIKey = Depends(get_api_key)):
    """
    Endpoint to perform a health check and get the model version.
    """
    logging.info("Home endpoint called from %s", request.client.host)
    return {"health_check": "OK", "model_version": model.__version__}

@app.post("/prediction/")
async def get_prediction(
    Age: conint(ge=preprocessor.age_min_max[0], le=preprocessor.age_min_max[1]),
    Sex: Sex_type,
    Bmi: confloat(ge=preprocessor.bmi_min_max[0], le=preprocessor.bmi_min_max[1]),
    Children: conint(ge=preprocessor.chl_min_max[0], le=preprocessor.chl_min_max[1]),
    Smoker: Smoker_type,
    Region: Region_type,
    request: Request,
    api_key: APIKey = Depends(get_api_key)
    ):
    """
    Endpoint to get a prediction from the model based on input data.
    """
    try:
        input_data = {
        'age': [Age],
        'sex': [Sex.value],
        'bmi': [Bmi],
        'children': [Children],
        'smoker': [Smoker.value],
        'region': [Region.value]
        }
        
        input_data['charges'] = [0]
        input_data = pd.DataFrame.from_dict(input_data)

        regressor = model.load_model()
        prediction = model.predictor(input_data, regressor)

        logging.info("Prediction made successfully")

        # JSON serializable data
        input_data_serializable = input_data.applymap(lambda x: x.item() if hasattr(x, 'item') else x).to_dict(orient='list')

        return {
            "predicted_medical_insurance_cost": round(prediction[0, 0], 2),
            "details": {
                "input_data": input_data_serializable,
                "model_version": model.__version__
            }
        }
    
    except ValidationError as e:
        logging.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid input data")
    except RuntimeError as e:
        logging.error(f"Runtime error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

if __name__ == "__main__":
    import uvicorn
    import subprocess

    # Run tests before starting the server
    result = subprocess.run(["pytest", "app/tests"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Tests failed. Exiting.")
        exit(1)
    else:
        print("Tests ended successfully.")

    # uvicorn.run(app, host="0.0.0.0", port=8080)
    uvicorn.run(app, host="127.0.0.1", port=8000)