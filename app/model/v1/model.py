import joblib
import os
from pathlib import Path
from data import data
from dotenv import load_dotenv
import sklearn
import logging

__version__ = "0.1.0"

# Load environment variables
load_dotenv()
# # Get the model version from the environment
path_to_model = os.getenv('PATH_TO_MODEL')

def load_model():
    """
    Load the trained machine learning model.

    Returns:
        object: Loaded machine learning model.

    Raises:
        RuntimeError: If loading the model fails.
    """
    try:
        with open(f"{path_to_model}/trained_model-{__version__}.pkl", "rb") as f:
            model = joblib.load(f)
            logging.info("Model loaded successfully")
            return model

    except Exception as e:
        logging.error(f'Faild to load model: {e}')
        raise RuntimeError('Model loading failed')

def predictor(input_data, regressor):
    """
    Make predictions using the provided input data and regressor.

    Args:
        input_data (DataFrame): Input data for prediction.
        regressor (object): Machine learning regressor object.

    Returns:
        ndarray: Predicted values.

    Raises:
        RuntimeError: If prediction fails.
    """
    try:
        processed_data = data.preprocessor.transform(input_data, drop_target=True, scale=True)
        prediction = regressor.predict(processed_data.values)
        output = data.preprocessor.inverse_transform_target(prediction.reshape(-1, 1))

        return output

    except Exception as e:
        logging.error(f'Prediction failed: {e}')
        raise RuntimeError(f'Prediction failed: {e}')