# FastAPI ML Model Deployment

This project deploys a machine learning model using FastAPI. The application provides a REST API to make predictions based on input data.

Link to the notebook on Kaggle: https://www.kaggle.com/code/saeedghamshadzai/medical-insurance-cost-prediction-classical-ml

# Application setup commands

Run these commands to set up your virtual environment:

    Create virtual environment and activate it:
        python -m venv app/env
        source app/env/bin/activate     # On Windows use `app\env\Scripts\activate` or `app\env\Scripts\Activate.ps1`

    Install dependencies:
        pip install --upgrade pip
        pip install -r requirements.txt 

    Configure Environment Variables:
        cp app/.env.template app/.env   # On Windows use Copy-Item app\.env.template app\.env
    
    Run the application:
        fastapi dev app/api/main.py
    Using docker:
        docker-compose -f docker-compose.yaml up --build

## Project Structure

```plaintext
Medical-Insurance-Cost-Prediction-Classical-ML/
│
├── app/
│   ├── api
│   │   ├── __init__.py
│   │   └── main.py
|   |
│   ├── env
│   ├── model/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── notebook.ipynb
│   │   │   ├── model.py
│   │   └── trained_model-0.1.0.pkl
|   |
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   └── dataset/
│   │       └── medical_insurance.csv
|   |
│   |── tests/
│   |    ├── __init__.py
│   |    └── test_main.py
|   |
|   ├── setup.py
|   ├── pytest.ini
|   └── .env
|
├── requirements.txt
├── dockerfile
├── docker-compose.yaml
├── .dockerignore
├── .gitignore
└── README.md