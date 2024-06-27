# FastAPI ML Model Deployment

This project deploys a machine learning model using FastAPI. The application provides a REST API to make predictions based on input data.

# Application setup commands

Run these commands to set up your virtual environment:
    python -m venv env
    source env/bin/activate     # On Windows use `env\Scripts\activate` or `env\Scripts\Activate.ps1`
    pip install -r requirements.txt     # Install dependencies 
    cp .env.example .env    # Configure Environment Variables
    
    Run the application:
        fastapi dev app/api/main.py
    Using docker:
        docker-compose -f docker-compose.yml up --build

## Project Structure

```plaintext
my-fastapi-project/
│
├── app/
│   ├── __pycache__
│   ├── api
│   │   ├── __init__.py
│   │   ├── data.py
│   │   └── csv
│   |       └── medical_insurance.csv
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
│   ├── notebook
│   ├── packages.egg-info
│   |── tests/
│   |    ├── __init__.py
│   |    └── test_main.py
|   |
|   ├── setup.py
|   ├── requirements.txt
|   ├── pytest.ini
|   └── .env
|
├── Dockerfile
├── docker-compose.yaml
├── .dockerignore
├── .gitignore
├── .pytest_cache
└── README.md