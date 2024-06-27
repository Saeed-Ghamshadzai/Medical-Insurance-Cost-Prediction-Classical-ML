# FastAPI ML Model Deployment

This project deploys a machine learning model using FastAPI. The application provides a REST API to make predictions based on input data.

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