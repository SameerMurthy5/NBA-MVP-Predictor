from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import pandas as pd
import boto3
import pickle
import os
import zipfile
import importlib.util

# load ml_module
def load_ml_module():
    s3 = boto3.client('s3')
    zip_file = 'ml_module.zip'
    module_dir = 'ml_module'

    # Download the zipped module from S3
    s3.download_file(S3_BUCKET, zip_file, zip_file)

    # Extract the module
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('.')

    # Clean up the zip file after extraction
    os.remove(zip_file)

    # Import the module dynamically
    spec = importlib.util.spec_from_file_location('ml_export', f"{module_dir}/ml_export.py")
    ml_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ml_module)

    return ml_module

# Load the Entire Pickled DataFrame from S3
def load_all_data():
    filename = "NBA_DATA.pkl"
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=S3_BUCKET, Key=filename)
    stats = pickle.loads(obj['Body'].read())
    return stats



def load_csv_from_s3(filename):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=S3_BUCKET, Key=filename)
    df = pd.read_csv(obj['Body'])
    return df

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# S3 Configuration
S3_BUCKET = "mvp-predictions-models"
MODEL_KEY = "mvp_model.pkl"




# Load ML module from S3
ml_module = load_ml_module()

#Load the Entire DataFrame Once
all_stats = load_all_data()

# Define Input Schema
class YearInput(BaseModel):
    year: int

@app.get("/")
def read_root():
    return {"message": "MVP Prediction API"}

@app.post("/predict")
def predict(year: YearInput):
    try:
        # Define the Predictors
        predictors = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', 
                      '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 
                      'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Year', 'W', 'L', 'W/L%', 
                      'GB', 'PS/G', 'PA/G', 'SRS']

        # Get Predictions
        model, results = ml_module.train_mvp_predictor(all_stats, predictors, year.year)
        first_10_rows = results.head(10)

        # Prepare Detailed Output
        prediction_details = [
            {
                "player": row["Player"],
                "prediction_score": row["predictions"],
                "actual_share": row["Share"],
                "predicted_rank": row["Predicted_Rk"]
            }
            for i, row in first_10_rows.iterrows()
        ]

        return {
            "year": year.year,
            "top_10_mvp_predictions": prediction_details
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))