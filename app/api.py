from fastapi import APIRouter, HTTPException
import numpy as np
import joblib
import os
from .schemas import CandidateInput, PredictionResponse

router = APIRouter()

# Load model
model_path = os.path.join("models", "saved", "svm_model_linear.joblib")
scaler_path = os.path.join("models", "saved", "scaler_linear.joblib")

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Model files not found. Please train the model first.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@router.get("/")
async def root():
    return {"message": "Welcome to Candidate Selection API"}

@router.post("/predict", response_model=PredictionResponse)
async def predict_candidate(candidate: CandidateInput):
    try:
        # Prepare input data
        X = np.array([[candidate.experience_years, candidate.technical_score]])
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = float(model.decision_function(X_scaled)[0])  # SVM decision function score
        
        return {
            "prediction": int(prediction),
            "probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model_info")
async def get_model_info():
    try:
        return {
            "model_type": type(model).__name__,
            "kernel": model.kernel,
            "C": model.C,
            "gamma": model.gamma if hasattr(model, 'gamma') else 'scale'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 