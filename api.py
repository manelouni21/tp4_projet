"""API de prédiction du churn (TP3)."""
import json
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="API de prédiction du churn (TP3)")

# Chargement une seule fois, au démarrage
model = joblib.load("model.joblib")
expected_columns = joblib.load("expected_columns.joblib")
with open("best_run.json") as f:
    best_run = json.load(f)

class CustomerData(BaseModel):
    CreditScore: int = Field(ge=300, le=850, examples=[619])
    Geography: Literal["France", "Spain", "Germany"] = Field(examples=["France"])
    Gender: Literal["Female", "Male"] = Field(examples=["Female"])
    Age: int = Field(ge=18, le=100, examples=[42])
    Tenure: int = Field(ge=0, le=10, examples=[2])
    Balance: float = Field(ge=0, examples=[0.0])
    NumOfProducts: int = Field(ge=1, le=4, examples=[1])
    HasCrCard: int = Field(ge=0, le=1, examples=[1])
    IsActiveMember: int = Field(ge=0, le=1, examples=[1])
    EstimatedSalary: float = Field(ge=0, examples=[101348.88])

@app.get("/")
def home():
    return {"message": "API opérationnelle. Voir /docs"}

@app.get("/health")
def health():
    return {"status": "ok", "modele": best_run["run_name"], "f1": best_run["f1"]}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        df = pd.DataFrame([data.model_dump()])[expected_columns]
        pred = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1])
        return {"prediction": pred, "probabilite_churn": round(proba, 4)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))