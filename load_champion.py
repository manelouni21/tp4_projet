"""Charge le modèle « champion » depuis le registre (sans connaître l'essai) et le teste."""
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# TODO 4 : charger la version qui porte l'alias « champion »
champion = mlflow.sklearn.load_model("models:/churn-model@champion")

client = {
    "CreditScore": 619, "Geography": "France", "Gender": "Female", "Age": 42,
    "Tenure": 2, "Balance": 0.0, "NumOfProducts": 1, "HasCrCard": 1,
    "IsActiveMember": 1, "EstimatedSalary": 101348.88,
}
columns = joblib.load("expected_columns.joblib")
df = pd.DataFrame([client])[columns]

proba = champion.predict_proba(df)[0][1]
print(f"Probabilité de churn (champion du registre) : {proba:.4f}")

# Le champion du registre doit donner la même réponse que model.joblib (utilisé par l'API)
reference = joblib.load("model.joblib").predict_proba(df)[0][1]
print("Identique à model.joblib :", abs(proba - reference) < 1e-9)