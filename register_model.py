"""Enregistre le meilleur modèle dans le registre MLflow et le promeut « champion »."""
import json
import mlflow
from mlflow import MlflowClient

MODEL_NAME = "churn-model"
MIN_F1 = 0.40  # même seuil que dans test_model.py

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Résumé de l'essai retenu par select_best.py (partie 4)
with open("best_run.json") as f:
    best = json.load(f)

# TODO 1 : porte de qualité : refuser d'enregistrer un modèle dont le F1 est trop bas
if best["f1"] < MIN_F1:
    raise SystemExit(f"Refusé : F1 = {best['f1']} < {MIN_F1}")

# TODO 2 : enregistrer le modèle de cet essai dans le registre (il reçoit un numéro de version)
version = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
client.set_model_version_tag(MODEL_NAME, version.version, "f1", str(best["f1"]))
client.set_model_version_tag(MODEL_NAME, version.version, "run_name", best["run_name"])

# TODO 3 : donner l'alias « champion » à cette version (= le modèle utilisé en production)
client.set_registered_model_alias(MODEL_NAME, "champion", version.version)

print(f"Modèle '{MODEL_NAME}' version {version.version} enregistré et promu champion "
      f"(essai {best['run_name']}, F1 = {best['f1']})")