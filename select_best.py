"""Choisit le meilleur essai MLflow et exporte son pipeline pour l'API."""
import json
import joblib
import mlflow
import mlflow.sklearn

from common import load_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# TODO 1 : récupérer les essais de l'expérience, triés par F1 décroissant
runs = mlflow.search_runs(
    experiment_names=["churn"],
    order_by=["metrics.f1 DESC"],
)
print(runs[["tags.mlflow.runName", "metrics.accuracy", "metrics.recall", "metrics.f1"]]
      .round(3).to_string(index=False))

# TODO 2 : le meilleur essai est la ligne d'indice ... (les essais sont déjà triés)
best = runs.iloc[0]
run_id = best["run_id"]
print(f"\nMeilleur essai :", best["tags.mlflow.runName"], "| run_id :", run_id)

# TODO 3 : recharger le pipeline sauvegardé dans l'essai (artefact nommé "model")
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

X_train, X_test, y_train, y_test = load_split()
joblib.dump(model, "model.joblib")
joblib.dump(list(X_train.columns), "expected_columns.joblib")

info = {
    "run_id": run_id,
    "run_name": best["tags.mlflow.runName"],
    "accuracy": round(float(best["metrics.accuracy"]), 4),
    "recall": round(float(best["metrics.recall"]), 4),
    "f1": round(float(best["metrics.f1"]), 4),
}

with open("best_run.json", "w") as f:
    json.dump(info, f, indent=2)

print("Fichiers créés : model.joblib, expected_columns.joblib, best_run.json")