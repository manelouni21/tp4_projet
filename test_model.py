
"""Tests de qualité du modèle avant sa mise en production."""

import joblib
import mlflow
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, f1_score

from common import load_split

# Seuils minimaux de qualité
MIN_ACCURACY = 0.80
MIN_F1 = 0.40

# Chargement du modèle et des données de test
model = joblib.load("model.joblib")
X_train, X_test, y_train, y_test = load_split()
y_pred = model.predict(X_test)


def test_accuracy_minimale():
    """Vérifie que l'accuracy atteint le seuil minimal."""
    assert accuracy_score(y_test, y_pred) >= MIN_ACCURACY


def test_f1_minimal():
    """Vérifie que le score F1 atteint le seuil minimal."""
    assert f1_score(y_test, y_pred) >= MIN_F1


def test_sortie_binaire():
    """Le modèle doit uniquement renvoyer 0 ou 1."""
    assert set(y_pred) <= {0, 1}


def test_champion_dans_le_registre():
    """Vérifie que le champion respecte le seuil F1."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    client = MlflowClient()
    version = client.get_model_version_by_alias(
        "churn-model", "champion"
    )

    assert float(version.tags["f1"]) >= MIN_F1
