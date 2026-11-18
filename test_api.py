
"""Tests automatiques de l'API de prédiction du churn."""

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

# Client valide réutilisé dans les tests
VALID = {
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88,
}


def test_home():
    """Vérifie que la page d'accueil fonctionne."""
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    """Vérifie l'état de santé de l'API."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valide():
    """Vérifie une prédiction avec des données valides."""
    response = client.post("/predict", json=VALID)

    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in [0, 1]
    assert 0 <= body["probabilite_churn"] <= 1


def test_predict_age_negatif():
    """Un âge négatif doit être refusé."""
    data = {**VALID, "Age": -5}
    response = client.post("/predict", json=data)

    assert response.status_code == 422


def test_predict_champ_manquant():
    """Un champ obligatoire manquant doit être refusé."""
    data = VALID.copy()
    data.pop("Geography")

    response = client.post("/predict", json=data)

    assert response.status_code == 422


def test_predict_pays_inconnu():
    """Un pays non autorisé doit être refusé."""
    data = {**VALID, "Geography": "Tunisia"}
    response = client.post("/predict", json=data)

    assert response.status_code == 422


def test_predict_deterministe():
    """Deux requêtes identiques doivent donner le même résultat."""
    r1 = client.post("/predict", json=VALID)
    r2 = client.post("/predict", json=VALID)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json() == r2.json()
