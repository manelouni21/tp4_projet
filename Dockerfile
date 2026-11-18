
# Image Python légère
FROM python:3.14.5-slim

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code et les modèles
COPY api.py .
COPY model.joblib expected_columns.joblib best_run.json ./

# Version de l'application
ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}

# Port de l'API
EXPOSE 8000

# Lancer FastAPI avec Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
