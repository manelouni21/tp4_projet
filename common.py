"""Constantes et fonctions partagées par tous les scripts du TP3."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "churn.csv"
TARGET = "Exited"
RANDOM_STATE = 42

def load_split():
    """Charge churn.csv et renvoie X_train, X_test, y_train, y_test.
    Le découpage est identique dans tous les scripts (même random_state)."""
    df = pd.read_csv(DATA_FILE).drop(columns=["RowNumber", "CustomerId", "Surname"])
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

def build_pipeline(classifier, X):
    """Construit le pipeline complet (prétraitement + classifieur), comme au TP1."""
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])