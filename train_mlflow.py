"""Entraîne plusieurs modèles et trace chaque essai dans MLflow."""
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from common import build_pipeline, load_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
# TODO 1 : donner un nom à l'expérience (elle regroupe tous les essais)
mlflow.set_experiment("churn")

X_train, X_test, y_train, y_test = load_split()

experiments = [
    ("logreg_C1", LogisticRegression(C=1.0, max_iter=1000)),
    ("logreg_C0.1", LogisticRegression(C=0.1, max_iter=1000)),
    ("rf_100_depth5", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ("rf_200_depth10", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    # TODO 2 : ajoutez un essai de votre choix (ex. un autre max_depth, ou GradientBoostingClassifier)
    ("gradient_boosting", GradientBoostingClassifier(random_state=42)),
]

for run_name, classifier in experiments:
    # TODO 3 : ouvrir un essai (run) MLflow qui porte le nom run_name
    with mlflow.start_run(run_name=run_name):
        pipeline = build_pipeline(classifier, X_train)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # TODO 4 : enregistrer tous les hyperparamètres du classifieur
        mlflow.log_params(classifier.get_params())
        mlflow.log_param("model_type", type(classifier).__name__)

        # TODO 5 : enregistrer les métriques (rappel et F1 de la classe 1)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1", f1_score(y_test, y_pred))

        # Le pipeline complet est sauvegardé comme artefact de l'essai
        # (format cloudpickle : nécessaire pour les modèles à base d'arbres avec MLflow)
        mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")

        print(f"{run_name:<16} accuracy={accuracy_score(y_test, y_pred):.3f} "
              f"recall={recall_score(y_test, y_pred):.3f} f1={f1_score(y_test, y_pred):.3f}")