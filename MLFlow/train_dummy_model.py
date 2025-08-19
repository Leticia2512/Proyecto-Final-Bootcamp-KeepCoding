# train_dummy_model.py
from datetime import datetime
from pathlib import Path
import mlflow
from mlflow_setup import init_mlflow

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, class_names, out_path: Path):
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Inicia MLflow (usa tu función del repo)
    init_mlflow("DataAvengers")

    # Datos
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    # Pipeline simple
    C = 1.0
    max_iter = 200
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, n_jobs=None))
    ])

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=run_name):
        # Log de parámetros
        mlflow.log_params({
            "model": "LogisticRegression",
            "C": C,
            "max_iter": max_iter,
            "with_scaler": True
        })

        # Entrena
        pipe.fit(X_train, y_train)

        # Evalúa
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Métricas
        mlflow.log_metric("accuracy", float(acc))

        # Artefactos: matriz de confusión y classification report
        out_dir = Path("artifacts_dummy")
        out_dir.mkdir(exist_ok=True)
        cm_path = out_dir / "confusion_matrix.png"
        plot_confusion_matrix(cm, iris.target_names, cm_path)
        mlflow.log_artifact(str(cm_path))

        report_txt = out_dir / "classification_report.txt"
        with open(report_txt, "w", encoding="utf-8") as f:
            f.write(classification_report(
                y_test, y_pred, target_names=iris.target_names))
        mlflow.log_artifact(str(report_txt))

        # Log del modelo
        mlflow.sklearn.log_model(pipe, artifact_path="model")

    print(
        f"Listo ✅  Acc: {acc:.4f}. Revisa el run en ./mlruns o con 'mlflow ui'.")
