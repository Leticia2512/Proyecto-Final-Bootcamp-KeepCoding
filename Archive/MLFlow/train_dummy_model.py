# train_dummy_model.py
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import sklearn

import mlflow
from mlflow_setup import init_mlflow


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
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
    # Inicializa MLflow (tu función fija tracking_uri y experimento)
    init_mlflow("DataAvengers")

    # Cierra cualquier run activo por si un script previo quedó a medias
    if mlflow.active_run() is not None:
        mlflow.end_run()

    # ---------------- Datos ----------------
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    # ---------------- Modelo ----------------
    C = 1.0
    max_iter = 200
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter,
         solver="lbfgs", multi_class="auto"))
    ])

    run_name = f"{os.getenv('USERNAME','user')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # Parámetros y metadata útil
        mlflow.log_params({
            "model": "LogisticRegression",
            "C": C,
            "max_iter": max_iter,
            "with_scaler": True,
            "sklearn_version": sklearn.__version__,
        })

        # Entrenamiento
        pipe.fit(X_train, y_train)

        # Evaluación
        y_pred = pipe.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)

        # Artefactos
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

        # Modelo
        mlflow.sklearn.log_model(pipe, artifact_path="model")

    print(
        f"Listo ✅  Acc: {acc:.4f}. Revisa el run en ./mlruns o lanza 'mlflow ui'.")
