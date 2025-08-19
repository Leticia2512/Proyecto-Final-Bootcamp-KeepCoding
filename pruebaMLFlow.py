from mlflow_setup import init_mlflow
import mlflow

init_mlflow("BasicExperiment2")

with mlflow.start_run(run_name="run1"):
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("val_loss", 0.42)
    # Guardar un modelo pequeño
    with open("dummy.txt", "w") as f:
        f.write("Este es un artefacto de ejemplo2")
    mlflow.log_artifact("dummy.txt")
