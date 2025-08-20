from datetime import datetime
import mlflow
from mlflow_setup import init_mlflow

init_mlflow("DataAvengersSofia")

with mlflow.start_run(run_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # métricas
    mlflow.log_metric("accuracy", 0.913)
    mlflow.log_metric("loss", 0.327)

    # parámetros
    mlflow.log_params({"model": "xgboost", "max_depth": 6, "lr": 0.05})

    # artefactos (ficheros de salida)
    with open("notas.txt", "w", encoding="utf-8") as f:
        f.write("Resultado del experimento...")
    mlflow.log_artifact("notas.txt")
