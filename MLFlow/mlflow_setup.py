# mlflow_setup.py (fragmento)
from pathlib import Path
import os
import mlflow


def init_mlflow(exp_name: str):
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        # Sube un nivel (de MLFlow/ a la raíz del repo)
        repo_root = Path(__file__).resolve().parents[1]
        runs_dir = (repo_root / "mlruns").resolve()
        uri = runs_dir.as_uri()

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp_name)
