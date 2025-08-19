from pathlib import Path
import os
import getpass
import platform
import mlflow


def init_mlflow(exp_name: str):
    # 1) Permite anular por variable de entorno si alguien quiere usar su propio remoto/servidor
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        # ajusta si tu estructura es distinta
        repo_root = Path(__file__).resolve().parent
        runs_dir = (repo_root / "mlruns").resolve()
        uri = runs_dir.as_uri()  # -> file:///C:/.../mlruns (válido en Windows)

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp_name)

    # Etiquetas útiles para identificar a cada persona/equipo/máquina
    mlflow.set_tag("user", os.getenv("USERNAME", getpass.getuser()))
    mlflow.set_tag("machine", platform.node())
    # si queréis exportarlo en el CI
    mlflow.set_tag("branch", os.getenv("GIT_BRANCH", ""))
