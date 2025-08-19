import mlflow
from pathlib import Path


def init_mlflow(exp_name="odir-multimodal"):
    #base_dir = Path(__file__).resolve().parent
    #mlruns_dir = base_dir / "mlruns"
    mlruns_dir = "C:\\Users\\sofia\\Documents\\ProyectoFinalKC\\mlruns"
    uri = "file:" + str(mlruns_dir)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp_name)
    return mlruns_dir
