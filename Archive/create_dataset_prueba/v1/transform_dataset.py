import os
import ast
import numpy as np
import pandas as pd
from pathlib import Path

PARQUET_IN = "EDA/dataset_meta_final.parquet"
OUT_DIR = "DataNau"


def main():
    df = pd.read_parquet(PARQUET_IN)

    # Construir dataset por ojo
    eyes_rows = []
    for _, row in df.iterrows():
        eyes_rows.append({
            "eye": "left",
            "image_name": row["Left-Fundus"],
            "idx_list": row["left_idx"],
            "Patient Age": row["Patient Age"],
            "Patient_Sex_Binario": row["Patient_Sex_Binario"],
        })
        eyes_rows.append({
            "eye": "right",
            "image_name": row["Right-Fundus"],
            "idx_list": row["right_idx"],
            "Patient Age": row["Patient Age"],
            "Patient_Sex_Binario": row["Patient_Sex_Binario"],
        })

    eyes_df = pd.DataFrame(eyes_rows)

    # Guardar parquet
    out_file = Path(OUT_DIR) / "dataset_eyes_long.parquet"
    eyes_df.to_parquet(out_file, index=False)

    # Info rápida
    all_idxs = [i for lst in eyes_df["idx_list"] for i in lst] or [0]
    num_classes = max(all_idxs) + 1
    print(f"Guardado dataset ojo-level: {out_file}")
    print(f"Registros ojo-level: {len(eyes_df)}")
    print(f"num_classes (por idx): {num_classes}")


if __name__ == "__main__":
    main()
