from __future__ import annotations
import pathlib
import json
from typing import List, Dict

import faiss  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

# ------------------ utils ------------------


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    tokens = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def embed_chunks(chunks: List[str], model_name: str):
    model = SentenceTransformer(model_name)
    emb = model.encode(chunks, batch_size=32,
                       show_progress_bar=True, normalize_embeddings=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, dim


def save_chunks_jsonl(chunks: List[str], path: pathlib.Path):
    with path.open("w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks):
            f.write(json.dumps({"id": i, "text": ch},
                    ensure_ascii=False) + "\n")

# ------------------ main ------------------


def main():
    base_dir = pathlib.Path(__file__).resolve().parent
    fixed_dir = base_dir / "fixed"
    out_dir = base_dir / "vectorstores"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Agrupación de documentos por dolencia (usa los nombres de tus .txt en Fixed_david)
    groups: Dict[str, List[str]] = {
        "dmae": [
            "Guia_SERV_01_segundaRevision.txt",
            "consenso_DMAE.txt",
            "GPC_DMRE_Version_extensa.txt",
            "Guia_consensuada_DMAE.txt",
        ],
        "catarata": [
            "GPC-Catarata-en-adulto-y-adulto-mayor_Version-extensa-y-Anexos.txt",
            "GPC_523_Catarata_Adulto_actualiz_2013.txt",
            "192GER.txt",
        ],
        "retinopatia": [
            "Guia_de_EMD_y_RD_SAO_2022.txt",
            "guiaclinicaretinopatiadiabetica2016.txt",
            "16-pai-retinopatia-diabetica.txt",
        ],
        "miopia": [
            "Guia_SERV_18.txt",
            "IMI-2021_Resumen-Clinico-Del-IMI_Miopia-Patologica.txt",
        ],
    }

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    for disease, files in groups.items():
        print(f"\n Procesando grupo: {disease.upper()}")
        all_chunks = []
        for fname in files:
            txt_path = fixed_dir / fname
            if not txt_path.exists():
                print(f"No encontrado: {fname}")
                continue
            text = txt_path.read_text(encoding="utf-8")
            chunks = split_into_chunks(text, 1000, 150)
            print(f"   {fname}: {len(chunks)} chunks")
            all_chunks.extend(chunks)

        if not all_chunks:
            print(f"Ningún chunk generado para {disease}")
            continue

        # Embeddings e índice
        index, dim = embed_chunks(all_chunks, model_name)

        # Guardado
        idx_path = out_dir / f"{disease}_index.faiss"
        jsonl_path = out_dir / f"{disease}_chunks.jsonl"

        faiss.write_index(index, str(idx_path))
        save_chunks_jsonl(all_chunks, jsonl_path)

        print(f"Guardado índice en {idx_path}")
        print(f"Guardado chunks en {jsonl_path}")

    print("\nProceso completado. Vectorstores en", out_dir)


if __name__ == "__main__":
    main()
