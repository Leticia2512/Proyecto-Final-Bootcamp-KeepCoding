#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import pathlib
from typing import List

import faiss  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from tqdm import tqdm  # type: ignore

# --------------------------------- utils ------------------------------------


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Divide `text` en bloques de ~`chunk_size` tokens (palabras) con `overlap`."""
    tokens = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = end - overlap  # retrocede para solapamiento
    return chunks


def embed_chunks(chunks: List[str], model_name: str):
    """Genera embeddings con Sentence‑Transformers y devuelve (faiss_index, dim)."""
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

# --------------------------------- main -------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Genera chunks y embeddings de un .txt limpio")
    parser.add_argument("txt_file", type=pathlib.Path,
                        help="Ruta al archivo .txt limpio")
    parser.add_argument("--chunk_size", type=int,
                        default=1000, help="Tokens aprox. por chunk")
    parser.add_argument("--overlap", type=int, default=150,
                        help="Solapamiento entre chunks")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Modelo SBERT para embeddings")
    args = parser.parse_args()

    txt_path: pathlib.Path = args.txt_file
    if not txt_path.exists():
        raise SystemExit(f"❌ Archivo '{txt_path}' no encontrado")

    text = txt_path.read_text(encoding="utf-8")
    print("→ Dividiendo en chunks …")
    chunks = split_into_chunks(text, args.chunk_size, args.overlap)
    print(f"   {len(chunks)} chunks generados")

    print("→ Generando embeddings con", args.model)
    index, dim = embed_chunks(chunks, args.model)

    # Guardado
    out_dir = txt_path.parent
    stem = txt_path.stem
    idx_path = out_dir / f"{stem}_index.faiss"
    jsonl_path = out_dir / f"{stem}_chunks.jsonl"

    faiss.write_index(index, str(idx_path))
    save_chunks_jsonl(chunks, jsonl_path)

    print("✅ Índice FAISS guardado en", idx_path)
    print("✅ Chunks JSONL guardado en", jsonl_path)


if __name__ == "__main__":
    main()
