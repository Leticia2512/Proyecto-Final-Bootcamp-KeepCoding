import os
import json
import pathlib
from typing import List
from datetime import datetime

import faiss  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from secrets_ApiKey import get_openai_api_key

import mlflow

# ================== CONFIG ==================
BASE_DIR = pathlib.Path(__file__).resolve().parent

INDEX_PATHS = {
    "dmae":        BASE_DIR / "vectorstores" / "david" / "dmae_index.faiss",
    "catarata":    BASE_DIR / "vectorstores" / "david" / "catarata_index.faiss",
    "retinopatia": BASE_DIR / "vectorstores" / "david" / "retinopatia_index.faiss",
    "miopia":      BASE_DIR / "vectorstores" / "david" / "miopia_index.faiss",
}

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPERIMENT_NAME = "RAG_Clinico"

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.0

# Prompt orientado a recomendaciones clínicas
template = """
SISTEMA: Eres un asistente clínico especializado en OFTALMOLOGÍA (nivel experto). Responde siempre en español. 
TU ROL: ayudar al profesional sanitario con recomendaciones basadas únicamente en el CONTEXTO proporcionado; 
no hagas conjeturas fuera de ese contexto. No dispares consejos médicos definitivos sin indicar nivel de incertidumbre. 
Siempre sugiere derivación cuando haya signos de alarma.

INSTRUCCIONES:
1) Usa exclusivamente la información incluida en el campo "context".
2) Si falta información clave, enumérala en "Información faltante" y explica por qué es crítica.
3) Organiza la respuesta en las siguientes secciones:
   - Diagnóstico diferencial (con probabilidad, justificación, nivel de evidencia, confianza).
   - Pruebas diagnósticas recomendadas (con prioridad, hallazgos esperados, contraindicaciones, referencias).
   - Opciones de tratamiento (con indicación, dosis si procede, contraindicaciones, nivel de evidencia).
   - Señales de alarma / Red Flags (que requieren derivación urgente).
   - Plan de seguimiento (cuándo, qué monitorizar, próximos pasos).
   - Comunicación con el paciente (explicación breve y clara en 2-4 frases).
   - Información faltante.
   - Nivel de confianza global.
   - Referencias.
4) Indica prioridad en URGENTE / ALTA / MEDIA / BAJA.
5) Usa viñetas o listas claras; no es necesario devolver JSON ni estructuras de programación.
6) Recuerda: esto es apoyo clínico; siempre recomienda evaluación presencial cuando sea apropiado.

FORMATO DE SALIDA ESPERADO:

───────────────────────────────
🔹 Diagnóstico diferencial
- Nombre: …
- Probabilidad estimada: …
- Razonamiento: …
- Nivel de evidencia: …
- Confianza: …

🔹 Pruebas diagnósticas recomendadas
- Prueba: …
- Prioridad: …
- Hallazgos esperados: …
- Contraindicaciones: …
- Referencias: …

🔹 Opciones de tratamiento
- Opción: …
- Indicación: …
- Dosis/Régimen: …
- Contraindicaciones: …
- Evidencia: …
- Confianza: …

Red Flags
- …

Plan de seguimiento
- Cuándo: …
- Qué monitorizar: …
- Próximos pasos: …

Comunicación con el paciente
"…"

Información faltante
- …

Nivel de confianza global
- …

Referencias
- …

Contexto:
{context}

Pregunta:
{question}

Responde con:
- Recomendaciones sobre nuevas pruebas diagnósticas.
- Recomendaciones generales de tratamiento.
- Aspectos importantes a considerar para esta dolencia.
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# ¿Está disponible la API de "Prompts" de MLflow?
try:
    from mlflow import prompts as mlflow_prompts
    HAS_MLFLOW_PROMPTS = True
except Exception:
    HAS_MLFLOW_PROMPTS = False


# ================== CORE ==================
def init_mlflow():
    project_root = BASE_DIR.parent
    uri = "file:" + str((project_root / "mlruns").resolve())
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def load_faiss_index(index_path: pathlib.Path, model_name: str):
    """Carga índice FAISS y el modelo de embeddings usado al crearlo."""
    model = SentenceTransformer(model_name)
    index = faiss.read_index(str(index_path))
    return model, index


def retrieve_chunks(query: str, model: SentenceTransformer, index: faiss.Index, docs_jsonl: pathlib.Path, k: int = 6):
    """Busca los chunks más relevantes para una query."""
    chunks = []
    with docs_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb, k)
    results = [chunks[i] for i in I[0] if 0 <= i < len(chunks)]
    return results


def generate_query_variants(base_query: str, n_variants: int = 3) -> List[str]:
    """Genera variantes de la query base usando LLM."""
    llm = ChatOpenAI(model_name=OPENAI_MODEL, openai_api_key=get_openai_api_key(), temperature=0.0)
    variant_prompt = f"Genera {n_variants} variantes semánticas de la siguiente consulta en español, una por línea:\n\n{base_query}"
    resp = llm.predict(variant_prompt)
    variants = [v.strip("-• \n") for v in resp.split("\n") if v.strip()]
    return [base_query] + variants


def reciprocal_rank_fusion(results_per_query: List[List[dict]], top_k: int = 6):
    """Fusión muy simple (dedup + top_k)."""
    seen = set()
    fused = []
    for docs in results_per_query:
        for d in docs:
            txt = d.get("text", "")
            if txt not in seen:
                seen.add(txt)
                fused.append(d)
    return fused[:top_k]


def log_prompt_and_response_to_mlflow(
    base_query: str,
    context: str,
    answer: str,
    provider_meta: dict,
    queries: List[str],
    fused_docs: List[dict],
    index_path: pathlib.Path,
    k: int,
):
    """
    Registra parámetros/artefactos y, si está disponible, también en la pestaña 'Prompts' de MLflow.
    """
    # ---- Params & Tags ----
    mlflow.log_params({
        "emb_model": EMB_MODEL_NAME,
        "llm_model": OPENAI_MODEL,
        "temperature": OPENAI_TEMPERATURE,
        "top_k": k,
        "index_name": index_path.name,
    })
    mlflow.set_tags({
        "task": "clinical_rag",
        "component": "RAG_Fusion",
    })

    # ---- Artefactos útiles (trazabilidad) ----
    out_dir = BASE_DIR / "mlflow_artifacts" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "queries.json").write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "fused_docs.json").write_text(json.dumps(fused_docs, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "prompt.txt").write_text(prompt.format(context=context, question=base_query), encoding="utf-8")
    (out_dir / "answer.txt").write_text(answer, encoding="utf-8")
    (out_dir / "provider_meta.json").write_text(json.dumps(provider_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    mlflow.log_artifacts(str(out_dir), artifact_path="rag")

    # ---- Métricas sencillas ----
    mlflow.log_metrics({
        "n_queries": len(queries),
        "n_docs_fused": len(fused_docs),
        "context_chars": len(context),
        "answer_chars": len(answer),
    })

    # ---- (Opcional) Log a pestaña Prompts si está disponible ----
    if HAS_MLFLOW_PROMPTS:
        # Esquema simple esperado por mlflow.prompts (lista de interacciones)
        record = {
            "inputs": {
                "context": context,
                "question": base_query,
            },
            "outputs": {
                "answer": answer,
            },
            "metadata": {
                **provider_meta,
                "emb_model": EMB_MODEL_NAME,
                "top_k": k,
                "index_path": str(index_path),
                "queries_used": queries,
                "fused_docs_preview": [d.get("text", "")[:200] for d in fused_docs],
            },
        }
        try:
            mlflow_prompts.log_prompts(
                prompts=[record],
                name="RAG_Clinico_Fusion",
                context="clinical_rag_session"
            )
        except Exception as e:
            # Fallback silencioso si la firma exacta difiere en tu versión
            mlflow.log_text(json.dumps(record, ensure_ascii=False, indent=2), artifact_file="rag/prompt_record_fallback.json")


def run_rag_fusion(disease: str, edad: int, sexo: str, k: int = 6):
    # 1) Selección del índice
    index_path = INDEX_PATHS.get(disease)
    if not index_path or not index_path.exists():
        raise SystemExit(f"❌ No existe índice para {disease} en {index_path}")

    jsonl_path = index_path.with_name(index_path.stem.replace("_index", "_chunks.jsonl"))

    # 2) Embeddings + FAISS
    model, index = load_faiss_index(index_path, EMB_MODEL_NAME)

    # 3) Construir query base
    base_query = (
        f"Paciente de {edad} años, sexo {sexo}, con diagnóstico de {disease}. "
        f"¿Qué recomendaciones clínicas debo considerar?"
    )

    # 4) Variantes de query
    queries = generate_query_variants(base_query, n_variants=3)

    # 5) Recuperación
    results_per_query = [retrieve_chunks(q, model, index, jsonl_path, k=k) for q in queries]

    # 6) Fusión
    fused_docs = reciprocal_rank_fusion(results_per_query, top_k=k)

    # 7) Contexto final
    context = "\n\n".join([d.get("text", "") for d in fused_docs])

    # 8) LLM respuesta
    llm = ChatOpenAI(model_name=OPENAI_MODEL, openai_api_key=get_openai_api_key(), temperature=OPENAI_TEMPERATURE)
    final_prompt = prompt.format(context=context, question=base_query)
    answer = llm.predict(final_prompt)

    # 9) Registrar en MLflow
    provider_meta = {
        "provider": "openai",
        "model": OPENAI_MODEL,
        "temperature": OPENAI_TEMPERATURE,
    }
    log_prompt_and_response_to_mlflow(
        base_query=base_query,
        context=context,
        answer=answer,
        provider_meta=provider_meta,
        queries=queries,
        fused_docs=fused_docs,
        index_path=index_path,
        k=k,
    )

    # 10) Mostrar por consola (opcional)
    print("\n=== Consulta base ===")
    print(base_query)
    print("\n=== Respuesta ===\n", answer)


if __name__ == "__main__":
    # Inicializa MLflow y arranca un run
    init_mlflow()
    # Cierra cualquier run “huérfano” que pudiera estar abierto
    if mlflow.active_run():
        mlflow.end_run()

    run_name = "RAG_retinopatia_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=run_name):
        # Ejemplo
        mlflow.log_params({"disease": "retinopatia", "edad": 46, "sexo": "Mujer"})
        run_rag_fusion("miopia", edad=46, sexo="Mujer", k=6)
