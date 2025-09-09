import json
import pathlib
from typing import List
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from secrets_ApiKey import get_openai_api_key


# ================== CONFIG ==================
BASE_DIR = pathlib.Path(__file__).resolve().parent

INDEX_PATHS = {
    "dmae":        BASE_DIR / "vectorstores" / "dmae_index.faiss",
    "catarata":    BASE_DIR / "vectorstores" / "catarata_index.faiss",
    "retinopatia": BASE_DIR / "vectorstores" / "retinopatia_index.faiss",
    "miopia":      BASE_DIR / "vectorstores" / "miopia_index.faiss",
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
prompt = PromptTemplate(
    input_variables=["context", "question"], template=template)


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
    llm = ChatOpenAI(model_name=OPENAI_MODEL,
                     openai_api_key=get_openai_api_key(), temperature=0.0)
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


def run_rag_fusion(disease: str, edad: int, sexo: str, k: int = 6):

    # Caso de ojo normal
    if disease.lower() == "normal":
        answer = ("El análisis indica que se trata de un ojo donde no se detectan nuestas patologías.")

        return answer

    # 1) Selección del índice
    index_path = INDEX_PATHS.get(disease)

    if not index_path or not index_path.exists() or index_path == None:
        raise SystemExit(f"No existe índice para {disease} en {index_path}")

    jsonl_path = index_path.with_name(
        index_path.stem.replace("_index", "_chunks.jsonl"))

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
    results_per_query = [retrieve_chunks(
        q, model, index, jsonl_path, k=k) for q in queries]

    # 6) Fusión
    fused_docs = reciprocal_rank_fusion(results_per_query, top_k=k)

    # 7) Contexto final
    context = "\n\n".join([d.get("text", "") for d in fused_docs])

    # 8) LLM respuesta
    llm = ChatOpenAI(model_name=OPENAI_MODEL, openai_api_key=get_openai_api_key(
    ), temperature=OPENAI_TEMPERATURE)
    final_prompt = prompt.format(context=context, question=base_query)
    answer = llm.predict(final_prompt)

    return answer


if __name__ == "__main__":

    run_rag_fusion("normal", edad=46, sexo="Mujer", k=6)
