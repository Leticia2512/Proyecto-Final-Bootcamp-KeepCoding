
#RAG CLASICO 
''' 

from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1) Cargar el vectorstore 
embedding = ...  
db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

# 2) Retriever
retriever = db.as_retriever(search_kwargs={"k": 6})

# 3) Configurar LLM (OpenAI vía LangChain)
openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_key, temperature=0.0)

# 4) Prompt template (A DESARROLLAR)
template = """
Responde en español. Desarrollar. 
Contexto:
{context}

Pregunta:
{question}

Respuesta (clara y concisa):
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# 5) RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# 6) Ejemplo de consulta adaptado a la salida del DL

# Simulación de salida del modelo DL (ejemplo)
edad = 46
sexo = "Mujer"
diagnostico = "Retinopatía diabética"

# Construcción automática de la query
question = (
    f"Paciente de {edad} años, sexo {sexo}, con diagnóstico de {diagnostico}. "
    f"¿Qué recomendaciones iniciales se deben dar según las guías clínicas?"
)

# Llamada al RAG
resp = qa.invoke({"query": question})

# Mostrar resultados
print("Consulta generada:\n", question)
print("\nRespuesta:\n", resp["result"])

print("\nFuentes utilizadas:")
for i, d in enumerate(resp["source_documents"], 1):
    print(i, d.metadata.get("source", ""), d.page_content[:200], "...")
'''
#RAG FUSION 

#1
'''
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

# 1) Cargar vectorstore ,indicar la ruta , hay que adaptar por los indices segun enfermedad ?
embedding = ...
db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

# 2) Configurar retriever Podemos actuar aumentando el K para ampliar el contexto , se hace una vez por cada Query en el Fusion
retriever = db.as_retriever(search_kwargs={"k": 6})

# 3) Configurar LLM
openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_key, temperature=0.0)

# 4) Prompt template
template = """
Responde en español. Usa exclusivamente el contexto.
Contexto:
{context}

Pregunta:
{question}

Respuesta (clara y concisa, citando guías):
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# --- Funciones para RAG-Fusion ---
def generate_query_variants(base_query: str, n_variants: int = 3):
    """Genera variantes de la query base usando el LLM"""
    variant_prompt = f"Genera {n_variants} variantes semánticas de la siguiente consulta:\n\n{base_query}"
    resp = llm.predict(variant_prompt)
    return [base_query] + [q.strip("-• \n") for q in resp.split("\n") if q.strip()]

def reciprocal_rank_fusion(results_per_query, top_k=6):
    """Fusiona resultados de varias queries (muy simple: dedup + top_k)"""
    seen = {}
    fused = []
    for docs in results_per_query:
        for d in docs:
            if d.page_content not in seen:
                seen[d.page_content] = d
                fused.append(d)
    return fused[:top_k]

# 5) Ejemplo de simulación con salida del DL
edad = 46
sexo = "Mujer"
diagnostico = "retinopatia" 
base_query = (
    f"Paciente de {edad} años, sexo {sexo}, con diagnóstico de {diagnostico}. "
    f"¿Qué recomendaciones iniciales se deben dar según las guías clínicas?"
)

# 6) Generar variantes de query
queries = generate_query_variants(base_query, n_variants=3)
print("🔎 Variantes de query generadas:")
for q in queries:
    print("-", q)

# 7) Recuperar documentos para cada query
results_per_query = [retriever.get_relevant_documents(q) for q in queries]

# 8) Fusionar resultados
fused_docs = reciprocal_rank_fusion(results_per_query, top_k=6)

# 9) Construir contexto
context = "\n\n".join([d.page_content for d in fused_docs])

# 10) Prompt final y respuesta
final_prompt = prompt.format(context=context, question=base_query)
answer = llm.predict(final_prompt)

# 11) Mostrar
print("\nConsulta base:\n", base_query)
print("\nRespuesta:\n", answer)
print("\nFuentes usadas:")
for i, d in enumerate(fused_docs, 1):
    print(i, d.metadata.get("source", ""), d.page_content[:200], "...")

#RAG FUSION CON Hybrid Retrieval
#Garantiza una busqueda con keywords que puedan estar referidas a las dolencias
'''
#2

#!/usr/bin/env python3
"""
RAG-Fusion adaptado a multi-índice por dolencia
Usa índices FAISS creados con sentence-transformers
"""

import os
import pathlib
from typing import List
from secrets_ApiKey import get_openai_api_key

import faiss  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ------------------ Configuración ------------------

BASE_DIR = pathlib.Path(__file__).resolve().parent

INDEX_PATHS = {
    "dmae": BASE_DIR / "vectorstores" / "david" / "dmae_index.faiss",
    "catarata": BASE_DIR / "vectorstores" / "david" / "catarata_index.faiss",
    "retinopatia": BASE_DIR / "vectorstores" / "david" / "retinopatia_index.faiss",
    "miopia": BASE_DIR / "vectorstores" / "david" / "miopia_index.faiss",
}

# Modelo de embeddings que usaste al crear los índices
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM de OpenAI
openai_key = get_openai_api_key()
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_key, temperature=0.0)

# Prompt orientado a recomendaciones clínicas
template = """
Eres un asistente especializado en oftalmología. 
Responde en español y usa exclusivamente el contexto dado.

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

# ------------------ Funciones ------------------

def load_faiss_index(index_path: pathlib.Path, model_name: str):
    """Carga índice FAISS creado con sentence-transformers"""
    model = SentenceTransformer(model_name)
    index = faiss.read_index(str(index_path))
    return model, index

def retrieve_chunks(query: str, model: SentenceTransformer, index: faiss.Index, docs_jsonl: pathlib.Path, k: int = 6):
    """Busca los chunks más relevantes para una query"""
    import json
    # cargar chunks en memoria
    chunks = []
    with docs_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    # emb query
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb, k)
    results = [chunks[i] for i in I[0] if i < len(chunks)]
    return results

def generate_query_variants(base_query: str, n_variants: int = 3) -> List[str]:
    """Genera variantes de la query base usando LLM"""
    variant_prompt = f"Genera {n_variants} variantes semánticas de la siguiente consulta:\n\n{base_query}"
    resp = llm.predict(variant_prompt)
    return [base_query] + [q.strip("-• \n") for q in resp.split("\n") if q.strip()]

def reciprocal_rank_fusion(results_per_query: List[List[dict]], top_k: int = 6):
    """Fusiona resultados de varias queries (dedup + top_k)"""
    seen = {}
    fused = []
    for docs in results_per_query:
        for d in docs:
            if d["text"] not in seen:
                seen[d["text"]] = d
                fused.append(d)
    return fused[:top_k]

# ------------------ Main ------------------

def run_rag_fusion(disease: str, edad: int, sexo: str):
    # 1. Selección del índice correcto
    index_path = INDEX_PATHS.get(disease)
    if not index_path or not index_path.exists():
        raise SystemExit(f"❌ No existe índice para {disease}")

    jsonl_path = index_path.with_name(index_path.stem.replace("_index", "_chunks.jsonl"))

    model, index = load_faiss_index(index_path, EMB_MODEL_NAME)

    # 2. Construir query base
    base_query = (
        f"Paciente de {edad} años, sexo {sexo}, con diagnóstico de {disease}. "
        f"¿Qué recomendaciones clínicas debo considerar?"
    )

    # 3. Generar variantes de query
    queries = generate_query_variants(base_query, n_variants=3)

    # 4. Recuperar docs por cada query
    results_per_query = [retrieve_chunks(q, model, index, jsonl_path, k=6) for q in queries]

    # 5. Fusionar resultados
    fused_docs = reciprocal_rank_fusion(results_per_query, top_k=6)

    # 6. Construir contexto
    context = "\n\n".join([d["text"] for d in fused_docs])

    # 7. Prompt final + respuesta
    final_prompt = prompt.format(context=context, question=base_query)
    answer = llm.predict(final_prompt)

    # 8. Mostrar resultados
    print("Consulta base:", base_query)
    print("\nRespuesta:\n", answer)
    print("\nFuentes:")
    for i, d in enumerate(fused_docs, 1):
        print(i, d["text"][:200], "...")


if __name__ == "__main__":
    # Ejemplo de simulación con salida del modelo DL
    run_rag_fusion("retinopatia", edad=46, sexo="Mujer")


'''
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.docstore.document import Document
import os

# 1) Cargar vectorstore
embedding = ...
db = FAISS.load_local("vectorstores/faiss_david", embedding, allow_dangerous_deserialization=True)
retriever_vector = db.as_retriever(search_kwargs={"k": 6})

# 2) Cargar documentos para keyword search (BM25)
# ⚠️ Necesitas la lista de documentos originales (los chunks antes de FAISS)
docs = db.docstore._dict.values()   # hack: obtener los Document guardados en FAISS
retriever_keyword = BM25Retriever.from_documents(list(docs))
retriever_keyword.k = 6

# 3) Ensemble (fusiona vector + keyword)
retriever = EnsembleRetriever(
    retrievers=[retriever_vector, retriever_keyword],
    weights=[0.5, 0.5]
)

# 4) LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0)

# 5) Prompt
template = """
Responde en español, usando solo la información del contexto.
Contexto:
{context}

Pregunta:
{question}

Respuesta (clara, con citas de guías):
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# 6) QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# 7) Ejemplo consulta
question = "Paciente mujer de 46 años con diagnóstico de retinopatía diabética leve. ¿Qué recomendaciones iniciales indican las guías?"
resp = qa.invoke({"query": question})

print("\nRespuesta:\n", resp["result"])
print("\nFuentes usadas:")
for i, d in enumerate(resp["source_documents"], 1):
    print(i, d.metadata.get("source",""), d.page_content[:200], "...")

'''

