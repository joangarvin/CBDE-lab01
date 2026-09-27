import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import time
import statistics
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# 1. Conectar al cliente persistente de Chroma
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# Configurar el modelo de embeddings usado en C1
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 2. Obtener la colección existente de C0 / C1
try:
    collection = chroma_client.get_collection(
        name="book_corpus",
        embedding_function=sentence_transformer_ef
    )
except Exception as e:
    raise RuntimeError("Error al obtener la colección. Asegúrate de haber ejecutado C0.py y C1.py primero.") from e

# 3. Seleccionar las primeras 10 oraciones para las consultas
sample_data = collection.get(limit=10)
sample_docs = sample_data["documents"]

if len(sample_docs) < 10:
    raise ValueError("Se requieren al menos 10 oraciones en la colección para ejecutar C2.")

print("=== 10 Oraciones Seleccionadas para las Consultas [C2] ===")
for idx, doc in enumerate(sample_docs, 1):
    print(f"{idx}. {doc}")
print("=" * 55 + "\n")

# 4. Medir tiempos de consulta Top-2
query_times = []

print("Calculando Top-2 oraciones más similares para las 10 oraciones de muestra...")

for i, doc in enumerate(sample_docs, 1):
    start_time = time.perf_counter()
    
    # n_results=3 porque el primer resultado es la propia oración consultada
    results = collection.query(
        query_texts=[doc],
        n_results=3
    )
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    query_times.append(elapsed_time)

# 5. Calcular métricas de tiempo para [CQ1]
min_time = min(query_times)
max_time = max(query_times)
avg_time = statistics.mean(query_times)
std_dev = statistics.stdev(query_times) if len(query_times) > 1 else 0.0

print("\n--- Tiempos de consulta de similitud [C2] ---")
print(f"Mínimo: {min_time:.6f} s")
print(f"Máximo: {max_time:.6f} s")
print(f"Promedio: {avg_time:.6f} s")
print(f"Desviación Estándar: {std_dev:.6f} s")