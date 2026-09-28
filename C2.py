import time
import statistics
import chromadb
from chromadb.config import Settings
import torch
from sentence_transformers import SentenceTransformer

# 1. Conectar al cliente persistente de Chroma
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
)

# Configurar el modelo localmente para aceleración (GPU si está disponible)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# 2. Obtener la colección principal (Métrica 1: L2 por defecto en Chroma)
try:
    collection_l2 = chroma_client.get_collection(name="book_corpus")
except Exception as e:
    raise RuntimeError("Error al obtener la colección. Ejecuta C0.py y C1_v2.py primero.") from e

# 3. Seleccionar 10 oraciones para las consultas[cite: 1, 4]
sample_data = collection_l2.get(limit=10)
sample_ids = sample_data["ids"]
sample_docs = sample_data["documents"]

if len(sample_docs) < 10:
    raise ValueError("Se requieren al menos 10 oraciones en la colección.")

print("=== 10 Oraciones Seleccionadas para las Consultas [C2] ===")
for idx, doc in enumerate(sample_docs, 1):
    print(f"{idx}. {doc}")
print("=" * 55 + "\n")

# Pre-calcular los vectores de consulta (batch_size ajustado a 100)
query_embeddings = model.encode(sample_docs, batch_size=100, show_progress_bar=False).tolist()

# ---------------------------------------------------------
# Métrica 1: Búsqueda usando L2 (Distancia Euclidiana)[cite: 1, 4]
# ---------------------------------------------------------
times_l2 = []
print("Calculando Top-2 usando Métrica 1 (L2 / Distancia Euclidiana)...")

for i in range(10):
    start_time = time.perf_counter()
    
    results = collection_l2.query(
        query_embeddings=[query_embeddings[i]],
        n_results=3
    )
    
    end_time = time.perf_counter()
    times_l2.append(end_time - start_time)

# ---------------------------------------------------------
# Métrica 2: Búsqueda usando Cosine Similarity[cite: 1, 4]
# ---------------------------------------------------------
print("\nPreparando Métrica 2 (Similitud Coseno). Clonando colección en lotes de 100...")
try:
    chroma_client.delete_collection(name="book_corpus_cosine")
except Exception:
    pass

collection_cosine = chroma_client.create_collection(
    name="book_corpus_cosine",
    metadata={"hnsw:space": "cosine"}
)

all_data = collection_l2.get(include=["documents", "embeddings"])
all_ids = all_data["ids"]
all_embeddings = all_data["embeddings"]
all_documents = all_data["documents"]

# Inserción por lotes de 100 para la colección clonada
batch_size = 100
for i in range(0, len(all_ids), batch_size):
    collection_cosine.add(
        ids=all_ids[i:i+batch_size],
        embeddings=all_embeddings[i:i+batch_size],
        documents=all_documents[i:i+batch_size]
    )

times_cosine = []
print("Calculando Top-2 usando Métrica 2 (Similitud Coseno)...")

for i in range(10):
    start_time = time.perf_counter()
    
    results = collection_cosine.query(
        query_embeddings=[query_embeddings[i]],
        n_results=3
    )
    
    end_time = time.perf_counter()
    times_cosine.append(end_time - start_time)

# ---------------------------------------------------------
# 4. Calcular métricas de tiempo
# ---------------------------------------------------------
def print_metrics(times_list, metric_name):
    min_time = min(times_list)
    max_time = max(times_list)
    avg_time = statistics.mean(times_list)
    std_dev = statistics.stdev(times_list) if len(times_list) > 1 else 0.0

    print(f"\n--- Tiempos de consulta [C2] - {metric_name} ---")
    print(f"Mínimo: {min_time:.6f} s")
    print(f"Máximo: {max_time:.6f} s")
    print(f"Promedio: {avg_time:.6f} s")
    print(f"Desviación Estándar: {std_dev:.6f} s")

print_metrics(times_l2, "Métrica 1 (Distancia Euclidiana)")
print_metrics(times_cosine, "Métrica 2 (Coseno)")