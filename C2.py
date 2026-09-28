import time
import statistics
import math
import chromadb
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
    raise RuntimeError("Error al obtener la colección. Ejecuta C0.py y C1.py primero.") from e

# 3. Seleccionar los primeros 10 textos distintos, como en P2.
# Los IDs de C0 (sent_0, sent_1, ...) indican el orden del corpus.
sample_data = collection_l2.get(include=["documents"])
ordered_docs = sorted(
    zip(sample_data["ids"], sample_data["documents"]),
    key=lambda item: int(item[0].split("_")[-1]),
)
sample_ids = []
sample_docs = []
seen = set()
for sentence_id, doc in ordered_docs:
    if doc not in seen:
        sample_ids.append(sentence_id)
        sample_docs.append(doc)
        seen.add(doc)
    if len(sample_docs) == 10:
        break

if len(sample_docs) < 10:
    raise ValueError("Se requieren al menos 10 textos diferentes en la colección.")

print("=== 10 Oraciones Seleccionadas para las Consultas [C2] ===")
for idx, (sentence_id, doc) in enumerate(zip(sample_ids, sample_docs), 1):
    print(f"{idx}. ID {sentence_id}: {doc}")
print("=" * 55 + "\n")

def print_top2(results, query_id, query_doc, metric):
    # Pedimos tres candidatos para poder excluir la propia frase por ID.
    # No basta con descartar el primero: puede haber textos repetidos.
    neighbors = [
        (sentence_id, doc, distance)
        for sentence_id, doc, distance in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        )
        if sentence_id != query_id
    ][:2]

    print(f"\nConsulta ID {query_id}: {query_doc}")
    for sentence_id, doc, distance in neighbors:
        # Chroma devuelve L2 al cuadrado; la raíz coincide con la euclídea de P2.
        if metric == "l2":
            distance = math.sqrt(max(0.0, distance))
        print(f"  ID {sentence_id} | Distancia: {distance:.6f} | {doc}")

# Pre-calcular los vectores de consulta (batch_size ajustado a 100)
query_embeddings = model.encode(sample_docs, batch_size=100, show_progress_bar=False).tolist()

# ---------------------------------------------------------
# Métrica 1: Búsqueda usando L2 (Distancia Euclidiana)
# ---------------------------------------------------------
times_l2 = []
print("Calculando Top-2 usando Métrica 1 (L2 / Distancia Euclidiana)...")

for i in range(10):
    start_time = time.perf_counter()
    
    results = collection_l2.query(
        query_embeddings=[query_embeddings[i]],
        n_results=3,
        include=["documents", "distances"]
    )
    
    end_time = time.perf_counter()
    times_l2.append(end_time - start_time)
    print_top2(results, sample_ids[i], sample_docs[i], "l2")

# ---------------------------------------------------------
# Métrica 2: Búsqueda usando distancia coseno (menor = más similar)
# ---------------------------------------------------------
print("\nPreparando Métrica 2 (Distancia Coseno). Clonando colección en lotes de 100...")
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
print("Calculando Top-2 usando Métrica 2 (Distancia Coseno)...")

for i in range(10):
    start_time = time.perf_counter()
    
    results = collection_cosine.query(
        query_embeddings=[query_embeddings[i]],
        n_results=3,
        include=["documents", "distances"]
    )
    
    end_time = time.perf_counter()
    times_cosine.append(end_time - start_time)
    print_top2(results, sample_ids[i], sample_docs[i], "cosine")

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
