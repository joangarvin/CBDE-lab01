import time
import statistics
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch

# 1. Configurar/Conectar al cliente persistente de Chroma existente de C0
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 2. Configurar el modelo de embeddings con detección automática de GPU (CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cargando modelo en dispositivo: {device}")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# 3. Obtener la colección 'book_corpus' ya creada
try:
    collection = chroma_client.get_collection(name="book_corpus")
except Exception as e:
    raise RuntimeError("No se encontró la colección 'book_corpus'. Asegúrate de haber ejecutado C0.py primero.") from e

# 4. Obtener todos los documentos previamente insertados
results = collection.get(include=["documents"])
existing_ids = results["ids"]
existing_docs = results["documents"]

if not existing_ids:
    raise ValueError("La colección está vacía. Debes cargar el texto primero ejecutando C0.py.")

print(f"Total de oraciones recuperadas de Chroma: {len(existing_docs)}")

# 5. Generar y almacenar embeddings con lotes más grandes (OPTIMIZACIÓN: batch_size = 1000)
batch_size = 100
generation_times = []
insertion_times = []

print(f"Procesando en lotes de {batch_size} elementos...")

for i in range(0, len(existing_docs), batch_size):
    batch_sentences = existing_docs[i:i + batch_size]
    batch_ids = existing_ids[i:i + batch_size]

    # --- FASE 1: Generación de Embeddings en Memoria (CPU/GPU) ---
    start_gen_time = time.perf_counter()
    
    # encode() genera los vectores; lo convertimos a lista nativa para Chroma
    embeddings = model.encode(batch_sentences, batch_size=128, show_progress_bar=False).tolist()
    
    end_gen_time = time.perf_counter()
    generation_times.append(end_gen_time - start_gen_time)

    # --- FASE 2: Inserción / Actualización en Chroma (I/O) ---
    start_ins_time = time.perf_counter()
    
    # OPTIMIZACIÓN: Solo enviamos ids y embeddings. Omitimos 'documents' para no recargar la red/disco.
    collection.update(
        ids=batch_ids,
        embeddings=embeddings
    )
    
    end_ins_time = time.perf_counter()
    insertion_times.append(end_ins_time - start_ins_time)

# 6. Calcular y mostrar las métricas requeridas por el laboratorio
min_time = min(insertion_times)
max_time = max(insertion_times)
avg_time = statistics.mean(insertion_times)
std_dev = statistics.stdev(insertion_times) if len(insertion_times) > 1 else 0.0

print("\n--- Metrics for storing the embeddings por lote[C1] ---")
print(f"Minimum time: {min_time:.6f} s")
print(f"Maximum time: {max_time:.6f} s")
print(f"Average time: {avg_time:.6f} s")
print(f"Standard deviation: {std_dev:.6f} s")