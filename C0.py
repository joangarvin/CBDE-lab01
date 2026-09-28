import time
import statistics
import chromadb
from itertools import islice

# 1. Configurar/Conectar cliente de Chroma
# Se recomienda usar PersistentClient para mantener los datos almacenados en disco
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Si la colección ya existe de pruebas previas, la borramos para empezar limpios
try:
    chroma_client.delete_collection(name="book_corpus")
except Exception:
    pass

# Crear la colección. 
# NOTA: Para C0 (solo carga de texto/documentos), se puede desactivar la generación 
# automática de embeddings o pasar embedding_function=None para aislar el proceso.
collection = chroma_client.create_collection(
    name="book_corpus",
    embedding_function=None  # Desactiva la generación automática para medir inserción de texto puro
)

# 2. Cargar tus oraciones
# IMPORTANTE: Asegúrate de cargar EXACTAMENTE las mismas oraciones utilizadas en [P0]
def load_sentences(file_path, max_sentences=10_000):
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in islice(f, 10_000):
            limpia = line.strip()
            if limpia:
                sentences.append(limpia)
                if len(sentences) == max_sentences:
                    break
    if not sentences:
        raise ValueError("No se han encontrado frases.")
    return sentences

sentences = load_sentences("bookcorpus_10000_lineas.txt")

# 3. Insertar los textos en Chroma y medir tiempos de inserción
insertion_times = []
batch_size = 100  # Puedes ajustar el tamaño del batch (por lote)!!!!

print(f"Cargando {len(sentences)} oraciones en Chroma...")

for i in range(0, len(sentences), batch_size):
    batch_sentences = sentences[i:i + batch_size]
    # Generar IDs únicos para cada oración
    batch_ids = [f"sent_{j}" for j in range(i, i + len(batch_sentences))]

    start_time = time.perf_counter()
    
    # Insertar únicamente texto e IDs en la colección
    collection.add(
        documents=batch_sentences,
        ids=batch_ids
    )
    
    end_time = time.perf_counter()
    insertion_times.append(end_time - start_time)

# 4. Calcular y mostrar las métricas de tiempo requeridas para [CQ1]
min_time = min(insertion_times)
max_time = max(insertion_times)
avg_time = statistics.mean(insertion_times)
std_dev = statistics.stdev(insertion_times) if len(insertion_times) > 1 else 0.0

print("\n--- Tiempos de almacenamiento de texto [C0] ---")
print(f"Mínimo: {min_time:.6f} s")
print(f"Máximo: {max_time:.6f} s")
print(f"Promedio: {avg_time:.6f} s")
print(f"Desviación Estándar: {std_dev:.6f} s")