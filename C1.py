import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import time
import statistics
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# 1. Configurar/Conectar al cliente persistente de Chroma existente de C0
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# 2. Configurar el modelo de embeddings ligero sugerido en la práctica (all-MiniLM-L6-v2)
# Nota: Si no tienes instalado sentence_transformers, ejecútalo con: pip install sentence-transformers
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 3. Obtener la colección 'book_corpus' ya creada en C0.py
try:
    collection = chroma_client.get_collection(
        name="book_corpus",
        embedding_function=sentence_transformer_ef
    )
except Exception as e:
    raise RuntimeError("No se encontró la colección 'book_corpus'. Asegúrate de haber ejecutado C0.py primero.") from e

# 4. Obtener todos los documentos previamente insertados en C0
results = collection.get(include=["documents"])
existing_ids = results["ids"]
existing_docs = results["documents"]

if not existing_ids:
    raise ValueError("La colección está vacía. Debes cargar el texto primero ejecutando C0.py.")

print(f"Total de oraciones recuperadas de Chroma: {len(existing_docs)}")

# 5. Generar y almacenar embeddings por lotes (TAM_LOTE = 100 para consistencia con P1/P0)
batch_size = 100
batch_times = []
per_sentence_times = []

print(f"Generando y almacenando embeddings en lotes de {batch_size}...")

for i in range(0, len(existing_docs), batch_size):
    batch_sentences = existing_docs[i:i + batch_size]
    batch_ids = existing_ids[i:i + batch_size]
    current_batch_len = len(batch_sentences)

    start_time = time.perf_counter()
    
    # Al llamar a collection.update, Chroma genera los embeddings usando 
    # embedding_function y actualiza los registros existentes por ID
    collection.update(
        ids=batch_ids,
        documents=batch_sentences
    )
    
    end_time = time.perf_counter()
    
    elapsed_batch = end_time - start_time
    batch_times.append(elapsed_batch)
    
    # Tiempo imputado por cada frase en este lote
    elapsed_per_sentence = elapsed_batch / current_batch_len
    per_sentence_times.extend([elapsed_per_sentence] * current_batch_len)

# 6. Calcular y mostrar las métricas requeridas para [CQ1]
print("\n--- Tiempos de almacenamiento de embeddings [C1] (POR LOTE DE 100) ---")
print(f"Mínimo: {min(batch_times):.6f} s")
print(f"Máximo: {max(batch_times):.6f} s")
print(f"Promedio: {statistics.mean(batch_times):.6f} s")
print(f"Desviación Estándar: {statistics.stdev(batch_times) if len(batch_times) > 1 else 0.0:.6f} s")

print("\n--- Tiempos de almacenamiento de embeddings [C1] (POR FRASE INDIVIDUAL) ---")
print(f"Mínimo por frase: {min(per_sentence_times):.6f} s")
print(f"Máximo por frase: {max(per_sentence_times):.6f} s")
print(f"Promedio por frase: {statistics.mean(per_sentence_times):.6f} s")
print(f"Desviación Estándar: {statistics.stdev(per_sentence_times) if len(per_sentence_times) > 1 else 0.0:.6f} s")