import statistics
import time
import chromadb
from itertools import islice

# 1. Leer directamente las primeras 10.000 frases (una por línea)
frases = []
with open("bookcorpus_10000_lineas.txt", "r", encoding="utf-8") as f:
    for line in islice(f, 10_000):
        limpia = line.strip()
        if limpia:
            frases.append(limpia)

if not frases:
    raise ValueError("No se han encontrado frases.")


#Crear el client de chroma
chroma_client = chromadb.PersistentClient(path="./data/chroma_db")

#Crear collecció
NOMBRE_COLECCION = "book_corpus"

# Si la colección ya existe, la eliminamos
try:
    chroma_client.delete_collection(name=NOMBRE_COLECCION)
except Exception:
    pass  # Si no existía, continúa sin error

# La creamos limpia
collection = chroma_client.create_collection(name=NOMBRE_COLECCION)


# Afegir les frases en lots de 100.
TAM_LOTE = 100
tiempos = []

print("Insertando las 10000 frases...")

for inicio in range(0, len(frases), TAM_LOTE):
    lote_textos = frases[inicio:inicio + TAM_LOTE]

    # Chroma necessita identificadors de tipus string.
    identificadores = [str(inicio + i + 1) for i in range(len(lote_textos))]

    comienzo = time.perf_counter()

    collection.add(
        ids=identificadores,
        documents=lote_textos,
    )

    tiempos.append(time.perf_counter() - comienzo)


# 4. Mostrar les estadístiques.
print(f"Frases guardades a Chroma: {collection.count()}")
print("Temps per lot: generació + emmagatzematge + indexació")
print(f"Mínim: {min(tiempos):.6f} s")
print(f"Màxim: {max(tiempos):.6f} s")
print(f"Mitjana: {statistics.mean(tiempos):.6f} s")
print(f"Desviació estàndard: {statistics.pstdev(tiempos):.6f} s")
