import statistics
import time
from itertools import islice

import psycopg2
from nltk.tokenize import sent_tokenize
from config import load_config

# 1. Leer una parte del archivo de texto de BookCorpus.
with open("data/bookcorpus_10000_lineas.txt", "r", encoding="utf-8") as f:
    text = "".join(islice(f, 10_000)) # islice permet obtenir les 10000 primeres linies, .join les uneix a un text conjunt

# 2. Dividir en chunks (párrafos) y frases.
chunks = []
# text.split("\n\n") divideix el text on hi ha dos salts de linea seguits
# chunk.strip() elimina salts de linea i espais
for chunk in text.split("\n\n"):
    limpio = chunk.strip()

    if limpio:
        chunks.append(limpio)

rows = []

for chunk_id, chunk in enumerate(chunks, 1):
    frases = sent_tokenize(chunk, language="english")

    for sentence in frases:
        limpia = sentence.strip()

        if limpia:
            rows.append((chunk_id, limpia))

rows = rows[:10_000]

if not rows:
    raise ValueError("No se han encontrado frases.")

# Crear taules i insertar a postgresSQL
conn = psycopg2.connect(**load_config())
tiempos = []
TAM_LOTE = 100

try:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS book_sentences (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                chunk_id INT,
                sentence TEXT
            );
        """)
        conn.commit()

        for inicio in range(0, len(rows), TAM_LOTE):
            lote = rows[inicio:inicio + TAM_LOTE]
            comienzo = time.perf_counter()

            cur.executemany(
                "INSERT INTO book_sentences (chunk_id, sentence) VALUES (%s, %s);",
                lote,
            )
            conn.commit()
            tiempos.append(time.perf_counter() - comienzo)
finally:
    conn.close()

# Mostrar les medicions
print(f"Frases insertadas: {len(rows)}")
print("Tiempos por lote de hasta 100 frases, incluyendo commit:")
print(f"Mínimo: {min(tiempos):.6f} s")
print(f"Máximo: {max(tiempos):.6f} s")
print(f"Media: {statistics.mean(tiempos):.6f} s")
print(f"Desviación estándar: {statistics.pstdev(tiempos):.6f} s")