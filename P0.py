import statistics
import time
from itertools import islice

import psycopg2
from config import load_config

# 1. Leer directamente las primeras 10.000 frases del fichero (una por línea)
frases = []
with open("data/bookcorpus_10000_lineas.txt", "r", encoding="utf-8") as f:
    for line in islice(f, 10_000):
        limpia = line.strip()
        if limpia:
            frases.append((limpia,))  # Tupla de 1 elemento para psycopg2

if not frases:
    raise ValueError("No se han encontrado frases.")

# Crear taules i insertar a postgresSQL
conn = psycopg2.connect(**load_config())
tiempos = []
TAM_LOTE = 100

try:
    with conn.cursor() as cur:
        # Creamos la tabla sin la columna chunk_id
        cur.execute("""
            CREATE TABLE IF NOT EXISTS book_sentences (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                sentence TEXT
            );
        """)
        conn.commit()

       # Mirar si la taula està buïda
        cur.execute("SELECT 1 FROM book_sentences LIMIT 1")

        if cur.fetchone() is not None:
            raise ValueError("La tabla ya contiene datos, borrala primero")

        for inicio in range(0, len(frases), TAM_LOTE):
            lote = frases[inicio:inicio + TAM_LOTE]
            comienzo = time.perf_counter()

            cur.executemany(
                "INSERT INTO book_sentences (sentence) VALUES (%s);",
                lote,
            )
            conn.commit()
            tiempos.append(time.perf_counter() - comienzo)
finally:
    conn.close()

# Mostrar les medicions
print(f"Frases insertadas: {len(frases)}")
print("Tiempos por lote de hasta 100 frases, incluyendo commit:")
print(f"Mínimo: {min(tiempos):.6f} s")
print(f"Máximo: {max(tiempos):.6f} s")
print(f"Media: {statistics.mean(tiempos):.6f} s")
print(f"Desviación estándar: {statistics.pstdev(tiempos):.6f} s")