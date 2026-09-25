import statistics
import time

import psycopg2
from sentence_transformers import SentenceTransformer
from config import load_config

# Carrega model all-MiniLM-L6-v2
modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#conexió BD
conn = psycopg2.connect(**load_config())
tiempos = []
TAM_LOTE = 100

try:
    with conn.cursor() as cur:
        
        # Añadir una columna para guardar los vectores.
        # Guarda los vectores en un array
        cur.execute("""
            ALTER TABLE book_sentences
            ADD COLUMN IF NOT EXISTS embedding REAL[]
        """)
        conn.commit()

        # Leer los identificadores y los textos guardados en P0.
        cur.execute("SELECT id, sentence FROM book_sentences ORDER BY id")
        filas = cur.fetchall()
        conn.commit()

        if not filas:
            raise ValueError("La tabla está vacía. Ejecuta P0 primero.")

        # Generar un embedding para cada frase.
        textos = [texto for identificador, texto in filas]
        embeddings = modelo.encode(textos, show_progress_bar=True)

        # Preparar parejas (vector, identificador) para actualizar las filas.
        datos = [
            (vector.tolist(), fila[0])
            for fila, vector in zip(filas, embeddings)
        ]

        # Guardar los embeddings y medir el almacenamiento.
        for inicio in range(0, len(datos), TAM_LOTE):
            # lote de 100 elements ex: 0:100, 100:200...
            lote = datos[inicio:inicio + TAM_LOTE]
            comienzo = time.perf_counter()

            # Els insertem a la taula
            cur.executemany(
                "UPDATE book_sentences SET embedding = %s WHERE id = %s",
                lote,
            )
            conn.commit()
            tiempos.append(time.perf_counter() - comienzo)

finally:
    conn.close()

# Mostrar las estadísticas.
print(f"Embeddings guardados: {len(datos)}")
print("Tiempos de almacenamiento por lote de hasta 100 embeddings:")
print(f"Mínimo: {min(tiempos):.6f} s")
print(f"Máximo: {max(tiempos):.6f} s")
print(f"Media: {statistics.mean(tiempos):.6f} s")
print(f"Desviación estándar: {statistics.pstdev(tiempos):.6f} s")