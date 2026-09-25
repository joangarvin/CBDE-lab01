import statistics
import time

import numpy as np
import psycopg2
from config import load_config

# Conexión a la BD.
conn = psycopg2.connect(**load_config())
inicio = time.perf_counter()

try:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, sentence, embedding
            FROM book_sentences
            ORDER BY id
        """)
        filas = cur.fetchall()
finally:
    conn.close()

print(f"Tiempo de lectura y cierre: {time.perf_counter() - inicio:.6f} s")

# Preparar cada vector individualmente.
inicio = time.perf_counter()
datos = []

for identificador, texto, embedding in filas:
    if embedding is None or len(embedding) != 384:
        raise ValueError("Hay embeddings ausentes o incorrectos. Revisa P1.")

    vector = np.array(embedding, dtype=np.float64)

    if not np.isfinite(vector).all():
        raise ValueError("Hay un embedding no válido.")

    datos.append((identificador, texto, vector))

print(f"Tiempo de preparación: {time.perf_counter() - inicio:.6f} s")

# Elegir diez frases con textos diferentes.
consultas = []
vistos = set()

for dato in datos:
    texto = dato[1]

    if texto not in vistos:
        consultas.append(dato)
        vistos.add(texto)

    if len(consultas) == 10:
        break

if len(consultas) < 10:
    raise ValueError("Se necesitan al menos diez textos diferentes.")

# Realizar las búsquedas con las dos métricas.
for metrica in ["euclidea", "manhattan"]:
    tiempos = []
    print(f"\n--- DISTANCIA {metrica.upper()} ---")

    # Recorrer las diez frases escogidas.
    for id_consulta, texto_consulta, vector_consulta in consultas:
        inicio = time.perf_counter()
        resultados = []

        # Comparar con todas las demás frases.
        for identificador, texto, vector in datos:
            if identificador == id_consulta:
                continue

            if metrica == "euclidea":
                distancia = np.linalg.norm(vector - vector_consulta)
            else:
                distancia = np.sum(np.abs(vector - vector_consulta))

            resultados.append((float(distancia), identificador, texto))

        # Ordenar
        resultados.sort()

        # Seleccionar los dos más cercanos.
        mejores = resultados[:2]
        tiempos.append(time.perf_counter() - inicio)

        print(f"\nConsulta ID {id_consulta}: {texto_consulta}")
        for distancia, identificador, texto in mejores:
            print(
                f"  ID {identificador} | "
                f"Distancia: {distancia:.6f} | {texto}"
            )

    # Mostrar estadísticas por métrica.
    print("\nTiempos por búsqueda en memoria:")
    print(f"Mínimo: {min(tiempos):.6f} s")
    print(f"Máximo: {max(tiempos):.6f} s")
    print(f"Media: {statistics.mean(tiempos):.6f} s")
    print(f"Desviación estándar: {statistics.pstdev(tiempos):.6f} s")