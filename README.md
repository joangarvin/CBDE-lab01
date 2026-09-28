# CBDE — Laboratori 01: Bases de dades vectorials

Repositori de lliurament del laboratori 01 de CBDE. Conté els programes de la pràctica d’emmagatzematge de frases, generació d’embeddings i cerques per similitud amb PostgreSQL i Chroma.

## Fitxers

| Fitxer | Descripció |
| --- | --- |
| `P0.py` | Carrega les frases del corpus a PostgreSQL i mesura els temps d’inserció per lots. |
| `P1.py` | Genera els embeddings de les frases amb el model `all-MiniLM-L6-v2`, els desa a PostgreSQL i mesura els temps d’emmagatzematge. |
| `P2.py` | Recupera els embeddings de PostgreSQL i fa cerques en memòria amb les distàncies euclidiana i Manhattan. Mostra les dues frases més properes a cada consulta i els temps de cerca. |
| `C0.py` | Prepara la col·lecció `book_corpus` a Chroma i intenta carregar els textos sense generar embeddings automàticament, mesurant els temps d’inserció. |
| `C1.py` | Genera els embeddings dels documents amb el model `all-MiniLM-L6-v2`, els desa a Chroma i mostra els temps d’emmagatzematge. |
| `C2.py` | Fa cerques per similitud a Chroma amb les mètriques L2 i cosinus, i mostra les estadístiques dels temps de consulta. |
| `bookcorpus_10000_lineas.txt` | Chunk de bookcorpus usat a la pràctica. |
