# Base contextual (Eixo Contextual / RAG)

Coloque aqui os documentos da base de recuperação usados no experimento
v20f: os textos que cobrem os referentes culturais dos 41 casos
(pandemia, 7 a 1, padrão FIFA, "imagina na Copa", Cálice, lema da
bandeira, "ame-o ou deixe-o", tratamento precoce, Dom Casmurro etc.) e
os documentos distratores deliberados (música clássica, Segunda Guerra
Mundial), cuja função é testar a rejeição do irrelevante.

Formatos aceitos pelo `adtc/rag.py`:

- `*.txt` — um documento por arquivo;
- `*.json` — lista de objetos `{"id": ..., "titulo": ..., "texto": ...}`.

A base usada na tese está no notebook Kaggle v20f; exporte-a para cá
antes de rodar `scripts/rodar_experimento.py`.
