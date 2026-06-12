"""Eixo Contextual: recuperação de condições de produção (RAG).

Recuperação por TF-IDF e similaridade de cosseno, conforme a linha
metodológica da pesquisa. A base deve cobrir os referentes culturais dos
casos e incluir, de propósito, documentos distratores (temas alheios),
cuja função é testar não só a recuperação correta, mas também a rejeição
do irrelevante.
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PASTA_PADRAO = Path(__file__).resolve().parent.parent / "data" / "base_rag"


def carregar_base(pasta=None):
    """Lê documentos .txt e .json da pasta da base contextual.

    Formato .json esperado: lista de {"id": ..., "titulo": ..., "texto": ...}.
    Arquivos .txt entram inteiros como um documento cada.
    """
    pasta = Path(pasta) if pasta else PASTA_PADRAO
    docs = []
    for arq in sorted(pasta.glob("*.json")):
        for d in json.loads(arq.read_text(encoding="utf-8")):
            docs.append(
                {
                    "id": str(d.get("id", arq.stem)),
                    "titulo": d.get("titulo", arq.stem),
                    "texto": d["texto"],
                }
            )
    for arq in sorted(pasta.glob("*.txt")):
        docs.append({"id": arq.stem, "titulo": arq.stem, "texto": arq.read_text(encoding="utf-8")})
    return docs


class RecuperadorContextual:
    """Indexa a base e recupera os k trechos mais similares ao enunciado."""

    def __init__(self, documentos, k=3):
        if not documentos:
            raise ValueError(
                "Base RAG vazia. Adicione os documentos de contexto em data/base_rag/ "
                "(ver data/base_rag/README.md)."
            )
        self.documentos = documentos
        self.k = k
        # token_pattern inclui tokens de um caractere: referencias culturais
        # como "7 a 1" dependem de digitos isolados.
        self.vetorizador = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.matriz = self.vetorizador.fit_transform(d["texto"] for d in documentos)

    def recuperar(self, enunciado, k=None):
        k = k or self.k
        consulta = self.vetorizador.transform([enunciado])
        sims = cosine_similarity(consulta, self.matriz)[0]
        ordem = sims.argsort()[::-1][:k]
        return [
            {**self.documentos[i], "similaridade": float(sims[i])}
            for i in ordem
        ]
