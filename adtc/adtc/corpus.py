"""Carregamento do corpus de demonstração v20f (41 casos, 164 itens).

O corpus é curado por opção: seu propósito não é medir desempenho
generalizável, mas estressar a capacidade do sistema de reconhecer a
ironia dependente de memória discursiva e de distingui-la da
literalidade, da ambiguidade e da indecidibilidade.
"""

import csv
from pathlib import Path

ROTULOS_VALIDOS = {"IRONICO", "NAO_IRONICO", "AMBIGUO", "INDECIDIVEL"}

CAMINHO_PADRAO = Path(__file__).resolve().parent.parent / "data" / "corpus_v20f.csv"


def carregar_corpus(caminho=None):
    """Lê o corpus e devolve uma lista de dicts {caso, enunciado, rotulo}."""
    caminho = Path(caminho) if caminho else CAMINHO_PADRAO
    casos = []
    with open(caminho, newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            rotulo = linha["rotulo"].strip().upper()
            if rotulo not in ROTULOS_VALIDOS:
                raise ValueError(f"Rótulo inválido em {linha['caso']}: {rotulo}")
            casos.append(
                {
                    "caso": linha["caso"].strip(),
                    "enunciado": linha["enunciado"].strip(),
                    "rotulo": rotulo,
                }
            )
    return casos
