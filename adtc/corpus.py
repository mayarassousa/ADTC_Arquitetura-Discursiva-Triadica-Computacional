"""Carregamento e padronização do IDPT News."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .arquivos import ler_csv_robusto
from .integridade import hash_texto, normalizar_texto_hash


COLUNAS_TEXTO = (
    "text", "text_limpo", "texto", "texto_limpo", "noticia", "notícia",
    "conteudo", "conteúdo", "content", "body",
)
COLUNAS_ROTULO = ("label", "rotulo", "rótulo", "classe", "target", "y", "y_true")


@dataclass(slots=True)
class CorpusIDPT:
    treino: pd.DataFrame
    teste: pd.DataFrame
    coluna_texto_treino: str
    coluna_texto_teste: str


def achar_coluna(df: pd.DataFrame, candidatas: tuple[str, ...], tipo: str) -> str:
    for coluna in candidatas:
        if coluna in df.columns:
            return coluna
    raise ValueError(f"Nenhuma coluna de {tipo} encontrada. Colunas: {df.columns.tolist()}")


def normalizar_label(valor) -> int:
    if pd.isna(valor):
        raise ValueError("Rótulo ausente.")
    if isinstance(valor, (int, float, np.integer, np.floating)):
        inteiro = int(valor)
        if inteiro in (0, 1):
            return inteiro
    texto = str(valor).strip().strip('"').strip("'").lower()
    mapa = {
        "0": 0, "1": 1, "false": 0, "true": 1,
        "real": 0, "not_iro": 0, "not_ironic": 0, "non-ironic": 0,
        "nao_ironico": 0, "não_irônico": 0, "nao ironico": 0,
        "não irônico": 0, "ironic": 1, "ironico": 1, "irônico": 1,
        "satirical": 1, "satirico": 1, "satírico": 1,
        "satire": 1, "satira": 1, "sátira": 1,
    }
    if texto not in mapa:
        raise ValueError(f"Rótulo não reconhecido: {valor!r}")
    return mapa[texto]


def preparar_dataframe(df: pd.DataFrame, texto_col: str | None = None, label_col: str | None = None) -> pd.DataFrame:
    texto_col = texto_col or achar_coluna(df, COLUNAS_TEXTO, "texto")
    label_col = label_col or achar_coluna(df, COLUNAS_ROTULO, "rótulo")
    out = df.copy()
    out["text"] = out[texto_col].fillna("").astype(str)
    out["texto_modelo"] = out["text"].map(normalizar_texto_hash)
    out["label"] = out[label_col].map(normalizar_label).astype(int)
    out["text_hash"] = out["text"].map(hash_texto)
    if "id_texto_original" not in out.columns:
        out["id_texto_original"] = np.arange(len(out))
    return out


def carregar_idpt(treino: str | Path, teste: str | Path) -> CorpusIDPT:
    treino_df = ler_csv_robusto(treino)
    teste_df = ler_csv_robusto(teste)
    col_treino = achar_coluna(treino_df, COLUNAS_TEXTO, "texto")
    col_teste = achar_coluna(teste_df, COLUNAS_TEXTO, "texto")
    return CorpusIDPT(
        treino=preparar_dataframe(treino_df, texto_col=col_treino),
        teste=preparar_dataframe(teste_df, texto_col=col_teste),
        coluna_texto_treino=col_treino,
        coluna_texto_teste=col_teste,
    )
