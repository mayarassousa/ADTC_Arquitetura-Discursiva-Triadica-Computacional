"""Leitura robusta e localização determinística de artefatos."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd


def localizar_arquivo(bases: Iterable[Path], padroes: str | Iterable[str], obrigatorio: bool = True) -> Path | None:
    if isinstance(padroes, str):
        padroes = [padroes]
    achados: list[Path] = []
    for base in bases:
        if not base.exists():
            continue
        for padrao in padroes:
            achados.extend(p for p in base.rglob(padrao) if p.is_file())
    achados = sorted(set(achados), key=lambda p: (len(str(p)), str(p)))
    if achados:
        return achados[0]
    if obrigatorio:
        raise FileNotFoundError(f"Nenhum arquivo encontrado para os padrões: {list(padroes)}")
    return None


def ler_csv_robusto(caminho: str | Path) -> pd.DataFrame:
    caminho = Path(caminho)
    primeira = caminho.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if "\t" in primeira:
        sep = "\t"
    elif primeira.count(";") > primeira.count(","):
        sep = ";"
    else:
        sep = ","
    try:
        return pd.read_csv(caminho, sep=sep, engine="python", encoding="utf-8")
    except pd.errors.ParserError:
        return pd.read_csv(
            caminho,
            sep=sep,
            engine="python",
            encoding="utf-8",
            quotechar='"',
            escapechar="\\",
            on_bad_lines="warn",
        )


def salvar_csv(df: pd.DataFrame, caminho: str | Path) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False)
    return caminho
