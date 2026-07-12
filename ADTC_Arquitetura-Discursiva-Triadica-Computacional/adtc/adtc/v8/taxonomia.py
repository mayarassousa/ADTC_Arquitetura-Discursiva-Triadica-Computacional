"""Integração da taxonomia ADTC aos marcadores computacionais."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .arquivos import ler_csv_robusto

COLUNAS_ESPERADAS = (
    "tabela", "categoria_num", "categoria", "categoria_descricao",
    "n_recurso", "recurso", "definicao_fundamentacao", "exemplos_pb",
    "features_adt", "detectabilidade", "observacoes_anotacao",
)

MAPA_MARCADOR_TAXONOMIA: dict[str, list[int]] = {
    "aspas_ironicas": [5],
    "adversativa": [27],
    "negacao_polemica": [22],
    "discurso_relatado": [35],
    "avaliativo_invertido": [37],
    "aumentativo_ironico": [39],
    "diminutivo_ironico": [38],
    "antifrase_hiperbole": [1, 2],
    "maiusculas_enf": [7],
    "contraste_polaridade": [55],
    "onomatopeia": [15],
    "interrogacao_retorica": [24],
    "pontuacao_expressiva": [6],
    "exclamacao_inversao": [25],
    "formulaicas_modal": [11, 45],
    "alongamento_vocalico": [16],
    "mencao_contexto": [64, 65, 66],
    "pretericao": [34],
    "reduplicacao": [36],
}


def carregar_taxonomia(caminho: str | Path) -> pd.DataFrame:
    df = ler_csv_robusto(caminho)
    df.columns = [str(c).strip() for c in df.columns]
    faltantes = [c for c in COLUNAS_ESPERADAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas ausentes na taxonomia: {faltantes}")
    out = df.copy()
    out["n_recurso"] = out["n_recurso"].astype(int)
    out["categoria_num"] = out["categoria_num"].astype(int)
    return out


def parse_marcadores(marcadores) -> list[str]:
    if pd.isna(marcadores):
        return []
    saida: list[str] = []
    for parte in str(marcadores).split(";"):
        codigo = parte.strip().split(":", 1)[0].strip()
        if codigo:
            saida.append(codigo)
    return saida


def ids_recursos(marcadores) -> list[int]:
    ids: list[int] = []
    for codigo in parse_marcadores(marcadores):
        ids.extend(MAPA_MARCADOR_TAXONOMIA.get(codigo, []))
    return sorted(set(ids))


def explicar_recursos(marcadores, taxonomia: pd.DataFrame) -> str:
    ids = ids_recursos(marcadores)
    if not ids:
        return "Sem recurso taxonômico mapeado a partir dos marcadores superficiais."
    sub = taxonomia[taxonomia["n_recurso"].isin(ids)].sort_values(["categoria_num", "n_recurso"])
    return " | ".join(
        f"{int(r.n_recurso)}. {r.recurso} [Cat. {int(r.categoria_num)}: {r.categoria}; detectabilidade: {r.detectabilidade}]"
        for r in sub.itertuples(index=False)
    )


def categorias_recursos(marcadores, taxonomia: pd.DataFrame) -> str:
    ids = ids_recursos(marcadores)
    if not ids:
        return ""
    sub = taxonomia[taxonomia["n_recurso"].isin(ids)][["categoria_num", "categoria"]].drop_duplicates()
    return " | ".join(
        f"Cat. {int(r.categoria_num)} — {r.categoria}" for r in sub.itertuples(index=False)
    )


def integrar_taxonomia(df: pd.DataFrame, taxonomia: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "marcadores" not in out.columns:
        out["marcadores"] = ""
    out["taxonomia_recurso_ids"] = out["marcadores"].map(lambda x: ";".join(map(str, ids_recursos(x))))
    out["taxonomia_recursos"] = out["marcadores"].map(lambda x: explicar_recursos(x, taxonomia))
    out["taxonomia_categorias"] = out["marcadores"].map(lambda x: categorias_recursos(x, taxonomia))
    return out


def resumos_taxonomia(taxonomia: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    categorias = (
        taxonomia.groupby(["categoria_num", "categoria"], dropna=False)
        .size().reset_index(name="n_recursos").sort_values("categoria_num")
    )
    detectabilidade = taxonomia["detectabilidade"].value_counts(dropna=False).reset_index()
    detectabilidade.columns = ["detectabilidade", "n_recursos"]
    mapa = pd.DataFrame(
        [
            {"marcador_adtc": marcador, "n_recurso": rid}
            for marcador, ids in MAPA_MARCADOR_TAXONOMIA.items()
            for rid in ids
        ]
    ).merge(
        taxonomia[["n_recurso", "recurso", "categoria_num", "categoria", "features_adt", "detectabilidade"]],
        on="n_recurso", how="left",
    )
    return categorias, detectabilidade, mapa
