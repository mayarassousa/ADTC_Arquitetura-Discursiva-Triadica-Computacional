"""Amostragem e métricas da avaliação manual da pertinência contextual."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def normalizar_binario(valor) -> float:
    if pd.isna(valor):
        return np.nan
    if isinstance(valor, (bool, int, float, np.integer, np.floating)):
        if float(valor) in (0.0, 1.0):
            return float(valor)
    texto = str(valor).strip().lower()
    mapa = {"1": 1.0, "sim": 1.0, "s": 1.0, "true": 1.0, "0": 0.0, "não": 0.0, "nao": 0.0, "n": 0.0, "false": 0.0}
    return mapa.get(texto, np.nan)


def selecionar_amostra(df: pd.DataFrame, n_alta: int = 70, n_media: int = 20, seed: int = 42) -> pd.DataFrame:
    """Seleciona amostra estratificada por prioridade ADTC.

    A amostra privilegia os casos em que a auditoria é mais necessária. Por isso,
    as métricas resultantes descrevem o recuperador nessa região de risco e não
    constituem estimativa aleatória do teste inteiro.
    """
    rng = np.random.default_rng(seed)
    partes: list[pd.DataFrame] = []
    for prioridade, n in (("alta", n_alta), ("media", n_media)):
        sub = df[df["prioridade_rag_integrado"].astype(str).eq(prioridade)].copy()
        if len(sub) > n:
            sub = sub.iloc[rng.choice(len(sub), size=n, replace=False)]
        partes.append(sub)
    amostra = pd.concat(partes, ignore_index=True)
    return amostra.sample(frac=1, random_state=seed).reset_index(drop=True)


def preparar_planilha_manual(amostra: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Cria o formato largo usado para os julgamentos manuais."""
    colunas_base = [c for c in [
        "id_texto_original", "text", "label", "bert_pred", "bert_conf", "tipo_erro",
        "zona_adtc", "prioridade_rag_integrado", "marcadores", "taxonomia_recursos",
        "tipo_contexto_necessario", "consulta_integrada_adtc",
    ] if c in amostra.columns]
    out = amostra[colunas_base].copy()
    for k in range(1, top_k + 1):
        for origem, destino in [
            (f"fragmento_top{k}_rag_integrado", f"top{k}_fragmento"),
            (f"fonte_sugerida_top{k}_rag_integrado", f"top{k}_fonte"),
            (f"origem_contexto_top{k}_rag_integrado", f"top{k}_origem_contexto"),
            (f"score_semantico_top{k}_rag_integrado", f"top{k}_score_semantico"),
            (f"score_discursivo_top{k}_rag_integrado", f"top{k}_score_discursivo"),
            (f"score_integrado_top{k}_rag_integrado", f"top{k}_score_integrado"),
        ]:
            out[destino] = amostra[origem] if origem in amostra.columns else ""
        out[f"manual_top{k}_pertinente"] = ""
        out[f"manual_top{k}_ajuda_origem_evento"] = ""
        out[f"manual_top{k}_ajuda_tema"] = ""
        out[f"manual_top{k}_ajuda_ironia_ou_falso_gatilho"] = ""
        out[f"manual_top{k}_observacao"] = ""
    out["manual_melhor_topk"] = ""
    out["manual_parecer_contextual"] = ""
    out["manual_decisao_sobre_contexto"] = ""
    return out


def calcular_metricas(df_manual: pd.DataFrame, ks: tuple[int, ...] = (1, 2, 5)) -> pd.DataFrame:
    """Calcula Precision@k e Hit@k a partir de julgamentos binários.

    Precision@k é a proporção média de fragmentos pertinentes nas k primeiras
    posições. Hit@k vale 1 para um caso quando pelo menos um dos k primeiros
    fragmentos é pertinente; a média dos casos é a cobertura contextual em k.
    """
    resultados: list[dict] = []
    for k in ks:
        cols = [f"manual_top{i}_pertinente" for i in range(1, k + 1)]
        faltantes = [c for c in cols if c not in df_manual.columns]
        if faltantes:
            raise ValueError(f"Colunas manuais ausentes: {faltantes}")
        binarios = df_manual[cols].apply(lambda coluna: coluna.map(normalizar_binario))
        completos = binarios.dropna(how="all")
        if completos.empty:
            precision = hit = cobertura = np.nan
            n = 0
        else:
            precision = float(completos.mean(axis=1, skipna=True).mean())
            hit = float(completos.fillna(0).max(axis=1).mean())
            cobertura = float(completos.notna().any(axis=1).mean())
            n = len(completos)
        resultados.extend(
            [
                {"metrica": f"Precision@{k}", "valor": precision, "valor_percentual": precision * 100 if pd.notna(precision) else np.nan, "n_itens": n},
                {"metrica": f"Hit@{k}", "valor": hit, "valor_percentual": hit * 100 if pd.notna(hit) else np.nan, "n_itens": n},
            ]
        )
    resultados.append({"metrica": "Cobertura avaliada", "valor": cobertura, "valor_percentual": cobertura * 100 if pd.notna(cobertura) else np.nan, "n_itens": n})
    return pd.DataFrame(resultados)


def integrar_avaliacao(dataset: pd.DataFrame, manual: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    if "id_texto_original" not in manual.columns:
        raise ValueError("A avaliação manual precisa conter id_texto_original.")
    campos = ["id_texto_original"]
    for k in range(1, top_k + 1):
        campos += [c for c in (
            f"manual_top{k}_pertinente", f"manual_top{k}_ajuda_origem_evento",
            f"manual_top{k}_ajuda_tema", f"manual_top{k}_ajuda_ironia_ou_falso_gatilho",
            f"manual_top{k}_observacao",
        ) if c in manual.columns]
    campos += [c for c in ("manual_melhor_topk", "manual_parecer_contextual", "manual_decisao_sobre_contexto") if c in manual.columns]
    out = dataset.drop(columns=[c for c in campos if c != "id_texto_original" and c in dataset.columns], errors="ignore")
    out = out.merge(manual[campos].drop_duplicates("id_texto_original"), on="id_texto_original", how="left")
    out["avaliacao_manual_realizada"] = out["id_texto_original"].isin(set(manual["id_texto_original"])).astype(int)
    return out


def validar_consistencia(df_manual: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Lista incoerências entre melhor_topk e os julgamentos binários."""
    problemas: list[dict] = []
    for _, row in df_manual.iterrows():
        melhor = row.get("manual_melhor_topk", np.nan)
        try:
            melhor = int(float(melhor))
        except Exception:
            melhor = None
        positivos = [k for k in range(1, top_k + 1) if normalizar_binario(row.get(f"manual_top{k}_pertinente")) == 1.0]
        if melhor in range(1, top_k + 1) and melhor not in positivos:
            problemas.append({"id_texto_original": row.get("id_texto_original"), "problema": f"melhor_topk={melhor}, mas top{melhor} não está marcado como pertinente"})
        if melhor == 0 and positivos:
            problemas.append({"id_texto_original": row.get("id_texto_original"), "problema": f"melhor_topk=0, mas há posições pertinentes: {positivos}"})
    return pd.DataFrame(problemas)


def ler_avaliacao(caminho: str | Path) -> pd.DataFrame:
    caminho = Path(caminho)
    if caminho.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(caminho)
    return pd.read_csv(caminho)
