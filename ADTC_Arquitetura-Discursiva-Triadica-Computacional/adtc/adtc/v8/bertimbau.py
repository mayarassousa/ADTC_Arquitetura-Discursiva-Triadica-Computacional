"""Integração das predições pré-computadas do BERTimbau e métricas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from .arquivos import ler_csv_robusto
from .integridade import hash_texto


def tipo_erro(label: int, pred: int) -> str:
    return {(0, 0): "VN", (0, 1): "FP", (1, 0): "FN", (1, 1): "VP"}.get((int(label), int(pred)), "NA")


def garantir_tipo_erro(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"label", "bert_pred"}.issubset(out.columns):
        out["tipo_erro"] = [tipo_erro(y, p) for y, p in zip(out["label"], out["bert_pred"])]
        out["acertou_bert"] = out["label"].astype(int).eq(out["bert_pred"].astype(int))
    return out


def carregar_predicoes(caminho: str | Path, teste: pd.DataFrame) -> pd.DataFrame:
    pred = ler_csv_robusto(caminho)
    if "id_texto_original" not in pred.columns:
        pred["id_texto_original"] = np.arange(len(pred))
    if "label" not in pred.columns:
        if len(pred) != len(teste):
            raise ValueError("Predições sem label e com tamanho diferente do teste.")
        pred["label"] = teste["label"].to_numpy()
    if "bert_pred" not in pred.columns:
        for col in ("pred", "prediction_modelo", "predicao", "predicted_label", "y_pred"):
            if col in pred.columns:
                pred["bert_pred"] = pred[col]
                break
        else:
            raise ValueError(f"Arquivo sem coluna de predição. Colunas: {pred.columns.tolist()}")
    if "text_hash" not in pred.columns:
        if "text" in pred.columns:
            pred["text_hash"] = pred["text"].map(hash_texto)
        elif len(pred) == len(teste):
            pred["text_hash"] = teste["text_hash"].to_numpy()
    pred["bert_conf"] = pred.get("bert_conf", np.nan)
    pred["bert_p_iro"] = pred.get("bert_p_iro", np.nan)
    pred["label"] = pred["label"].astype(int)
    pred["bert_pred"] = pred["bert_pred"].astype(int)
    return garantir_tipo_erro(pred)


def integrar_predicoes(teste: pd.DataFrame, predicoes: pd.DataFrame) -> pd.DataFrame:
    out = teste.drop(columns=[c for c in ("bert_pred", "bert_conf", "bert_p_iro", "tipo_erro") if c in teste], errors="ignore").copy()
    cols = [c for c in ("id_texto_original", "bert_pred", "bert_conf", "bert_p_iro", "tipo_erro") if c in predicoes]
    out = out.merge(predicoes[cols].drop_duplicates("id_texto_original"), on="id_texto_original", how="left")
    if out["bert_pred"].isna().any():
        if len(out) != len(predicoes):
            raise ValueError("Falha no alinhamento das predições por ID e tamanhos incompatíveis.")
        out[[c for c in cols if c != "id_texto_original"]] = predicoes[[c for c in cols if c != "id_texto_original"]].to_numpy()
    out["bert_pred"] = out["bert_pred"].astype(int)
    return garantir_tipo_erro(out)


def calcular_metricas(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dados = df.dropna(subset=["label", "bert_pred"]).copy()
    y = dados["label"].astype(int)
    p = dados["bert_pred"].astype(int)
    cm = confusion_matrix(y, p, labels=[0, 1])
    metricas = pd.DataFrame(
        {
            "metrica": ["accuracy", "balanced_accuracy", "macro_f1"],
            "valor": [accuracy_score(y, p), balanced_accuracy_score(y, p), f1_score(y, p, average="macro")],
        }
    )
    matriz = pd.DataFrame(cm, index=["real_0", "real_1"], columns=["pred_0", "pred_1"]).reset_index(names="rotulo_ouro")
    relatorio = classification_report(y, p, output_dict=True, zero_division=0)
    return metricas, matriz, relatorio
