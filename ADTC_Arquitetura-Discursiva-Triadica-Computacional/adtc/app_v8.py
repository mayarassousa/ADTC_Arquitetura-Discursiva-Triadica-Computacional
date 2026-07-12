"""Explorador Gradio da planilha final V8.

A demo não produz novas predições. Ela permite examinar a trilha auditável dos
itens já processados pelo notebook/pipeline, respeitando o regime de predições
pré-computadas da V8.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import pandas as pd

PLANILHA = Path(os.getenv("ADTC_V8_PLANILHA", "outputs/v8/planilha_final_tese_v8_rag_integrado_adtc_xai.xlsx"))


def carregar():
    if not PLANILHA.exists():
        return pd.DataFrame()
    return pd.read_excel(PLANILHA, sheet_name="Dataset_ADTC_RAG_XAI")


_DF = carregar()
_IDS = _DF["id_texto_original"].astype(str).tolist() if not _DF.empty else []


def exibir(id_texto):
    if _DF.empty:
        return "Planilha não encontrada. Defina ADTC_V8_PLANILHA.", "", "", ""
    row = _DF[_DF["id_texto_original"].astype(str).eq(str(id_texto))]
    if row.empty:
        return "ID não encontrado.", "", "", ""
    r = row.iloc[0]
    bert = f"Predição: {r.get('bert_pred')}\nConfiança: {r.get('bert_conf')}\nTipo de erro: {r.get('tipo_erro')}"
    adtc = f"Zona: {r.get('zona_adtc')}\nMarcadores: {r.get('marcadores')}\nTaxonomia: {r.get('taxonomia_recursos')}\nContexto necessário: {r.get('tipo_contexto_necessario')}"
    rag = f"Top-1: {r.get('fragmento_top1_rag_integrado')}\nOrigem: {r.get('origem_contexto_top1_rag_integrado')}\nScore integrado: {r.get('score_integrado_top1_rag_integrado')}"
    manual = f"Melhor top-k: {r.get('manual_melhor_topk')}\nDecisão: {r.get('manual_decisao_sobre_contexto')}\nParecer: {r.get('manual_parecer_contextual')}"
    return str(r.get("text", "")), bert, adtc, rag + "\n\n" + manual


demo = gr.Interface(
    fn=exibir,
    inputs=gr.Dropdown(choices=_IDS, label="ID do texto"),
    outputs=[
        gr.Textbox(label="Texto", lines=8),
        gr.Textbox(label="BERTimbau", lines=4),
        gr.Textbox(label="ADTC e taxonomia", lines=7),
        gr.Textbox(label="RAG e avaliação humana", lines=10),
    ],
    title="ADTC V8 — Explorador da trilha auditável",
    description="BERTimbau pré-computado + zonas ADTC + taxonomia + RAG integrado + avaliação humana.",
)

if __name__ == "__main__":
    demo.launch()
