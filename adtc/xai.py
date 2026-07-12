"""XAI neural opcional para o BERTimbau.

A V8 diferencia dois regimes:
- evidencial: o checkpoint carregado é o mesmo que produziu as predições;
- demonstração não evidencial: usa-se o BERTimbau base apenas para testar a
  infraestrutura de LIME, Integrated Gradients, SHAP e atenção.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ResultadoXAI:
    tabela: pd.DataFrame
    status: pd.DataFrame


def selecionar_amostra(df: pd.DataFrame, n_por_tipo: int = 2, seed: int = 42) -> pd.DataFrame:
    if "tipo_erro" not in df.columns:
        return df.sample(min(len(df), n_por_tipo * 4), random_state=seed)
    partes = []
    for tipo in ("VP", "VN", "FP", "FN"):
        sub = df[df["tipo_erro"].eq(tipo)]
        if not sub.empty:
            partes.append(sub.sample(min(n_por_tipo, len(sub)), random_state=seed))
    return pd.concat(partes, ignore_index=True) if partes else df.head(0)


def carregar_modelo(nome_ou_caminho: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(nome_ou_caminho)
    modelo = AutoModelForSequenceClassification.from_pretrained(nome_ou_caminho)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo.to(device).eval()
    return tokenizer, modelo, device


def preditor_proba(tokenizer, modelo, device: str, max_len: int = 256) -> Callable[[list[str]], np.ndarray]:
    import torch

    def predizer(textos: list[str]) -> np.ndarray:
        lote = tokenizer(textos, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        lote = {k: v.to(device) for k, v in lote.items()}
        with torch.no_grad():
            logits = modelo(**lote).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        return probs

    return predizer


def _top_tokens_lime(texto: str, predizer, top_n: int = 20, num_samples: int = 300) -> str:
    try:
        from lime.lime_text import LimeTextExplainer
        explainer = LimeTextExplainer(class_names=["não irônico", "irônico"])
        exp = explainer.explain_instance(texto, predizer, labels=[1], num_features=top_n, num_samples=num_samples)
        return " | ".join(f"{tok}:{peso:.4f}" for tok, peso in exp.as_list(label=1))
    except Exception as exc:
        return f"ERRO_LIME:{exc!r}"


def _top_tokens_ig(texto: str, tokenizer, modelo, device: str, max_len: int = 256, top_n: int = 20) -> str:
    try:
        import torch
        from captum.attr import LayerIntegratedGradients
        encoded = tokenizer(texto, truncation=True, max_length=max_len, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        embeddings = modelo.get_input_embeddings()

        def forward(ids, mask):
            return modelo(input_ids=ids, attention_mask=mask).logits[:, 1]

        lig = LayerIntegratedGradients(forward, embeddings)
        baseline = torch.full_like(input_ids, tokenizer.pad_token_id or 0)
        attrs = lig.attribute(input_ids, baselines=baseline, additional_forward_args=(attention_mask,), n_steps=16)
        scores = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu().tolist())
        pares = sorted(zip(tokens, scores), key=lambda x: abs(float(x[1])), reverse=True)[:top_n]
        return " | ".join(f"{t}:{float(s):.4f}" for t, s in pares)
    except Exception as exc:
        return f"ERRO_IG:{exc!r}"


def _top_tokens_attention(texto: str, tokenizer, modelo, device: str, max_len: int = 256, top_n: int = 20) -> str:
    try:
        import torch
        encoded = tokenizer(texto, truncation=True, max_length=max_len, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = modelo(**encoded, output_attentions=True)
        att = out.attentions[-1].mean(dim=1)[0, 0].detach().cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].detach().cpu().tolist())
        pares = sorted(zip(tokens, att), key=lambda x: float(x[1]), reverse=True)[:top_n]
        return " | ".join(f"{t}:{float(s):.4f}" for t, s in pares)
    except Exception as exc:
        return f"ERRO_ATTENTION:{exc!r}"


def executar_xai(
    df: pd.DataFrame,
    modelo_nome: str,
    modelo_evidencial: bool,
    n_por_tipo: int = 2,
    max_len: int = 256,
    seed: int = 42,
) -> ResultadoXAI:
    amostra = selecionar_amostra(df, n_por_tipo=n_por_tipo, seed=seed).copy()
    try:
        tokenizer, modelo, device = carregar_modelo(modelo_nome)
        predizer = preditor_proba(tokenizer, modelo, device, max_len=max_len)
    except Exception as exc:
        status = pd.DataFrame([{"modelo": modelo_nome, "status": "falha_carregamento", "detalhe": repr(exc), "xai_modo": "nao_executado"}])
        return ResultadoXAI(pd.DataFrame(), status)

    registros = []
    for _, row in amostra.iterrows():
        texto = str(row.get("text", ""))
        probs = predizer([texto])[0]
        registros.append(
            {
                "id_texto_original": row.get("id_texto_original"),
                "xai_model_pred": int(np.argmax(probs)),
                "xai_model_p_iro": float(probs[1]) if len(probs) > 1 else np.nan,
                "xai_model_conf": float(np.max(probs)),
                "lime_top_tokens": _top_tokens_lime(texto, predizer),
                "ig_top_tokens": _top_tokens_ig(texto, tokenizer, modelo, device, max_len=max_len),
                "attention_top_tokens": _top_tokens_attention(texto, tokenizer, modelo, device, max_len=max_len),
                "shap_top_tokens": "NAO_EXECUTADO_POR_PADRAO",
                "xai_modo": "evidencial_mesmo_checkpoint" if modelo_evidencial else "demonstracao_modelo_base_nao_evidencial",
            }
        )
    tabela = pd.DataFrame(registros)
    status = pd.DataFrame([{"modelo": modelo_nome, "status": "ok", "n_casos": len(tabela), "xai_modo": tabela["xai_modo"].iloc[0] if not tabela.empty else "nao_executado"}])
    return ResultadoXAI(tabela, status)
