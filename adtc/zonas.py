"""Zonas interpretativas, contexto necessário e prioridade de auditoria."""

from __future__ import annotations

import pandas as pd


def atribuir_zona(pred: int, n_marcadores: int, limiar_alta: int = 3) -> str:
    pred = int(pred)
    n = int(n_marcadores or 0)
    if pred == 1 and n >= limiar_alta:
        return "Ironia predita com alta/média marcação superficial"
    if pred == 1:
        return "Ironia predita com baixa marcação: exige contexto"
    if pred == 0 and n >= 1:
        return "Não ironia predita com marcadores: possível falso gatilho"
    return "Não ironia predita sem marcadores relevantes"


def adicionar_zonas(df: pd.DataFrame, limiar_alta: int = 3) -> pd.DataFrame:
    out = df.copy()
    if "n_marc_sup" not in out.columns:
        out["n_marc_sup"] = 0
    out["zona_adtc"] = [atribuir_zona(p, n, limiar_alta) for p, n in zip(out["bert_pred"], out["n_marc_sup"])]
    return out


def classificar_contexto_necessario(row: pd.Series) -> str:
    zona = str(row.get("zona_adtc", "")).lower()
    marc = str(row.get("marcadores", "")).lower()
    texto = str(row.get("text", "")).lower()
    tipos: list[str] = []
    if "baixa marcação" in zona or "exige contexto" in zona:
        tipos.append("memoria_discursiva_evento_alvo")
    if "falso gatilho" in zona:
        tipos.append("diferenciar_recurso_jornalistico_de_ironia")
    if any(x in marc for x in ("aspas", "discurso_relatado")):
        tipos.append("fonte_enunciativa_citacao")
    if any(x in marc for x in ("negacao", "adversativa", "contraste")):
        tipos.append("contraste_argumentativo")
    if any(x in marc for x in ("avaliativo", "hiperbole", "aumentativo", "diminutivo")):
        tipos.append("avaliacao_lexical_intensificacao")
    if any(x in texto for x in ("governo", "presidente", "ministro", "congresso", "stf", "deputado", "senador", "prefeito")):
        tipos.append("contexto_politico_institucional")
    if any(x in texto for x in ("covid", "pandemia", "vacina", "cloroquina", "gripezinha")):
        tipos.append("memoria_sanitaria_politica")
    if any(x in texto for x in ("copa", "futebol", "seleção", "gol", "7 a 1", "neymar")):
        tipos.append("memoria_esportiva")
    if not tipos:
        tipos.append("contexto_tematico_geral")
    return "; ".join(dict.fromkeys(tipos))


def prioridade_rag(row: pd.Series, modo_post_hoc: bool = False) -> str:
    zona = str(row.get("zona_adtc", "")).lower()
    if "baixa marcação" in zona or "exige contexto" in zona or "falso gatilho" in zona:
        return "alta"
    if modo_post_hoc and str(row.get("tipo_erro", "")).upper() in {"FP", "FN"}:
        return "alta"
    if "ironia predita" in zona:
        return "media"
    return "baixa"


def adicionar_contexto_e_prioridade(df: pd.DataFrame, modo_post_hoc: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["tipo_contexto_necessario"] = out.apply(classificar_contexto_necessario, axis=1)
    out["prioridade_rag_integrado"] = out.apply(lambda r: prioridade_rag(r, modo_post_hoc), axis=1)
    return out
