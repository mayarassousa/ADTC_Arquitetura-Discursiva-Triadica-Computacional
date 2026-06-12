"""Extração dos sinais discursivos a partir das respostas livres do Tucano.

Converte as quatro respostas do modelo no dicionário de sinais lido pela
regra de decisão:

  contraste_textual                  SIM | NAO | AMBIGUO | INSUFICIENTE
  contexto_recuperado_relevante      SIM | NAO | INSUFICIENTE
  contexto_sustenta_ironia           SIM | NAO | INSUFICIENTE
  contexto_sustenta_indecidibilidade SIM | NAO
  posicao_sujeito                    CRITICA | DEBOCHE | ADESAO_LITERAL |
                                     DISTANCIAMENTO | INDETERMINADA
  recurso_taxonomico                 categoria da taxonomia (Miguel, 2026)

Valor de falha: ERRO_PARSE.

NOTA DE REPRODUCAO: esta e uma implementacao de referencia. Para
reproduzir exatamente os resultados da tese (95,1%; 39/41 casos),
substitua as funcoes deste modulo pelas do notebook Kaggle v20f.
"""

import re
import unicodedata

ERRO = "ERRO_PARSE"


def _norm(texto):
    t = unicodedata.normalize("NFD", texto or "")
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", t).upper().strip()


def parse_contraste(resposta_textual):
    t = _norm(resposta_textual)
    if re.search(r"\bAMBIGUO\b", t):
        return "AMBIGUO"
    if re.search(r"\bNAO\b", t[:60]):
        return "NAO"
    if re.search(r"\bSIM\b", t[:60]):
        return "SIM"
    if "INSUFICIENTE" in t:
        return "INSUFICIENTE"
    return ERRO


def parse_contexto(resposta_contextual):
    t = _norm(resposta_contextual)
    relevante = "NAO" if re.search(r"\b(NENHUM|NAO HA|IRRELEVANTE)\b", t) else "SIM"
    sustenta_ironia = "SIM" if re.search(r"\bIRONI", t) else "NAO"
    sustenta_indec = "SIM" if re.search(r"INDECID|INSUFICIENTE|NAO E POSSIVEL", t) else "NAO"
    if not t:
        return ERRO, ERRO, ERRO
    return relevante, sustenta_ironia, sustenta_indec


def parse_sujeito(resposta_sujeito):
    t = _norm(resposta_sujeito)
    for rotulo, padrao in [
        ("DEBOCHE", r"\bDEBOCHE\b"),
        ("DISTANCIAMENTO", r"\bDISTANCIAMENTO\b"),
        ("CRITICA", r"\bCRITIC"),
        ("ADESAO_LITERAL", r"\bADESAO\b|\bLITERAL\b"),
        ("INDETERMINADA", r"\bINDETERMINAD"),
    ]:
        if re.search(padrao, t):
            return rotulo
    return ERRO


def parse_recurso(resposta_decisao):
    t = _norm(resposta_decisao)
    for rotulo, padrao in [
        ("MEMORIA_DISCURSIVA", r"MEMORIA"),
        ("ASPAS_IRONICAS", r"ASPAS"),
        ("ANTIFRASE", r"ANTIFRASE"),
        ("CONTRASTE_POLARIDADE", r"CONTRASTE"),
        ("POSICAO_CRITICA", r"POSICAO CRITIC"),
        ("MARCADOR_LEXICAL", r"MARCADOR"),
        ("AMBIGUIDADE_CONTEXTUAL", r"AMBIGU"),
    ]:
        if re.search(padrao, t):
            return rotulo
    return ERRO


def extrair_sinais(respostas):
    """respostas: dict com as chaves textual, contextual, sujeito, decisao."""
    ctx_rel, ctx_ironia, ctx_indec = parse_contexto(respostas.get("contextual", ""))
    return {
        "contraste_textual": parse_contraste(respostas.get("textual", "")),
        "contexto_recuperado_relevante": ctx_rel,
        "contexto_sustenta_ironia": ctx_ironia,
        "contexto_sustenta_indecidibilidade": ctx_indec,
        "posicao_sujeito": parse_sujeito(respostas.get("sujeito", "")),
        "recurso_taxonomico": parse_recurso(respostas.get("decisao", "")),
    }
