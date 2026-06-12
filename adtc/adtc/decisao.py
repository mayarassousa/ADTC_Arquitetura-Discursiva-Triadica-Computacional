"""Cascata de decisao triadica v20f, transcrita da tese (Trechos 2 e 5).

A ordem das clausulas e discursivamente motivada. Vem primeiro as duas
travas de literalidade; depois a indecidibilidade, para enunciados curtos
demais; em seguida a ambiguidade, para enunciados lexicalmente abertos;
entao a memoria discursiva cultural, que reconhece como ironicos os
enunciados que retomam criticamente formulacoes em circulacao na memoria
coletiva brasileira (mencao ecoica, Sperber e Wilson, 1986); e, por fim,
a decisao por pontuacao.

NOTA DE REPRODUCAO: os blocos marcados com "COMPLETAR (notebook v20f)"
correspondem a linhas de pontuacao omitidas na transcricao da tese
("calculo de escores omitido"). Cole-as do notebook para reproducao
exata dos resultados.
"""

from adtc.detectores import motivo_literalidade_discursiva
from adtc.detectores import (
    texto_eh_ambiguo_curto,
    texto_eh_indecidivel_curto,
    texto_tem_contraste_forte,
    texto_tem_memoria_critica,
)
from adtc.travas import texto_eh_literal_discursivo, texto_eh_literal_factual


def regra_adtc_v20f(sinais, texto, tem_doc_inventado=False):
    contraste  = sinais.get("contraste_textual", "ERRO_PARSE")
    ctx_rel    = sinais.get("contexto_recuperado_relevante", "ERRO_PARSE")
    ctx_ironia = sinais.get("contexto_sustenta_ironia", "ERRO_PARSE")
    ctx_indec  = sinais.get("contexto_sustenta_indecidibilidade", "ERRO_PARSE")
    sujeito    = sinais.get("posicao_sujeito", "ERRO_PARSE")
    recurso    = sinais.get("recurso_taxonomico", "ERRO_PARSE")

    sujeito_ironico = sujeito in ["CRITICA", "DEBOCHE", "DISTANCIAMENTO"]
    recurso_ironico = recurso in ["CONTRASTE_POLARIDADE", "ANTIFRASE", "ASPAS_IRONICAS", "MEMORIA_DISCURSIVA", "POSICAO_CRITICA", "MARCADOR_LEXICAL"]
    recurso_memoria = recurso == "MEMORIA_DISCURSIVA"
    recurso_amb = recurso == "AMBIGUIDADE_CONTEXTUAL"

    score_ironia = 0.0
    score_indec = 0.0
    score_literal = 0.0
    motivos = []

    if contraste == "SIM": score_ironia += 2.0; motivos.append("contraste textual")
    elif contraste == "NAO": score_literal += 1.0; motivos.append("sem contraste textual")
    elif contraste == "INSUFICIENTE": score_indec += 0.8; motivos.append("contraste insuficiente")
    else: score_indec += 0.3; motivos.append("contraste com ERRO_PARSE")

    if ctx_ironia == "SIM": score_ironia += 2.0; motivos.append("contexto sustenta ironia")
    elif ctx_ironia == "NAO": score_literal += 1.3; motivos.append("contexto não sustenta ironia")
    elif ctx_ironia == "INSUFICIENTE": score_indec += 0.7; motivos.append("contexto insuficiente para ironia")
    else: score_indec += 0.3; motivos.append("contexto ironia com ERRO_PARSE")

    # ------------------------------------------------------------------
    # COMPLETAR (notebook v20f): linhas de pontuacao de ctx_indec,
    # posicao do sujeito e recurso taxonomico, omitidas na transcricao
    # da tese. Cole-as aqui, sem alteracao, a partir do notebook.
    # ------------------------------------------------------------------

    revisao = False

    # 0. Trava prioritária de literalidade factual: vem antes de qualquer score.
    if texto_eh_literal_factual(texto):
        label = "NAO_IRONICO"
        motivo = "trava de literalidade factual: evento factual sem marcador avaliativo irônico"
        fonte_decisao = "literalidade_factual_prioritaria"
        forca = "ALTA" if contraste == "NAO" and sujeito == "ADESAO_LITERAL" else "MEDIA"
        risco = "BAIXO"
        revisao = False

    # 0b. Trava prioritária de literalidade discursiva/fraseológica.
    elif texto_eh_literal_discursivo(texto):
        label = "NAO_IRONICO"
        motivo = motivo_literalidade_discursiva(texto)
        fonte_decisao = "literalidade_discursiva_prioritaria"
        forca = "ALTA" if contraste == "NAO" and sujeito == "ADESAO_LITERAL" else "MEDIA"
        risco = "BAIXO"
        revisao = False

    # 1. Indecidibilidade curta: falta contexto para estabilizar a leitura.
    elif texto_eh_indecidivel_curto(texto):
        label = "INDECIDIVEL"
        motivo = "enunciado curto demais; faltam condições contextuais para decidir"
        fonte_decisao = "indecidibilidade_contextual_curta"
        forca = "ALTA" if ctx_indec == "SIM" else "MEDIA"
        risco = "MEDIO"
        revisao = True

    # 2. Ambiguidade textual própria: dupla leitura plausível no enunciado.
    elif texto_eh_ambiguo_curto(texto) and not texto_tem_contraste_forte(texto):
        label = "AMBIGUO"
        motivo = "enunciado lexicalmente aberto, com dupla leitura possível"
        fonte_decisao = "ambiguidade_textual_curta"
        forca = "ALTA" if recurso_amb or ctx_indec == "SIM" else "MEDIA"
        risco = "MEDIO"
        revisao = True

    # 3. Memória discursiva cultural crítica pode sustentar ironia.
    elif texto_tem_memoria_critica(texto):
        label = "IRONICO"
        motivo = "memória discursiva cultural com retomada crítica"
        fonte_decisao = "memoria_discursiva_cultural"
        forca = "ALTA" if recurso_memoria or ctx_ironia == "SIM" else "MEDIA"
        risco = "MEDIO" if ctx_ironia == "SIM" else "ALTO"
        revisao = risco == "ALTO"

    # 4. Contraste textual/contextual sustentado.
    elif score_ironia >= max(score_literal + 0.8, score_indec + 0.8) and score_ironia >= 2.5:
        label = "IRONICO"
        motivo = "soma de sinais favorece ironia"
        fonte_decisao = "decisao_triadica_combinada"
        forca = "ALTA" if score_ironia >= 4.5 else "MEDIA"
        risco = "BAIXO" if forca == "ALTA" else "MEDIO"

    # 5. Literalidade não prioritária, mas ainda predominante.
    elif score_literal >= max(score_ironia, score_indec) and score_literal >= 1.8:
        label = "NAO_IRONICO"
        motivo = "sinais de literalidade predominam"
        fonte_decisao = "literalidade_por_score"
        forca = "MEDIA"
        risco = "MEDIO"

    # 6. Ambiguidade/indecidibilidade por sinais combinados.
    elif recurso_amb and ctx_indec == "SIM":
        label = "AMBIGUO"
        motivo = "recurso de ambiguidade contextual + contexto indica dupla leitura"
        fonte_decisao = "ambiguidade_contextual"
        forca = "MEDIA"
        risco = "MEDIO"
        revisao = True

    else:
        label = "INDECIDIVEL"
        motivo = "sinais insuficientes ou conflitantes para decisão segura"
        fonte_decisao = "indecidibilidade_por_conflito"
        forca = "BAIXA"
        risco = "ALTO"
        revisao = True

    return {
        "label": label,
        "motivo": motivo,
        "motivos": motivos,
        "fonte_decisao": fonte_decisao,
        "forca_evidencia": forca,
        "risco_sobreajuste": risco,
        "revisao_humana": revisao,
        "scores": {
            "ironia": score_ironia,
            "literal": score_literal,
            "indecidibilidade": score_indec,
        },
        "sinais": sinais,
    }
