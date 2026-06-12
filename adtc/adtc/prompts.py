"""Perguntas dos quatro eixos da ADTC, transcritas sem alteração da tese.

É nelas que a teoria se traduz em interrogação concreta. O gabarito é
cego: não é apresentado ao modelo em nenhum momento.
"""

PERGUNTA_EIXO_TEXTUAL = (
    "Há marcas textuais de ironia no enunciado? Responda SIM, NÃO ou "
    "AMBÍGUO e cite apenas marcas presentes no texto."
)

PERGUNTA_EIXO_CONTEXTUAL = (
    "Usando apenas os trechos RAG recuperados, que contexto externo é "
    "relevante para interpretar o enunciado? Cite o documento ou trecho "
    "que sustenta sua resposta."
)

PERGUNTA_EIXO_SUJEITO = (
    "Que posição do sujeito enunciador aparece: crítica, deboche, adesão "
    "literal, distanciamento ou indeterminada? Justifique com o texto e, "
    "se houver, com os trechos RAG."
)

PERGUNTA_DECISAO_TRIADICA = (
    "Com base no texto, nos trechos RAG recuperados e na posição do "
    "sujeito, a leitura é IRONICO, NAO_IRONICO, AMBIGUO ou INDECIDIVEL? "
    "Justifique sem inventar contexto."
)

EIXOS = {
    "textual": PERGUNTA_EIXO_TEXTUAL,
    "contextual": PERGUNTA_EIXO_CONTEXTUAL,
    "sujeito": PERGUNTA_EIXO_SUJEITO,
    "decisao": PERGUNTA_DECISAO_TRIADICA,
}


def montar_prompt(eixo, enunciado, trechos_rag=None):
    """Monta o prompt de um eixo para um enunciado.

    AJUSTE: confira o formato exato (cabecalhos, ordem dos blocos) com o
    usado no notebook v20f para reproducao fiel dos resultados da tese.
    """
    if eixo not in EIXOS:
        raise ValueError(f"Eixo desconhecido: {eixo}")
    blocos = [f"Enunciado: {enunciado}"]
    if trechos_rag:
        contexto = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(trechos_rag))
        blocos.append(f"Trechos RAG recuperados:\n{contexto}")
    blocos.append(f"Pergunta: {EIXOS[eixo]}")
    return "\n\n".join(blocos)
