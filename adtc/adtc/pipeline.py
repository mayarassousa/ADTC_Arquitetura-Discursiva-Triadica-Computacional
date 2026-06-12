"""Pipeline completo: enunciado -> 4 eixos -> sinais -> decisao auditada.

Cada enunciado e interrogado quatro vezes, uma por eixo (164 itens para
os 41 casos). A saida de cada caso e um registro de auditoria com fonte
da decisao, forca da evidencia, risco de sobreajuste e recomendacao de
revisao humana.
"""

from adtc.decisao import regra_adtc_v20f
from adtc.prompts import montar_prompt
from adtc.sinais import extrair_sinais


def analisar_enunciado(enunciado, tucano, recuperador, k_rag=3):
    """Roda os quatro eixos sobre um enunciado e devolve o registro auditado."""
    docs = recuperador.recuperar(enunciado, k=k_rag)
    trechos = [f"{d['titulo']}: {d['texto']}" for d in docs]

    respostas = {}
    for eixo in ("textual", "contextual", "sujeito", "decisao"):
        usa_rag = eixo in ("contextual", "sujeito", "decisao")
        prompt = montar_prompt(eixo, enunciado, trechos if usa_rag else None)
        respostas[eixo] = tucano.responder(prompt)

    sinais = extrair_sinais(respostas)
    registro = regra_adtc_v20f(sinais, enunciado)
    registro["enunciado"] = enunciado
    registro["respostas_tucano"] = respostas
    registro["rag"] = docs
    return registro
