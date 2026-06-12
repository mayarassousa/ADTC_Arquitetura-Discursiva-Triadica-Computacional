"""Teste de fumaca da infraestrutura (nao reproduz resultados da tese).

Verifica apenas que o corpus carrega, o RAG indexa/recupera e a cascata
de decisao executa de ponta a ponta. Os detectores sao substituidos por
funcoes neutras (sempre False) exclusivamente para exercitar o fluxo de
codigo; isso NAO equivale aos detectores do notebook v20f e NAO produz
os resultados reportados na tese.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adtc import detectores
from adtc.corpus import carregar_corpus
from adtc.rag import RecuperadorContextual
from adtc.sinais import extrair_sinais


def _neutralizar_detectores():
    nomes = [
        "texto_tem_contraste_forte", "texto_tem_memoria_critica",
        "texto_eh_ambiguo_curto", "texto_eh_indecidivel_curto",
        "texto_tem_emoticon_contraste", "texto_eh_noticia_ou_explicativo_factual",
        "texto_eh_fraseologia_literal", "texto_eh_afetivo_ou_reclamacao_literal_curta",
    ]
    for n in nomes:
        setattr(detectores, n, lambda texto: False)
    detectores.motivo_literalidade_discursiva = lambda texto: "stub"


def test_corpus():
    casos = carregar_corpus()
    assert len(casos) == 41
    assert sum(c["rotulo"] == "IRONICO" for c in casos) == 23
    assert sum(c["rotulo"] == "NAO_IRONICO" for c in casos) == 14
    assert sum(c["rotulo"] == "AMBIGUO" for c in casos) == 2
    assert sum(c["rotulo"] == "INDECIDIVEL" for c in casos) == 2


def test_rag():
    docs = [
        {"id": "d1", "titulo": "Copa 2014", "texto": "Derrota por 7 a 1 para a Alemanha na Copa do Mundo de 2014."},
        {"id": "d2", "titulo": "Distrator", "texto": "A Quinta Sinfonia de Beethoven estreou em Viena em 1808."},
    ]
    r = RecuperadorContextual(docs, k=1)
    top = r.recuperar("Meu projeto virou meu 7 a 1 particular.")
    assert top[0]["id"] == "d1"


def test_cascata():
    _neutralizar_detectores()
    # reimporta a cascata apos a neutralizacao
    import importlib
    from adtc import travas, decisao
    importlib.reload(travas)
    importlib.reload(decisao)

    sinais = extrair_sinais({
        "textual": "SIM. Ha contraste entre 'que otimo' e o evento negativo.",
        "contextual": "O documento sobre a pandemia sustenta leitura ironica.",
        "sujeito": "Posicao critica do enunciador.",
        "decisao": "IRONICO, por contraste de polaridade.",
    })
    reg = decisao.regra_adtc_v20f(sinais, "Que otimo, mais um atraso.")
    assert reg["label"] in {"IRONICO", "NAO_IRONICO", "AMBIGUO", "INDECIDIVEL"}
    assert reg["fonte_decisao"]
    assert "scores" in reg


if __name__ == "__main__":
    test_corpus(); test_rag(); test_cascata()
    print("Smoke test OK: corpus (41 casos), RAG e cascata executam.")
