import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adtc.v8.config import ConfigV8
from adtc.v8.rag import RecuperadorHibrido, aplicar_rag, construir_base_integrada
from adtc.v8.zonas import adicionar_contexto_e_prioridade, adicionar_zonas


def test_fluxo_rag_lexical():
    treino = pd.DataFrame({"text": ["A Polícia Federal realizou a Operação Ossobuco."]})
    curada = pd.DataFrame(
        {
            "id": ["c1"],
            "tema": ["jornalismo aspas"],
            "fragmento": ["Aspas jornalísticas podem indicar citação, não ironia."],
            "fonte_sugerida": ["base teste"],
        }
    )
    teste = pd.DataFrame(
        {
            "id_texto_original": [1],
            "text": ["A Polícia Federal deflagrou a Operação Ossobuco."],
            "label": [0],
            "bert_pred": [1],
            "tipo_erro": ["FP"],
            "n_marc_sup": [1],
            "marcadores": ["aspas_ironicas:Cat1"],
            "taxonomia_recursos": ["Aspas irônicas"],
            "taxonomia_categorias": ["Cat. 1"],
        }
    )
    teste = adicionar_zonas(teste)
    teste = adicionar_contexto_e_prioridade(teste, modo_post_hoc=True)
    base = construir_base_integrada(curada, treino, incluir_seeds=False)
    cfg = ConfigV8(usar_neural=False, usar_reranker=False, candidatos_por_canal=5, top_k_bruto=3, top_k_final=2)
    motor = RecuperadorHibrido(base, cfg)
    out = aplicar_rag(teste, motor, cfg)
    assert len(out) == 1
    assert out.loc[0, "fragmento_top1_rag_integrado"]
    assert out.loc[0, "score_integrado_top1_rag_integrado"] >= 0
