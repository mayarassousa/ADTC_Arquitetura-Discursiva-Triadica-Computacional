import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adtc.v8.integridade import aplicar_trava_antivazamento, hash_texto


def test_remove_overlap_exato():
    treino = pd.DataFrame({"text": ["texto A", "texto B"], "label": [0, 1]})
    teste = pd.DataFrame({"text": ["texto A"], "label": [0]})
    treino["text_hash"] = treino["text"].map(hash_texto)
    teste["text_hash"] = teste["text"].map(hash_texto)
    resultado = aplicar_trava_antivazamento(treino, teste)
    assert len(resultado.treino) == 1
    assert len(resultado.removidas) == 1
    assert int(resultado.diagnostico.loc[resultado.diagnostico["item"] == "overlap_hash_pos_trava", "valor"].iloc[0]) == 0
