import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adtc.v8.avaliacao import calcular_metricas


def test_precision_e_hit():
    df = pd.DataFrame(
        {
            "manual_top1_pertinente": [1, 0],
            "manual_top2_pertinente": [0, 1],
            "manual_top3_pertinente": [0, 0],
            "manual_top4_pertinente": [0, 0],
            "manual_top5_pertinente": [0, 0],
        }
    )
    m = calcular_metricas(df).set_index("metrica")
    assert m.loc["Precision@1", "valor"] == 0.5
    assert m.loc["Hit@1", "valor"] == 0.5
    assert m.loc["Precision@2", "valor"] == 0.5
    assert m.loc["Hit@2", "valor"] == 1.0
    assert m.loc["Precision@5", "valor"] == 0.2
    assert m.loc["Hit@5", "valor"] == 1.0
