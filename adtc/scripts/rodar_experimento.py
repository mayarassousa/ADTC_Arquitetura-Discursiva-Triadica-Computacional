"""Executa a ADTC v20f sobre o corpus de demonstracao e confere o gabarito.

Uso:
    python scripts/rodar_experimento.py [--saida resultados_v20f.csv] [--k 3]

Requisitos:
  1. GPU recomendada (T4 do Kaggle/Colab basta; CPU funciona, mas lento).
  2. Detectores do notebook v20f colados em adtc/detectores.py.
  3. Base contextual em data/base_rag/ (ver README da pasta).
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adtc.corpus import carregar_corpus
from adtc.modelo import TucanoAuditado
from adtc.pipeline import analisar_enunciado
from adtc.rag import RecuperadorContextual, carregar_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="resultados_v20f.csv")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    casos = carregar_corpus()
    recuperador = RecuperadorContextual(carregar_base(), k=args.k)
    tucano = TucanoAuditado()

    acertos = 0
    linhas = []
    for caso in casos:
        reg = analisar_enunciado(caso["enunciado"], tucano, recuperador, k_rag=args.k)
        ok = reg["label"] == caso["rotulo"]
        acertos += ok
        print(f"{caso['caso']}  gabarito={caso['rotulo']:<12} adtc={reg['label']:<12} "
              f"fonte={reg['fonte_decisao']}  {'OK' if ok else 'ERRO'}")
        linhas.append({
            "caso": caso["caso"],
            "enunciado": caso["enunciado"],
            "gabarito": caso["rotulo"],
            "label_adtc": reg["label"],
            "acerto": int(ok),
            "fonte_decisao": reg["fonte_decisao"],
            "forca_evidencia": reg["forca_evidencia"],
            "risco_sobreajuste": reg["risco_sobreajuste"],
            "revisao_humana": reg["revisao_humana"],
            "motivo": reg["motivo"],
        })

    with open(args.saida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)

    print(f"\nAcuracia: {acertos}/{len(casos)} casos "
          f"({100 * acertos / len(casos):.1f}%) — auditoria em {args.saida}")


if __name__ == "__main__":
    main()
