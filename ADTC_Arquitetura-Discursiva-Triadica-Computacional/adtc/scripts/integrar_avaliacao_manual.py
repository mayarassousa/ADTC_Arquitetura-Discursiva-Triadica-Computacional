#!/usr/bin/env python
"""Valida e calcula as métricas da avaliação manual V8."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adtc.v8.avaliacao import calcular_metricas, ler_avaliacao, validar_consistencia


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("planilha", type=Path)
    ap.add_argument("--saida", type=Path, default=Path("outputs/v8/metricas_avaliacao_manual_v8.csv"))
    args = ap.parse_args()
    df = ler_avaliacao(args.planilha)
    metricas = calcular_metricas(df)
    problemas = validar_consistencia(df)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(args.saida, index=False)
    problemas.to_csv(args.saida.with_name("diagnostico_consistencia_" + args.saida.name), index=False)
    print(metricas.to_string(index=False))
    print(f"Inconsistências: {len(problemas)}")


if __name__ == "__main__":
    main()
