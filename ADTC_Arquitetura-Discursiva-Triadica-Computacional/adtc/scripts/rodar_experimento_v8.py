#!/usr/bin/env python
"""Executa o Pipeline V8 modular.

Exemplo:
python scripts/rodar_experimento_v8.py \
  --treino data/v8/idpt/training_news.csv \
  --teste data/v8/idpt/idpt_news_teste_adtc_corrigido.csv \
  --predicoes data/v8/idpt/idpt_news_predicoes_teste.csv \
  --taxonomia data/v8/taxonomia/quadro_taxonomia_ironia_pb_integrado.csv \
  --base-rag data/v8/base_rag/base_news_rag_curada_com_proveniencia.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adtc.v8 import ConfigV8, PathsV8, PipelineV8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--treino", type=Path, required=True)
    ap.add_argument("--teste", type=Path, required=True)
    ap.add_argument("--predicoes", type=Path, required=True)
    ap.add_argument("--taxonomia", type=Path, required=True)
    ap.add_argument("--base-rag", type=Path, required=True)
    ap.add_argument("--saida", type=Path, default=Path("outputs/v8"))
    ap.add_argument("--avaliacao-manual", type=Path)
    ap.add_argument("--xai-sintese", type=Path)
    ap.add_argument("--somente-tfidf", action="store_true")
    ap.add_argument("--sem-reranker", action="store_true")
    ap.add_argument("--modo-operacional", action="store_true", help="Não usa tipo_erro na consulta/prioridade.")
    args = ap.parse_args()

    paths = PathsV8(
        treino=args.treino,
        teste=args.teste,
        predicoes_teste=args.predicoes,
        taxonomia=args.taxonomia,
        base_rag_curada=args.base_rag,
        saida=args.saida,
        avaliacao_manual=args.avaliacao_manual,
        xai_sintese=args.xai_sintese,
    )
    config = ConfigV8(
        usar_neural=not args.somente_tfidf,
        usar_reranker=not args.sem_reranker,
        modo_post_hoc=not args.modo_operacional,
    )
    resultado = PipelineV8(paths, config).executar()
    print("Pipeline concluído.")
    for nome, caminho in resultado.arquivos.items():
        print(f"- {nome}: {caminho}")


if __name__ == "__main__":
    main()
