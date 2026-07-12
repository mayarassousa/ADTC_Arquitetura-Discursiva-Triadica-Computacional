"""Orquestração ponta a ponta do Pipeline V8 modular."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .arquivos import ler_csv_robusto, salvar_csv
from .avaliacao import calcular_metricas as calcular_metricas_manuais
from .avaliacao import integrar_avaliacao, ler_avaliacao, preparar_planilha_manual, selecionar_amostra, validar_consistencia
from .bertimbau import calcular_metricas as calcular_metricas_bert, carregar_predicoes, integrar_predicoes
from .config import ConfigV8, PathsV8
from .corpus import carregar_idpt
from .integridade import aplicar_trava_antivazamento, diagnosticar_quase_duplicatas, diagnosticar_split_treino_validacao
from .planilha import gerar_planilha, gerar_relatorio_markdown
from .rag import RecuperadorHibrido, aplicar_rag, construir_base_integrada
from .taxonomia import carregar_taxonomia, integrar_taxonomia, resumos_taxonomia
from .xai import executar_xai
from .zonas import adicionar_contexto_e_prioridade, adicionar_zonas


@dataclass(slots=True)
class ResultadoPipelineV8:
    dataset: pd.DataFrame
    base_rag: pd.DataFrame
    taxonomia: pd.DataFrame
    metricas_bert: pd.DataFrame
    matriz_bert: pd.DataFrame
    diagnostico_integridade: pd.DataFrame
    diagnostico_rag: pd.DataFrame
    amostra_manual: pd.DataFrame
    metricas_manuais: pd.DataFrame
    problemas_avaliacao: pd.DataFrame
    xai_sintese: pd.DataFrame
    arquivos: dict[str, Path]


class PipelineV8:
    def __init__(self, paths: PathsV8, config: ConfigV8 | None = None):
        self.paths = paths
        self.config = config or ConfigV8()
        self.config.validar()
        self.paths.criar_saida()

    def executar(self) -> ResultadoPipelineV8:
        out = self.paths.saida

        # 1. Corpus e integridade
        corpus = carregar_idpt(self.paths.treino, self.paths.teste)
        anti = aplicar_trava_antivazamento(
            corpus.treino,
            corpus.teste,
            remover=self.config.remover_vazamento_exato,
            abortar_se_restante=self.config.abortar_se_overlap_restante,
        )
        diag_split = diagnosticar_split_treino_validacao(anti.treino, seed=self.config.seed)
        diagnostico_integridade = pd.concat(
            [anti.diagnostico.assign(bloco="treino_teste"), diag_split.assign(bloco="treino_validacao")],
            ignore_index=True,
        )
        salvar_csv(diagnostico_integridade, out / "diagnostico_integridade_v8.csv")
        salvar_csv(anti.removidas, out / "linhas_removidas_vazamento_v8.csv")
        if self.config.diagnosticar_quase_duplicatas:
            quase = diagnosticar_quase_duplicatas(anti.treino, corpus.teste, self.config.limiar_quase_duplicata)
            salvar_csv(quase, out / "candidatos_quase_duplicados_v8.csv")

        # 2. BERTimbau congelado
        pred = carregar_predicoes(self.paths.predicoes_teste, corpus.teste)
        teste = integrar_predicoes(corpus.teste, pred)
        metricas_bert, matriz_bert, _ = calcular_metricas_bert(teste)
        salvar_csv(metricas_bert, out / "metricas_bertimbau_v8.csv")
        salvar_csv(matriz_bert, out / "matriz_confusao_bertimbau_v8.csv")

        # 3. Marcadores, taxonomia e zonas
        taxonomia = carregar_taxonomia(self.paths.taxonomia)
        # Caso exista um arquivo de teste ADTC pronto no mesmo caminho do teste,
        # o usuário pode substituir paths.teste por ele; as colunas adicionais são
        # preservadas. Na ausência, marcadores ficam vazios e n_marc_sup=0.
        if "marcadores" not in teste:
            teste["marcadores"] = ""
        if "n_marc_sup" not in teste:
            teste["n_marc_sup"] = teste["marcadores"].fillna("").astype(str).map(lambda s: len([x for x in s.split(";") if x.strip()]))
        teste = integrar_taxonomia(teste, taxonomia)
        teste = adicionar_zonas(teste)
        teste = adicionar_contexto_e_prioridade(teste, modo_post_hoc=self.config.modo_post_hoc)

        # 4. Base integrada e RAG
        base_curada = ler_csv_robusto(self.paths.base_rag_curada)
        base_integrada = construir_base_integrada(base_curada, anti.treino, incluir_seeds=True)
        salvar_csv(base_integrada, out / "base_rag_integrada_adtc_v8.csv")
        recuperador = RecuperadorHibrido(base_integrada, self.config)
        dataset = aplicar_rag(teste, recuperador, self.config)
        salvar_csv(dataset, out / "rag_integrado_adtc_teste_todo_v8_top10.csv")

        diagnostico_rag = pd.DataFrame(
            {
                "item": [
                    "total_teste", "base_rag_integrada", "prioridade_alta", "prioridade_media", "prioridade_baixa",
                    "reranker_aplicado_consultas", "motor", "modelo_embedding", "modelo_reranker",
                ],
                "valor": [
                    len(dataset), len(base_integrada),
                    int(dataset["prioridade_rag_integrado"].eq("alta").sum()),
                    int(dataset["prioridade_rag_integrado"].eq("media").sum()),
                    int(dataset["prioridade_rag_integrado"].eq("baixa").sum()),
                    int(dataset.get("reranker_aplicado", pd.Series(False, index=dataset.index)).fillna(False).astype(bool).sum()),
                    recuperador.motor,
                    recuperador.modelo_embedding or "nenhum",
                    recuperador.modelo_reranker or "nenhum",
                ],
            }
        )
        salvar_csv(diagnostico_rag, out / "diagnostico_rag_integrado_adtc_v8.csv")

        # 5. Avaliação manual
        amostra_base = selecionar_amostra(
            dataset,
            n_alta=self.config.n_prioridade_alta,
            n_media=self.config.n_prioridade_media,
            seed=self.config.seed,
        )
        amostra_manual = preparar_planilha_manual(amostra_base, top_k=self.config.top_k_manual)
        amostra_path = out / "amostra_avaliacao_manual_rag_integrado_adtc_v8.xlsx"
        amostra_manual.to_excel(amostra_path, index=False)
        metricas_manuais = pd.DataFrame()
        problemas_avaliacao = pd.DataFrame()
        if self.paths.avaliacao_manual and self.paths.avaliacao_manual.exists():
            manual = ler_avaliacao(self.paths.avaliacao_manual)
            metricas_manuais = calcular_metricas_manuais(manual)
            problemas_avaliacao = validar_consistencia(manual, top_k=self.config.top_k_manual)
            dataset = integrar_avaliacao(dataset, manual, top_k=self.config.top_k_manual)
            amostra_manual = manual
            salvar_csv(metricas_manuais, out / "metricas_avaliacao_manual_rag_integrado_v8.csv")
            salvar_csv(problemas_avaliacao, out / "diagnostico_consistencia_avaliacao_manual_v8.csv")

        # 6. XAI opcional
        xai_sintese = pd.DataFrame()
        if self.paths.xai_sintese and self.paths.xai_sintese.exists():
            xai_sintese = ler_csv_robusto(self.paths.xai_sintese)
        elif self.config.executar_xai:
            xai = executar_xai(
                dataset,
                modelo_nome=self.config.caminho_modelo_xai,
                modelo_evidencial=not self.config.permitir_modelo_base_demo,
                n_por_tipo=self.config.n_amostras_xai_por_tipo,
                max_len=self.config.max_len_xai,
                seed=self.config.seed,
            )
            xai_sintese = xai.tabela
            salvar_csv(xai.status, out / "xai_status_modelo_v8.csv")
            if not xai_sintese.empty:
                salvar_csv(xai_sintese, out / "xai_sintese_v8.csv")
        if not xai_sintese.empty and "id_texto_original" in xai_sintese:
            dataset = dataset.merge(xai_sintese.drop_duplicates("id_texto_original"), on="id_texto_original", how="left")

        # 7. Artefatos finais
        planilha = gerar_planilha(
            out / "planilha_final_tese_v8_rag_integrado_adtc_xai.xlsx",
            dataset=dataset,
            base_rag=base_integrada,
            taxonomia=taxonomia,
            metricas_bert=metricas_bert,
            matriz_bert=matriz_bert,
            diagnostico_integridade=diagnostico_integridade,
            diagnostico_rag=diagnostico_rag,
            amostra_manual=amostra_manual,
            metricas_manuais=metricas_manuais,
            xai_sintese=xai_sintese,
        )
        relatorio = gerar_relatorio_markdown(
            out / "RELATORIO_TECNICO_V8_MODULAR.md",
            dataset,
            metricas_bert,
            metricas_manuais,
        )
        salvar_csv(dataset, out / "dataset_final_v8_auditado.csv")

        categorias, detectabilidade, mapa = resumos_taxonomia(taxonomia)
        salvar_csv(categorias, out / "taxonomia_resumo_categorias_v8.csv")
        salvar_csv(detectabilidade, out / "taxonomia_resumo_detectabilidade_v8.csv")
        salvar_csv(mapa, out / "mapa_marcadores_taxonomia_v8.csv")

        arquivos = {
            "planilha_final": planilha,
            "relatorio": relatorio,
            "amostra_manual": amostra_path,
            "dataset_final": out / "dataset_final_v8_auditado.csv",
        }
        return ResultadoPipelineV8(
            dataset=dataset,
            base_rag=base_integrada,
            taxonomia=taxonomia,
            metricas_bert=metricas_bert,
            matriz_bert=matriz_bert,
            diagnostico_integridade=diagnostico_integridade,
            diagnostico_rag=diagnostico_rag,
            amostra_manual=amostra_manual,
            metricas_manuais=metricas_manuais,
            problemas_avaliacao=problemas_avaliacao,
            xai_sintese=xai_sintese,
            arquivos=arquivos,
        )
