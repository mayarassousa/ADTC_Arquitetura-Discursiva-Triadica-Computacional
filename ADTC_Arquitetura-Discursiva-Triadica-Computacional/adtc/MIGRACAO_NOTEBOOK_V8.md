# Migração do notebook V8 para a estrutura modular

Este documento registra a correspondência entre as seções do notebook monolítico e os módulos adicionados ao repositório.

| Seção do notebook | Responsabilidade | Módulo/arquivo modular |
|---|---|---|
| 0–2 | ambiente, imports e localização de arquivos | `config.py`, `arquivos.py` |
| 3 | IDPT, hashes, anti-vazamento e predições | `corpus.py`, `integridade.py`, `bertimbau.py` |
| 4–5b | taxonomia e marcadores | `taxonomia.py` |
| 6 | métricas, tipos de erro e zonas | `bertimbau.py`, `zonas.py` |
| 7–17 | recuperação híbrida e preparação contextual | `rag.py` |
| 14 | XAI neural | `xai.py` |
| 18 | RAG integrado e re-ranqueamento ADTC | `rag.py` |
| 19 | amostra manual | `avaliacao.py` |
| 19b | leitura da planilha preenchida e métricas | `avaliacao.py`, `pipeline.py` |
| 20 | planilha final da tese | `planilha.py` |
| 21 | relatório e empacotamento | `planilha.py`, `pipeline.py` |

## Alterações estruturais importantes

### 1. Fim do estado global

No notebook, funções consultavam variáveis como `base_rag`, `train_pos_trava` e `OUT_DIR` por meio do escopo global. Na versão modular, esses objetos entram explicitamente como parâmetros ou atributos de configuração.

### 2. Separação entre post-hoc e produção

O notebook usa `tipo_erro` na consulta integrada e na prioridade. O módulo `zonas.py` torna essa escolha explícita por meio de `modo_post_hoc`. Em produção, use `False`.

### 3. RAG com proveniência preservada

O recuperador retorna, para cada candidato, `id`, `fonte_sugerida`, `origem_contexto` e `tipo_contexto`. O re-ranqueamento não apaga a posição original nem os scores intermediários.

### 4. Avaliação manual como etapa formal

`avaliacao.py` define o esquema de preenchimento, calcula Precision@k/Hit@k, integra os julgamentos ao dataset e aponta inconsistências entre `manual_melhor_topk` e as colunas binárias.

### 5. XAI opcional e regime declarado

O módulo `xai.py` registra se o checkpoint é evidencial ou apenas demonstrativo. Falhas de dependência ou carregamento não interrompem o restante do pipeline.

## Notebook preservado

O notebook original é mantido em `notebooks/v8/legado/` para rastreabilidade. O novo `pipeline_v8_modular.ipynb` contém apenas configuração, chamada do pipeline e inspeção das saídas.
