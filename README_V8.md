# ADTC V8 — BERTimbau, explicabilidade discursiva e auditabilidade

Esta pasta acrescenta ao repositório da ADTC a implementação modular do **Pipeline V8**, originalmente desenvolvido em notebook. A V8 não substitui a implementação v20f/Tucano já existente: ela constitui outro experimento da tese, voltado ao **IDPT News**, com **BERTimbau pré-computado** como modelo de referência e a ADTC como camada de auditoria.

## Proposição metodológica

> O BERTimbau decide; a ADTC audita; o RAG recupera evidências potenciais; a avaliação humana decide sua pertinência.

A contribuição da V8 não é um novo classificador. O pipeline preserva a decisão neural e produz, para cada texto, uma trilha verificável contendo:

- classe, probabilidade, confiança e tipo de erro no regime post-hoc;
- zona ADTC e prioridade de auditoria;
- marcadores linguístico-discursivos e recursos taxonômicos;
- tipo de contexto necessário;
- candidatos recuperados, scores e proveniência;
- avaliação humana da pertinência;
- XAI neural opcional, com o regime evidencial explicitamente registrado.

## Estrutura adicionada

```text
adtc/
  v8/
    config.py        configurações e caminhos
    arquivos.py      localização e leitura robusta
    corpus.py        carregamento/padronização do IDPT News
    integridade.py   hash, anti-vazamento e quase duplicatas
    bertimbau.py     predições pré-computadas e métricas
    taxonomia.py     marcadores -> recursos da taxonomia ADTC
    zonas.py         zonas, contexto necessário e prioridade
    rag.py           TF-IDF + embeddings + reranker + score ADTC
    avaliacao.py     amostra manual, Precision@k, Hit@k e consistência
    xai.py           LIME, IG e atenção em regime opcional
    planilha.py      dossiê final em Excel e relatório Markdown
    pipeline.py      orquestração ponta a ponta

data/v8/
  idpt/              arquivos do corpus autorizados para uso
  taxonomia/         taxonomia ADTC
  base_rag/          base curada e documentação de proveniência

notebooks/v8/
  pipeline_v8_modular.ipynb
  legado/            notebook monolítico preservado para rastreabilidade

scripts/
  rodar_experimento_v8.py
  integrar_avaliacao_manual.py
  gerar_planilha_final.py

tests/v8/
  test_integridade.py
  test_metricas_rag.py
  test_smoke_v8.py
```

## Instalação

```bash
cd adtc
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate     # Windows
pip install -r requirements-v8.txt
```

Para XAI completa:

```bash
pip install -r requirements-xai.txt
```

## Arquivos de entrada

Coloque os artefatos nas pastas descritas em `data/v8/README.md`. O comando mínimo exige:

- treino do IDPT News;
- teste com rótulo e, preferencialmente, marcadores ADTC já extraídos;
- predições pré-computadas do BERTimbau;
- taxonomia ADTC;
- base RAG curada com proveniência.

Os dados do IDPT não são incluídos automaticamente neste pacote. Use somente os arquivos cujo uso e redistribuição estejam autorizados.

## Execução

```bash
python scripts/rodar_experimento_v8.py \
  --treino data/v8/idpt/training_news.csv \
  --teste data/v8/idpt/idpt_news_teste_adtc_corrigido.csv \
  --predicoes data/v8/idpt/idpt_news_predicoes_teste.csv \
  --taxonomia data/v8/taxonomia/quadro_taxonomia_ironia_pb_integrado.csv \
  --base-rag data/v8/base_rag/base_news_rag_curada_com_proveniencia.csv \
  --saida outputs/v8
```

Para uma execução sem downloads de modelos neurais, use:

```bash
python scripts/rodar_experimento_v8.py ... --somente-tfidf --sem-reranker
```

Para regenerar os resultados após preencher a avaliação manual:

```bash
python scripts/rodar_experimento_v8.py ... \
  --avaliacao-manual outputs/v8/amostra_avaliacao_manual_rag_integrado_adtc_v8_preenchida.xlsx
```

## Regime post-hoc e regime operacional

A reprodução estrita do experimento usa `modo_post_hoc=True`: o tipo de erro pode entrar na prioridade e na consulta porque o ouro está disponível para análise retrospectiva. Isso não deve ser usado em produção.

Para excluir qualquer variável dependente do rótulo-ouro:

```bash
python scripts/rodar_experimento_v8.py ... --modo-operacional
```

## Precision@k e Hit@k

- **Precision@k**: proporção média de fragmentos pertinentes entre as `k` primeiras posições.
- **Hit@k**: proporção de casos em que há pelo menos um fragmento pertinente entre as `k` primeiras posições.

Assim, Hit@5 responde “o sistema encontrou ao menos uma evidência útil?”, enquanto Precision@5 responde “quão concentradas estão as evidências úteis no top-5?”.

## Saídas principais

```text
outputs/v8/
  dataset_final_v8_auditado.csv
  planilha_final_tese_v8_rag_integrado_adtc_xai.xlsx
  amostra_avaliacao_manual_rag_integrado_adtc_v8.xlsx
  metricas_bertimbau_v8.csv
  metricas_avaliacao_manual_rag_integrado_v8.csv
  diagnostico_integridade_v8.csv
  diagnostico_rag_integrado_adtc_v8.csv
  RELATORIO_TECNICO_V8_MODULAR.md
```

## Demo

A aplicação Gradio é um **explorador de resultados**, e não um classificador novo:

```bash
export ADTC_V8_PLANILHA=outputs/v8/planilha_final_tese_v8_rag_integrado_adtc_xai.xlsx
python app_v8.py
```

## Testes

```bash
pytest tests/v8 -q
```

## Limites declarados

1. O score `discursivo` é uma heurística auditável informada pela ADTC; não equivale à modelagem plena das condições de produção.
2. A melhora do pipeline integrado não isola causalmente a contribuição de cada componente sem estudo de ablação.
3. A XAI só é evidencial quando usa o mesmo checkpoint que produziu as predições auditadas.
4. A pertinência contextual é definida pela avaliação humana, não pelo score automático.
5. O treino usado no RAG funciona como memória textual sem rótulo; não retreina o BERTimbau.

## Citação

Consulte `CITATION_V8.cff`.
