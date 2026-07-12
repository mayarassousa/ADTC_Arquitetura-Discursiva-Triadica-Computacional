# Dados do Pipeline V8

A estrutura esperada é:

```text
data/v8/
  idpt/
    training_news.csv
    idpt_news_teste_adtc_corrigido.csv
    idpt_news_predicoes_teste.csv
  taxonomia/
    quadro_taxonomia_ironia_pb_integrado.csv
  base_rag/
    base_news_rag_curada_com_proveniencia.csv
```

Os arquivos do IDPT não são redistribuídos neste pacote. Use a cópia cedida/autorizada para a pesquisa e mantenha os créditos aos criadores do corpus.

O teste pode ser o arquivo bruto, mas a reprodução da camada ADTC exige as colunas `marcadores` e `n_marc_sup`, produzidas na etapa anterior de extração.
