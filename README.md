# ADTC — Arquitetura Discursiva Triádica Computacional

Framework teórico-metodológico que articula a Análise do Discurso de
matriz francesa (Pêcheux, Orlandi) e o Processamento de Linguagem
Natural, tendo a detecção de ironia em português brasileiro como caso de
validação. Este repositório implementa a versão **v20f**, documentada na
tese de doutorado de Mayara Suellen de Sousa Miguel (FFLCH-USP).

## A arquitetura

A ironia sem marcas textuais explícitas expõe uma insuficiência
**estrutural**, não quantitativa, do PLN convencional. A ADTC responde a
isso processando cada enunciado em três eixos simultâneos: o **Eixo
Textual** (intradiscurso, a materialidade linguística), o **Eixo
Contextual** (condições de produção e interdiscurso, operacionalizadas
por recuperação RAG) e o **Eixo do Sujeito** (posição enunciativa).

Na versão v20f, o modelo generativo **Tucano-2b4-Instruct** (Corrêa et
al., 2024) opera como **analisador parcial, não como classificador**:
uma testemunha cujas leituras, obtidas por geração determinística
(`do_sample=False`), são submetidas a uma camada de decisão discursiva
explícita. Essa camada combina duas travas de literalidade (factual e
discursiva/fraseológica) e uma cascata triádica de regras, e produz, por
item, um registro de auditoria: fonte da decisão, força da evidência,
risco de sobreajuste e recomendação de revisão humana.

## Resultados (v20f)

Sobre o corpus de demonstração de 41 casos (164 itens processados), a
arquitetura acertou 39 casos (95,1% dos itens). Antes da camada
discursiva, o Tucano leu ironia em todos os 164 itens, inclusive nos
factuais e fraseológicos; as travas reescreveram o sinal de contraste em
64 itens e o suavizaram em outros 8. Os dois erros remanescentes são
rastreáveis a uma questão de prioridade entre cláusulas, e não a um
raciocínio opaco: quando a ADTC erra, o erro é localizável e corrigível.

O corpus é curado por opção e não mede desempenho generalizável; ele
estressa a distinção entre ironia, literalidade, ambiguidade e
indecidibilidade.

## Estrutura

```
adtc/
  corpus.py      carregamento do corpus v20f
  prompts.py     perguntas dos quatro eixos (verbatim da tese)
  rag.py         eixo contextual: TF-IDF + similaridade de cosseno
  modelo.py      Tucano-2b4-Instruct, geração determinística
  sinais.py      respostas livres -> sinais discursivos
  detectores.py  *** colar aqui as funções do notebook v20f ***
  travas.py      travas de literalidade (verbatim da tese)
  decisao.py     cascata triádica v20f (verbatim da tese)
  pipeline.py    orquestração: enunciado -> 4 eixos -> decisão auditada
data/
  corpus_v20f.csv   41 casos (Quadro 1 da tese)
  base_rag/         documentos de contexto + distratores (adicionar)
scripts/rodar_experimento.py   execução completa com conferência cega
tests/test_smoke.py            teste de infraestrutura (não reproduz a tese)
app.py                          demo Gradio (Hugging Face Spaces)
```

## Reprodução

1. `pip install -r requirements.txt`
2. Cole em `adtc/detectores.py` as funções do notebook Kaggle v20f
   (os stubs indicam nome e assinatura) e complete as linhas de
   pontuação marcadas em `adtc/decisao.py`.
3. Exporte a base contextual do notebook para `data/base_rag/`.
4. `python scripts/rodar_experimento.py`

GPU recomendada (a T4 gratuita do Kaggle basta); em CPU a execução
funciona, porém lenta. O gabarito é cego: nunca é apresentado ao modelo,
sendo usado apenas na conferência posterior.

## Citação

Miguel, M. S. S. (2026). *Arquitetura Discursiva Triádica
Computacional*. Tese (Doutorado em Filologia e Língua Portuguesa),
FFLCH, Universidade de São Paulo. Ver `CITATION.cff`.

## Licenças

Código sob MIT (`LICENSE`). Corpus sob CC BY 4.0 (ver
`hf_dataset/README_dataset.md`). O modelo Tucano possui licença própria;
consulte o card oficial em huggingface.co/TucanoBR.
