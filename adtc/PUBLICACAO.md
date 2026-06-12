# Guia de publicação: GitHub + Hugging Face

## Antes de publicar (checklist)

1. Colar em `adtc/detectores.py` as funções do notebook Kaggle v20f.
2. Completar as linhas de pontuação marcadas em `adtc/decisao.py`
   ("COMPLETAR (notebook v20f)").
3. Conferir `adtc/sinais.py` e `adtc/prompts.py` (montagem do prompt)
   contra o notebook, para fidelidade de reprodução.
4. Exportar a base contextual para `data/base_rag/`.
5. Rodar `python scripts/rodar_experimento.py` no Kaggle (T4) e conferir
   se reproduz 39/41. Salvar o CSV de auditoria gerado.
6. Conferir as fontes das manchetes do corpus antes de publicar o
   dataset (nota no card).
7. Trocar `SEU_USUARIO` nos arquivos pelos seus usuários reais.

## 1. GitHub

Pela linha de comando, dentro da pasta do repositório:

```bash
git init
git add .
git commit -m "ADTC v20f: implementação de referência"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/adtc.git
git push -u origin main
```

(Crie antes o repositório vazio `adtc` em github.com/new. Alternativa
sem terminal: criar o repositório no site e arrastar os arquivos em
"uploading an existing file".)

## 2. Hugging Face: corpus como dataset

1. huggingface.co/new-dataset → nome sugerido: `adtc-corpus-v20f`.
2. Envie `data/corpus_v20f.csv`.
3. Use `hf_dataset/README_dataset.md` como README do dataset (o
   cabeçalho YAML já traz licença, idioma e tags).

O visualizador de datasets da HF lê o CSV automaticamente. Não há
modelo a publicar na aba Models: a v20f usa o Tucano original, sem
ajuste fino (um eventual modelo ajustado sobre a ADTC, previsto na
conclusão da tese, seria publicado lá no futuro).

## 3. Hugging Face: demo executável (Space)

1. huggingface.co/new-space → SDK **Gradio**.
2. Hardware:
   - **CPU basic (gratuito)**: 2 vCPU e 16 GB de RAM; o Tucano-2b4 cabe
     na memória, mas cada análise (4 gerações) leva alguns minutos.
     Funciona como prova de conceito.
   - **ZeroGPU**: GPU compartilhada alocada só durante a inferência,
     gratuita para quem usa o Space; para hospedar é preciso conta PRO
     (US$ 9/mês). Recomendado para a demo pública. Nesse caso,
     descomente `spaces` em `requirements.txt`.
3. Envie para o Space: `app.py`, a pasta `adtc/`, a pasta `data/`
   (com `base_rag/` preenchida) e `requirements.txt`.
4. Substitua o README gerado pelo conteúdo de
   `hf_space/README_space.md`, mantendo a linha `sdk_version` que a HF
   criar.
5. O Space compila sozinho; acompanhe a aba "Logs" na primeira
   execução (o download do Tucano ocorre uma única vez).

## 4. Amarração

No README do GitHub, inclua os links do dataset e do Space; nos cards
da HF, o link do GitHub e a referência da tese. Os três pontos passam a
se citar mutuamente, e o `CITATION.cff` faz o GitHub exibir o botão
"Cite this repository".
