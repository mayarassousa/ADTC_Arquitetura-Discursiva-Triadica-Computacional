---
title: ADTC — Detecção Auditável de Ironia (v20f)
emoji: 🦜
colorFrom: green
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
license: mit
language: pt
short_description: Arquitetura Discursiva Triádica Computacional (AD + PLN)
---

# ADTC — Arquitetura Discursiva Triádica Computacional

Demo da versão v20f: detecção auditável de ironia em português
brasileiro, articulando Análise do Discurso francesa e PLN. O
Tucano-2b4-Instruct opera como analisador parcial (geração
determinística); a decisão é tomada por uma camada discursiva explícita,
com travas de literalidade e cascata triádica, e cada item recebe um
registro de auditoria (fonte da decisão, força da evidência, risco de
sobreajuste, recomendação de revisão humana).

Código: https://github.com/SEU_USUARIO/adtc
Tese: Miguel, M. S. S. (2026). FFLCH-USP.

Observação: ao criar o Space pela interface da Hugging Face, mantenha a
linha `sdk_version` que ela gerar automaticamente neste cabeçalho.
