"""Detectores discursivos auxiliares das travas de literalidade.

ATENCAO — PONTO DE COLAGEM DO NOTEBOOK v20f.

As funcoes abaixo sao referenciadas pelas travas e pela cascata de
decisao documentadas na tese, mas suas implementacoes completas residem
no notebook Kaggle v20f. Para reproduzir os resultados reportados
(95,1% de acuracia; 39/41 casos), cole aqui, sem alteracao, as versoes
do notebook. Os stubs levantam erro proposital para impedir que uma
implementacao divergente produza resultados atribuidos indevidamente a
v20f.
"""

import re
import unicodedata

_MSG = (
    "Funcao '{nome}' e um stub: cole a implementacao do notebook Kaggle "
    "v20f neste modulo (adtc/detectores.py) antes de executar o pipeline."
)


def _norm_xai(texto):
    """Normalizacao usada pelos detectores (maiusculas, sem acentos)."""
    t = unicodedata.normalize("NFD", texto or "")
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", t).upper().strip()


def texto_tem_contraste_forte(texto):
    raise NotImplementedError(_MSG.format(nome="texto_tem_contraste_forte"))


def texto_tem_memoria_critica(texto):
    raise NotImplementedError(_MSG.format(nome="texto_tem_memoria_critica"))


def texto_eh_ambiguo_curto(texto):
    raise NotImplementedError(_MSG.format(nome="texto_eh_ambiguo_curto"))


def texto_eh_indecidivel_curto(texto):
    raise NotImplementedError(_MSG.format(nome="texto_eh_indecidivel_curto"))


def texto_tem_emoticon_contraste(texto):
    raise NotImplementedError(_MSG.format(nome="texto_tem_emoticon_contraste"))


def texto_eh_noticia_ou_explicativo_factual(texto):
    raise NotImplementedError(_MSG.format(nome="texto_eh_noticia_ou_explicativo_factual"))


def texto_eh_fraseologia_literal(texto):
    raise NotImplementedError(_MSG.format(nome="texto_eh_fraseologia_literal"))


def texto_eh_afetivo_ou_reclamacao_literal_curta(texto):
    raise NotImplementedError(_MSG.format(nome="texto_eh_afetivo_ou_reclamacao_literal_curta"))


def motivo_literalidade_discursiva(texto):
    raise NotImplementedError(_MSG.format(nome="motivo_literalidade_discursiva"))
