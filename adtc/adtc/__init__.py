"""ADTC — Arquitetura Discursiva Triádica Computacional.

Implementação de referência da versão v20f, conforme documentada na tese
de doutorado de Mayara Suellen de Sousa Miguel (FFLCH-USP).

O Tucano-2b4-Instruct é empregado como analisador parcial (testemunha
auditada), nunca como classificador autônomo: suas leituras por eixo são
submetidas a uma instância de decisão discursiva explícita e auditável.
"""

__version__ = "0.20.6"  # v20f

from adtc.corpus import carregar_corpus
from adtc.decisao import regra_adtc_v20f
from adtc.travas import texto_eh_literal_factual, texto_eh_literal_discursivo

__all__ = [
    "carregar_corpus",
    "regra_adtc_v20f",
    "texto_eh_literal_factual",
    "texto_eh_literal_discursivo",
]
