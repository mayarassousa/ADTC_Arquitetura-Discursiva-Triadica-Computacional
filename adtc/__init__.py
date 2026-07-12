"""ADTC V8: BERTimbau pré-computado + auditoria discursiva + RAG integrado.

Este subpacote refatora o notebook V8 em componentes reutilizáveis. Ele coexiste
com a implementação v20f já presente no repositório e não a substitui.
"""

from .config import ConfigV8, PathsV8
from .pipeline import PipelineV8, ResultadoPipelineV8

__all__ = ["ConfigV8", "PathsV8", "PipelineV8", "ResultadoPipelineV8"]
__version__ = "0.8.0"
