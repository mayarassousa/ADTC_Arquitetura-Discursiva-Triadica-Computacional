"""Configurações centrais da execução V8.

Os valores padrão reproduzem as decisões metodológicas do notebook final:
- BERTimbau não é retreinado;
- recuperação lexical e densa trabalham em conjunto;
- o pool é formado por até 60 candidatos por canal;
- o re-ranqueamento final combina 60% de score semântico e 40% de score
  heurístico discursivamente informado pela ADTC;
- avaliação manual é a instância final de pertinência contextual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(slots=True)
class PathsV8:
    treino: Path
    teste: Path
    predicoes_teste: Path
    taxonomia: Path
    base_rag_curada: Path
    saida: Path = Path("outputs/v8")
    predicoes_validacao: Path | None = None
    avaliacao_manual: Path | None = None
    xai_sintese: Path | None = None

    def criar_saida(self) -> Path:
        self.saida.mkdir(parents=True, exist_ok=True)
        return self.saida


@dataclass(slots=True)
class ConfigV8:
    seed: int = 42
    remover_vazamento_exato: bool = True
    abortar_se_overlap_restante: bool = True
    diagnosticar_quase_duplicatas: bool = False
    limiar_quase_duplicata: float = 0.985

    # RAG semântico
    usar_neural: bool = True
    usar_reranker: bool = True
    embedding_models: Sequence[str] = field(default_factory=lambda: (
        "BAAI/bge-m3",
        "intfloat/multilingual-e5-base",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ))
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    peso_lexical: float = 0.30
    peso_denso: float = 0.70
    candidatos_por_canal: int = 60
    top_k_bruto: int = 20
    top_k_final: int = 10

    # Re-ranqueamento informado pela ADTC
    peso_semantico_integrado: float = 0.60
    peso_discursivo_integrado: float = 0.40

    # Amostragem manual
    n_prioridade_alta: int = 70
    n_prioridade_media: int = 20
    top_k_manual: int = 5

    # Regime experimental. Quando True, tipo_erro pode ser usado na prioridade
    # e na consulta. Em produção, deve ser False porque tipo_erro depende do ouro.
    modo_post_hoc: bool = True

    # XAI
    executar_xai: bool = False
    caminho_modelo_xai: str = "neuralmind/bert-base-portuguese-cased"
    permitir_modelo_base_demo: bool = True
    max_len_xai: int = 256
    n_amostras_xai_por_tipo: int = 2

    def validar(self) -> None:
        if not 0 <= self.peso_lexical <= 1 or not 0 <= self.peso_denso <= 1:
            raise ValueError("Pesos lexical e denso devem estar entre 0 e 1.")
        if abs((self.peso_lexical + self.peso_denso) - 1.0) > 1e-9:
            raise ValueError("peso_lexical + peso_denso deve ser igual a 1.")
        if abs((self.peso_semantico_integrado + self.peso_discursivo_integrado) - 1.0) > 1e-9:
            raise ValueError("Pesos do score integrado devem somar 1.")
        if self.top_k_final > self.top_k_bruto:
            raise ValueError("top_k_final não pode exceder top_k_bruto.")
