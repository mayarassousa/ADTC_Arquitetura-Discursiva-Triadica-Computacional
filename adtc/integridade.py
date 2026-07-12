"""Travas de integridade e diagnóstico de sobreposição entre partições."""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

_HASH_LABEL = re.compile(r"#(?:ironia|sarcasmo|ironic|irony|sqn)\b", re.I)
_URL = re.compile(r"http\S+|www\.\S+")


def normalizar_texto_hash(texto) -> str:
    texto = "" if not isinstance(texto, str) else texto
    texto = texto.replace("\u00a0", " ")
    texto = _HASH_LABEL.sub(" ", texto)
    texto = _URL.sub(" URL ", texto)
    texto = texto.lower()
    return re.sub(r"\s+", " ", texto).strip()


def hash_texto(texto) -> str:
    return hashlib.md5(normalizar_texto_hash(texto).encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ResultadoAntivazamento:
    treino: pd.DataFrame
    removidas: pd.DataFrame
    diagnostico: pd.DataFrame


def aplicar_trava_antivazamento(
    treino: pd.DataFrame,
    teste: pd.DataFrame,
    remover: bool = True,
    abortar_se_restante: bool = True,
) -> ResultadoAntivazamento:
    treino = treino.copy()
    teste = teste.copy()
    if "text_hash" not in treino.columns:
        treino["text_hash"] = treino["text"].map(hash_texto)
    if "text_hash" not in teste.columns:
        teste["text_hash"] = teste["text"].map(hash_texto)

    hashes_teste = set(teste["text_hash"].astype(str))
    mask = treino["text_hash"].astype(str).isin(hashes_teste)
    removidas = treino.loc[mask].copy()
    treino_pos = treino.loc[~mask].copy() if remover else treino.copy()
    overlap_restante = len(set(treino_pos["text_hash"].astype(str)) & hashes_teste)
    diagnostico = pd.DataFrame(
        {
            "item": [
                "treino_bruto_original",
                "teste",
                "vazamentos_exatos_encontrados",
                "linhas_removidas",
                "treino_pos_trava",
                "overlap_hash_pos_trava",
            ],
            "valor": [len(treino), len(teste), int(mask.sum()), len(removidas) if remover else 0, len(treino_pos), overlap_restante],
        }
    )
    if abortar_se_restante and overlap_restante:
        raise RuntimeError(f"Overlap treino/teste ainda presente: {overlap_restante} hashes.")
    return ResultadoAntivazamento(treino_pos.reset_index(drop=True), removidas, diagnostico)


def diagnosticar_split_treino_validacao(treino: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    groups = treino["text_hash"].astype(str).to_numpy()
    y = treino["label"].astype(int).to_numpy()
    try:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        idx_treino, idx_val = next(sgkf.split(treino, y, groups=groups))
    except Exception as exc:
        warnings.warn(f"StratifiedGroupKFold falhou ({exc!r}); usando split estratificado.")
        try:
            idx_treino, idx_val = train_test_split(
                np.arange(len(treino)), test_size=0.10, random_state=seed, stratify=y
            )
        except ValueError:
            # Conjuntos muito pequenos podem não comportar um split estratificado.
            test_size = max(1, int(round(len(treino) * 0.10)))
            if test_size >= len(treino):
                test_size = 1
            idx_treino, idx_val = train_test_split(
                np.arange(len(treino)), test_size=test_size, random_state=seed, stratify=None
            )
    h_t = set(treino.iloc[idx_treino]["text_hash"].astype(str))
    h_v = set(treino.iloc[idx_val]["text_hash"].astype(str))
    return pd.DataFrame(
        {
            "item": ["treino_diagnostico", "validacao_diagnostico", "hashes_comuns_treino_validacao"],
            "valor": [len(idx_treino), len(idx_val), len(h_t & h_v)],
        }
    )


def diagnosticar_quase_duplicatas(
    treino: pd.DataFrame,
    teste: pd.DataFrame,
    limiar: float = 0.985,
    max_features: int = 50_000,
) -> pd.DataFrame:
    tr = treino[["text_hash", "texto_modelo"]].drop_duplicates("text_hash")
    te = teste[["text_hash", "texto_modelo"]].drop_duplicates("text_hash")
    corpus = pd.concat([tr["texto_modelo"], te["texto_modelo"]], ignore_index=True)
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(3, 5), analyzer="char")
    matriz = vec.fit_transform(corpus)
    sim = cosine_similarity(matriz[len(tr):], matriz[: len(tr)])
    max_sim = sim.max(axis=1)
    argmax = sim.argmax(axis=1)
    out = pd.DataFrame(
        {
            "test_hash": te["text_hash"].to_numpy(),
            "train_hash_mais_proximo": tr["text_hash"].to_numpy()[argmax],
            "similaridade_char_tfidf": max_sim,
        }
    )
    return out[out["similaridade_char_tfidf"] >= limiar].reset_index(drop=True)
