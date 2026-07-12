"""Recuperação híbrida e re-ranqueamento heurístico informado pela ADTC.

O módulo preserva a distinção metodológica central da V8:
- similaridade semântica não equivale a pertinência discursiva;
- o score chamado ``discursivo`` é uma heurística explícita e auditável;
- a pertinência final é decidida na avaliação humana.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import ConfigV8
from .zonas import classificar_contexto_necessario

STOP_PT = {
    "a","o","as","os","um","uma","uns","umas","de","do","da","dos","das","em","no","na","nos","nas",
    "por","para","com","sem","sob","sobre","entre","até","apos","após","antes","e","ou","mas","que","se",
    "ao","aos","à","às","como","mais","menos","muito","muita","muitos","muitas","também","ja","já","foi",
    "ser","são","era","eram","ter","tem","têm","há","não","sim","sua","seu","suas","seus","ele","ela",
    "eles","elas","isso","isto","aquele","aquela","aqueles","aquelas","este","esta","estes","estas",
    "diz","disse","afirmou","segundo","contra","durante","onde","quando","porque","ainda","apenas",
    "anos","ano","dia","dias","meses","mês","vai","pode","brasil","brasileiro","brasileira",
}


def normalizar_texto(texto) -> str:
    texto = "" if pd.isna(texto) else str(texto)
    return re.sub(r"\s+", " ", texto).strip().lower()


def fold(texto) -> str:
    texto = normalizar_texto(texto)
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


def tokens_chave(texto, max_terms: int = 24) -> list[str]:
    texto = "" if pd.isna(texto) else str(texto)
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9_-]{3,}", texto)
    freq: dict[str, int] = {}
    ordem: list[str] = []
    for token in tokens:
        chave = token.lower()
        if chave in STOP_PT:
            continue
        if chave not in freq:
            freq[chave] = 0
            ordem.append(chave)
        freq[chave] += 1
    return sorted(freq, key=lambda x: (-freq[x], ordem.index(x)))[:max_terms]


def entidades(texto, max_ent: int = 14) -> list[str]:
    texto = "" if pd.isna(texto) else str(texto)
    candidatos = re.findall(
        r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç]+){0,3}",
        texto,
    )
    vistos: set[str] = set()
    saida: list[str] = []
    for item in candidatos:
        item = item.strip()
        if len(item) < 4 or item.lower() in STOP_PT or item in vistos:
            continue
        vistos.add(item)
        saida.append(item)
        if len(saida) >= max_ent:
            break
    return saida


def jaccard(a, b, modo: str = "tokens") -> float:
    if modo == "entidades":
        sa = {x.lower() for x in entidades(a, 15)}
        sb = {x.lower() for x in entidades(b, 15)}
    else:
        sa = set(tokens_chave(a, 30))
        sb = set(tokens_chave(b, 30))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def normalizar_score(valores: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(valores), dtype=float)
    if arr.size == 0:
        return arr
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def memoria_discursiva_seed() -> pd.DataFrame:
    registros = [
        {
            "id": "memoria_covid_gripezinha",
            "tema": "pandemia covid gripezinha cloroquina vacina negacionismo",
            "fragmento": "Memória discursiva associada à pandemia de Covid-19 no Brasil: termos como gripezinha, cloroquina, isolamento, vacina e negação da gravidade podem funcionar como gatilhos de leitura irônica quando há contraste entre minimização discursiva e gravidade factual.",
            "fonte_sugerida": "Base ADTC de memória discursiva — Covid/pandemia",
            "origem_contexto": "memoria_discursiva_adtc",
            "tipo_contexto": "memoria_sanitaria_politica",
        },
        {
            "id": "memoria_aspas_jornalismo",
            "tema": "aspas discurso relatado jornalismo citação fonte enunciativa",
            "fragmento": "No jornalismo, aspas e discurso relatado podem indicar citação, atribuição de fala, distanciamento enunciativo ou responsabilidade da fonte. Essas marcas podem ser falsos gatilhos para classificadores de ironia, pois nem toda citação ou formulação entre aspas é irônica.",
            "fonte_sugerida": "Base ADTC — aspas e discurso relatado no jornalismo",
            "origem_contexto": "memoria_discursiva_adtc",
            "tipo_contexto": "fonte_enunciativa_citacao",
        },
        {
            "id": "memoria_adversativa_negacao",
            "tema": "negação adversativa concessiva contraste mas porém embora",
            "fragmento": "Negação, adversativas e concessivas sinalizam contraste argumentativo. Esse contraste pode sustentar ironia, mas também é regular em notícia informativa ou texto opinativo. A pertinência depende do alvo, do evento e da posição enunciativa.",
            "fonte_sugerida": "Base ADTC — contraste argumentativo",
            "origem_contexto": "memoria_discursiva_adtc",
            "tipo_contexto": "contraste_argumentativo",
        },
        {
            "id": "memoria_avaliacao_lexical",
            "tema": "adjetivo avaliativo hipérbole aumentativo diminutivo intensificação julgamento",
            "fragmento": "Avaliativos, hipérboles, aumentativos e diminutivos podem intensificar julgamento. Em ironia, produzem tensão entre elogio aparente e crítica implícita; em notícia real, podem apenas relatar avaliação atribuída a uma fonte.",
            "fonte_sugerida": "Base ADTC — avaliação lexical e intensificação",
            "origem_contexto": "memoria_discursiva_adtc",
            "tipo_contexto": "avaliacao_lexical_intensificacao",
        },
        {
            "id": "memoria_7a1",
            "tema": "futebol copa seleção 7 a 1 alemanha vergonha nacional",
            "fragmento": "Memória discursiva do 7 a 1: a derrota da seleção brasileira para a Alemanha em 2014 tornou-se marcador de fracasso, humilhação e comparação irônica em discursos esportivos, políticos e midiáticos.",
            "fonte_sugerida": "Base ADTC de memória discursiva — esporte",
            "origem_contexto": "memoria_discursiva_adtc",
            "tipo_contexto": "memoria_esportiva",
        },
        {
            "id": "memoria_padrao_fifa",
            "tema": "padrão fifa obras públicas estádio superfaturamento promessa legado copa",
            "fragmento": "Memória discursiva de 'padrão FIFA': expressão associada a promessas de qualidade, obras públicas, estádios e críticas a gastos. Pode sustentar ironia por contraste entre promessa institucional e precariedade percebida.",
            "fonte_sugerida": "Base ADTC de memória discursiva — política e Copa",
            "origem_contexto": "memoria_discursiva_adtc",
            "tipo_contexto": "memoria_politica_midiatica",
        },
    ]
    return pd.DataFrame(registros)


def construir_base_integrada(base_curada: pd.DataFrame, treino: pd.DataFrame, incluir_seeds: bool = True) -> pd.DataFrame:
    partes: list[pd.DataFrame] = []
    base = base_curada.copy()
    if "id" not in base:
        base["id"] = [f"curada_{i}" for i in range(len(base))]
    if "tema" not in base:
        base["tema"] = ""
    if "fragmento" not in base:
        base["fragmento"] = base.astype(str).agg(" ".join, axis=1)
    if "fonte_sugerida" not in base:
        base["fonte_sugerida"] = "Base curada ADTC/RAG"
    base["origem_contexto"] = base.get("origem_contexto", "base_curada_adtc")
    base["tipo_contexto"] = base.get("tipo_contexto", "contexto_curado")
    partes.append(base[["id", "tema", "fragmento", "fonte_sugerida", "origem_contexto", "tipo_contexto"]])

    if incluir_seeds:
        partes.append(memoria_discursiva_seed())

    if not treino.empty:
        tmp = treino[["text"]].copy().reset_index().rename(columns={"index": "idx", "text": "fragmento"})
        tmp["fragmento"] = tmp["fragmento"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str[:1800]
        tmp = tmp[tmp["fragmento"].str.len() > 60].drop_duplicates("fragmento").reset_index(drop=True)
        tmp["id"] = [f"idpt_treino_memoria_{i}" for i in range(len(tmp))]
        tmp["tema"] = tmp["fragmento"].map(lambda x: " ".join(tokens_chave(x, 8)))
        tmp["fonte_sugerida"] = "IDPT News — treino usado como memória textual sem rótulo"
        tmp["origem_contexto"] = "idpt_treino_sem_rotulo"
        tmp["tipo_contexto"] = "contexto_tematico_do_corpus"
        partes.append(tmp[["id", "tema", "fragmento", "fonte_sugerida", "origem_contexto", "tipo_contexto"]])

    out = pd.concat(partes, ignore_index=True)
    out["fragmento"] = out["fragmento"].fillna("").astype(str)
    return out[out["fragmento"].str.len() > 30].drop_duplicates("fragmento").reset_index(drop=True)


def construir_consulta(row: pd.Series, modo_post_hoc: bool = False) -> str:
    texto = str(row.get("text", ""))
    lead = re.sub(r"\s+", " ", texto).strip()[:900]
    campos = [
        lead,
        "entidades " + " ".join(entidades(texto, 12)),
        "palavras chave " + " ".join(tokens_chave(texto, 22)),
        "zona adtc " + str(row.get("zona_adtc", "")),
        "marcadores " + str(row.get("marcadores", "")),
        "taxonomia " + str(row.get("taxonomia_recursos", "")),
        "categorias " + str(row.get("taxonomia_categorias", "")),
        "contexto necessario " + classificar_contexto_necessario(row),
        "noticia origem evento alvo memoria discursiva",
    ]
    if modo_post_hoc:
        campos.insert(4, "tipo erro " + str(row.get("tipo_erro", "")))
    return normalizar_texto(" | ".join(campos))


@dataclass(slots=True)
class ResultadoRecuperacao:
    itens: list[dict]
    motor: str
    modelo_embedding: str | None
    modelo_reranker: str | None
    reranker_aplicado: bool


class RecuperadorHibrido:
    """TF-IDF + embeddings opcionais + cross-encoder opcional.

    O pool combina os top-N lexicais e os top-N densos. Assim, cada canal pode
    contribuir com candidatos que o outro canal não recuperaria.
    """

    def __init__(self, base: pd.DataFrame, config: ConfigV8):
        self.config = config
        self.base = base.copy().reset_index(drop=True)
        for coluna in ("id", "tema", "fragmento", "fonte_sugerida", "origem_contexto", "tipo_contexto"):
            if coluna not in self.base:
                self.base[coluna] = ""
        self.docs = (self.base["tema"].fillna("").astype(str) + " | " + self.base["fragmento"].fillna("").astype(str)).map(normalizar_texto).tolist()
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=1.0 if len(self.docs) < 2 else 0.95, sublinear_tf=True)
        self.x_lex = self.vectorizer.fit_transform(self.docs)
        self.embedder = None
        self.doc_embeddings = None
        self.reranker = None
        self.modelo_embedding: str | None = None
        self.modelo_reranker: str | None = None
        if config.usar_neural:
            self._carregar_embedder()
        if config.usar_reranker and self.embedder is not None:
            self._carregar_reranker()

    @property
    def motor(self) -> str:
        nome = "tfidf"
        if self.embedder is not None:
            nome += "+dense"
        if self.reranker is not None:
            nome += "+cross_encoder"
        return nome

    def _carregar_embedder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            warnings.warn(f"sentence-transformers indisponível: {exc!r}. Usando apenas TF-IDF.")
            return
        for nome in self.config.embedding_models:
            try:
                self.embedder = SentenceTransformer(nome)
                self.doc_embeddings = self.embedder.encode(self.docs, normalize_embeddings=True, show_progress_bar=False, batch_size=16)
                self.modelo_embedding = nome
                return
            except Exception as exc:
                warnings.warn(f"Falha ao carregar embedder {nome}: {exc!r}")
        self.embedder = None
        self.doc_embeddings = None

    def _carregar_reranker(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(self.config.reranker_model)
            self.modelo_reranker = self.config.reranker_model
        except Exception as exc:
            warnings.warn(f"Reranker indisponível: {exc!r}")
            self.reranker = None

    def recuperar(self, consulta: str, top_k: int | None = None) -> ResultadoRecuperacao:
        top_k = top_k or self.config.top_k_bruto
        q = normalizar_texto(consulta)
        if not q:
            return ResultadoRecuperacao([], self.motor, self.modelo_embedding, self.modelo_reranker, False)
        lex = cosine_similarity(self.vectorizer.transform([q]), self.x_lex).ravel()
        n = min(self.config.candidatos_por_canal, len(self.docs))
        idx_lex = np.argsort(-lex)[:n]
        dense = None
        if self.embedder is not None and self.doc_embeddings is not None:
            try:
                q_emb = self.embedder.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
                dense = np.dot(self.doc_embeddings, q_emb)
            except Exception as exc:
                warnings.warn(f"Falha no embedding da consulta: {exc!r}")
        if dense is not None:
            idx_dense = np.argsort(-dense)[:n]
            pool = np.array(sorted(set(idx_lex.tolist()) | set(idx_dense.tolist())), dtype=int)
            hibrido = self.config.peso_lexical * normalizar_score(lex[pool]) + self.config.peso_denso * normalizar_score(dense[pool])
        else:
            pool = idx_lex
            dense = np.zeros_like(lex)
            hibrido = normalizar_score(lex[pool])

        reranker_aplicado = False
        final = hibrido
        if self.reranker is not None:
            try:
                pares = [(q, self.docs[i]) for i in pool]
                final = normalizar_score(self.reranker.predict(pares))
                reranker_aplicado = True
            except Exception as exc:
                warnings.warn(f"Falha no reranker; mantendo score híbrido: {exc!r}")

        ordem = np.argsort(-final)[:top_k]
        itens: list[dict] = []
        for pos in ordem:
            i = int(pool[pos])
            base_row = self.base.iloc[i]
            itens.append(
                {
                    "score_semantico": float(final[pos]),
                    "score_lexical": float(lex[i]),
                    "score_denso": float(dense[i]),
                    "id": str(base_row["id"]),
                    "tema": str(base_row["tema"]),
                    "fragmento": str(base_row["fragmento"]),
                    "fonte_sugerida": str(base_row["fonte_sugerida"]),
                    "origem_contexto": str(base_row["origem_contexto"]),
                    "tipo_contexto": str(base_row["tipo_contexto"]),
                }
            )
        return ResultadoRecuperacao(itens, self.motor, self.modelo_embedding, self.modelo_reranker, reranker_aplicado)


def score_discursivo(row: pd.Series, candidato: dict) -> float:
    texto = row.get("text", "")
    frag = candidato.get("fragmento", "")
    tema = candidato.get("tema", "")
    origem = candidato.get("origem_contexto", "")
    tipo_ctx = str(row.get("tipo_contexto_necessario", ""))
    marcadores = str(row.get("marcadores", ""))
    zona = str(row.get("zona_adtc", ""))
    if not normalizar_texto(frag):
        return 0.0

    s_kw = jaccard(texto, f"{frag} {tema}", "tokens")
    s_ent = jaccard(texto, frag, "entidades")
    alvo = f"{fold(origem)} {fold(tema)} {fold(frag)}"
    s_ctx = 0.0
    for tipo in fold(tipo_ctx).split(";"):
        tipo = tipo.strip()
        if tipo and (tipo.replace("_", " ") in alvo or any(x in alvo for x in tipo.split("_")[:2])):
            s_ctx += 0.08
    if "memoria_discursiva_adtc" in fold(origem):
        s_ctx += 0.18
    if "idpt_treino" in fold(origem):
        s_ctx += 0.04
    if "falso gatilho" in fold(zona) and any(x in f"{fold(frag)} {fold(tema)}" for x in ("aspas", "citacao", "declaracao", "jornalismo")):
        s_ctx += 0.15
    if ("baixa marcacao" in fold(zona) or "exige contexto" in fold(zona)) and any(x in alvo for x in ("memoria", "evento", "alvo", "contexto", "noticia")):
        s_ctx += 0.15
    for marcador in ("aspas", "negacao", "adversativa", "discurso", "avaliativo", "contraste", "hiperbole"):
        if marcador in fold(marcadores) and marcador in f"{fold(frag)} {fold(tema)}":
            s_ctx += 0.04
    return round(min(1.0, 0.38 * s_kw + 0.34 * s_ent + min(0.28, s_ctx)), 4)


def reranquear(row: pd.Series, candidatos: list[dict], config: ConfigV8) -> list[dict]:
    saida: list[dict] = []
    for k, cand in enumerate(candidatos, start=1):
        sem = float(cand.get("score_semantico", 0.0))
        if not np.isfinite(sem):
            sem = 0.0
        disc = score_discursivo(row, cand)
        item = dict(cand)
        item["k_original"] = k
        item["score_discursivo"] = disc
        item["score_integrado"] = round(config.peso_semantico_integrado * sem + config.peso_discursivo_integrado * disc, 4)
        saida.append(item)
    return sorted(saida, key=lambda x: (-x["score_integrado"], -x["score_discursivo"], -x["score_semantico"]))[: config.top_k_final]


def aplicar_rag(df: pd.DataFrame, recuperador: RecuperadorHibrido, config: ConfigV8) -> pd.DataFrame:
    linhas: list[dict] = []
    for _, row in df.iterrows():
        item = row.to_dict()
        consulta = construir_consulta(row, modo_post_hoc=config.modo_post_hoc)
        bruto = recuperador.recuperar(consulta, top_k=config.top_k_bruto)
        ranking = reranquear(row, bruto.itens, config)
        item["consulta_integrada_adtc"] = consulta
        item["motor_rag_integrado"] = bruto.motor
        item["modelo_embedding_rag_integrado"] = bruto.modelo_embedding or "nenhum"
        item["modelo_reranker_rag_integrado"] = bruto.modelo_reranker or "nenhum"
        item["reranker_aplicado"] = bruto.reranker_aplicado
        item["n_candidatos_rag_integrado"] = len(bruto.itens)
        for pos in range(1, config.top_k_final + 1):
            cand = ranking[pos - 1] if pos <= len(ranking) else {}
            for chave in (
                "k_original", "score_semantico", "score_discursivo", "score_integrado", "id",
                "tema", "fragmento", "fonte_sugerida", "origem_contexto", "tipo_contexto",
            ):
                item[f"{chave}_top{pos}_rag_integrado"] = cand.get(chave, "" if chave not in {"score_semantico", "score_discursivo", "score_integrado"} else np.nan)
        linhas.append(item)
    return pd.DataFrame(linhas)
