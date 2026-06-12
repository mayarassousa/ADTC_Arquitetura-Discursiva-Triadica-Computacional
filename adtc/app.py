"""Demo da ADTC v20f para Hugging Face Spaces (Gradio).

Recebe um enunciado, recupera contexto (RAG), interroga o Tucano nos
quatro eixos com geracao deterministica e exibe a decisao auditada:
rotulo, fonte da decisao, forca da evidencia, risco de sobreajuste e
recomendacao de revisao humana.
"""

import traceback

import gradio as gr

from adtc.corpus import carregar_corpus
from adtc.pipeline import analisar_enunciado
from adtc.rag import RecuperadorContextual, carregar_base

# ZeroGPU (opcional): em Spaces com hardware ZeroGPU, o decorador abaixo
# aloca a GPU apenas durante a inferencia. Em CPU, e ignorado.
try:
    import spaces
    gpu_wrap = spaces.GPU(duration=120)
except Exception:  # fora do Spaces ou sem ZeroGPU
    gpu_wrap = lambda f: f

_TUCANO = None
_RECUPERADOR = None


def _carregar():
    global _TUCANO, _RECUPERADOR
    if _RECUPERADOR is None:
        _RECUPERADOR = RecuperadorContextual(carregar_base(), k=3)
    if _TUCANO is None:
        from adtc.modelo import TucanoAuditado
        _TUCANO = TucanoAuditado()
    return _TUCANO, _RECUPERADOR


@gpu_wrap
def analisar(enunciado):
    enunciado = (enunciado or "").strip()
    if not enunciado:
        return "Digite um enunciado.", "", ""
    try:
        tucano, recuperador = _carregar()
        reg = analisar_enunciado(enunciado, tucano, recuperador)
    except NotImplementedError as e:
        return (f"Detectores ainda nao instalados: {e}", "", "")
    except Exception:
        return (f"Erro na execucao:\n{traceback.format_exc()}", "", "")

    decisao = (
        f"Rotulo: {reg['label']}\n"
        f"Motivo: {reg['motivo']}\n"
        f"Fonte da decisao: {reg['fonte_decisao']}\n"
        f"Forca da evidencia: {reg['forca_evidencia']}\n"
        f"Risco de sobreajuste: {reg['risco_sobreajuste']}\n"
        f"Revisao humana recomendada: {'sim' if reg['revisao_humana'] else 'nao'}\n"
        f"Scores: {reg['scores']}"
    )
    eixos = "\n\n".join(
        f"[{nome.upper()}]\n{resp}" for nome, resp in reg["respostas_tucano"].items()
    )
    rag = "\n".join(
        f"({d['similaridade']:.2f}) {d['titulo']}: {d['texto'][:160]}" for d in reg["rag"]
    )
    return decisao, eixos, rag


exemplos = [c["enunciado"] for c in carregar_corpus()[:6]]

demo = gr.Interface(
    fn=analisar,
    inputs=gr.Textbox(label="Enunciado em português brasileiro", lines=2),
    outputs=[
        gr.Textbox(label="Decisão triádica auditada (ADTC v20f)", lines=8),
        gr.Textbox(label="Leituras do Tucano por eixo (testemunha auditada)", lines=12),
        gr.Textbox(label="Contexto recuperado (RAG)", lines=5),
    ],
    examples=[[e] for e in exemplos],
    title="ADTC — Arquitetura Discursiva Triádica Computacional (v20f)",
    description=(
        "Detecção auditável de ironia em português brasileiro. O Tucano-2b4-Instruct "
        "opera como analisador parcial; a decisão é tomada por uma camada discursiva "
        "explícita (travas de literalidade + cascata triádica), com registro de "
        "auditoria por item. Miguel (2026), tese de doutorado, FFLCH-USP."
    ),
)

if __name__ == "__main__":
    demo.launch()
