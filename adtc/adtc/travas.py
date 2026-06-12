"""Travas de literalidade: factual (v20e) e discursiva/fraseologica (v20f).

Funcoes transcritas da tese (Trechos 3 e 4 do capitulo de implementacao).
A logica e conservadora: na ausencia de qualquer marca de deslizamento
ironico, o estado discursivamente justificavel e a literalidade, e nao a
ironia. As travas documentam em codigo uma tese discursiva precisa:
reconhecer a literalidade nao e o estado-padrao de um modelo generativo,
mas uma decisao que precisa ser sustentada, e sustentada cedo, contra a
tendencia do modelo a ver ironia em toda parte.
"""

from adtc.detectores import (
    _norm_xai,
    texto_eh_afetivo_ou_reclamacao_literal_curta,
    texto_eh_ambiguo_curto,
    texto_eh_fraseologia_literal,
    texto_eh_indecidivel_curto,
    texto_eh_noticia_ou_explicativo_factual,
    texto_tem_contraste_forte,
    texto_tem_emoticon_contraste,
    texto_tem_memoria_critica,
)


def texto_eh_literal_factual(texto):
    """Detecta evento factual/neutro sem marcador avaliativo ou memória crítica.

    Esta função deve bloquear a leitura irônica quando só há fato negativo ou administrativo.
    """
    t = _norm_xai(texto)
    if texto_tem_contraste_forte(texto) or texto_tem_memoria_critica(texto) or texto_eh_ambiguo_curto(texto) or texto_eh_indecidivel_curto(texto):
        return False
    marcadores_factualidade = [
        "HOJE", "REUNIAO", "REMARCADA", "OBRA", "ENTREGUE", "PADRAO TECNICO",
        "ONIBUS", "ATRASOU", "INTERNET CAIU", "DURANTE A REUNIAO", "TERCA FEIRA",
        "CONTRATADO", "50 MINUTOS"
    ]
    marcadores_avaliativos = [
        "QUE OTIMO", "QUE MARAVILHA", "ADOREI", "CLARO QUE", "NADA COMO",
        "PARABENS AOS ENVOLVIDOS", "QUE LINDO", "EXCELENTE IDEIA"
    ]
    return any(m in t for m in marcadores_factualidade) and not any(m in t for m in marcadores_avaliativos)


def texto_eh_literal_discursivo(texto):
    """Trava v20f: literalidade discursiva/fraseológica.

    Retorna True quando o enunciado é convencional, jornalístico, corporativo,
    afetivo ou reclamativo sem alvo, contraste, inversão avaliativa ou memória crítica.
    """
    if texto_tem_contraste_forte(texto) or texto_tem_memoria_critica(texto) or texto_eh_ambiguo_curto(texto) or texto_eh_indecidivel_curto(texto):
        return False
    if texto_tem_emoticon_contraste(texto):
        return False
    return (
        texto_eh_noticia_ou_explicativo_factual(texto)
        or texto_eh_fraseologia_literal(texto)
        or texto_eh_afetivo_ou_reclamacao_literal_curta(texto)
    )
