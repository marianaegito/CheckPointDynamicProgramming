"""Questao 2 — Parte C: Divide and Conquer.

Esquema::

              DIVIDE  (meio = (lo + hi) // 2)
                 |
        +--------+--------+
        |                 |
    SOLVE LEFT        SOLVE RIGHT        -> chamadas recursivas
        |                 |
        +--------+--------+
                 |
          SOLVE CROSSING CASE            -> varredura linear a partir do meio
                 |
              COMBINE                    -> max(esquerda, direita, cruzado)

Explicacao exigida pelo enunciado
=================================
**Caso-base.** ``lo == hi``: o unico intervalo possivel e ``[lo, lo]`` e a
resposta e o proprio ``c[lo]``. Note que o intervalo vazio **nao** e permitido;
por isso, numa serie inteiramente negativa, a resposta e o elemento de maior
valor (menos negativo) — comportamento coerente com o problema real, em que
sempre existe um "periodo menos folgado".

**Divisao.** O vetor e partido ao meio pelo indice ``m = (lo + hi) // 2``,
gerando dois subproblemas de tamanho ``n/2``. A divisao e por **posicao**, nao
por valor: e isso que garante que as metades sejam sempre balanceadas e que a
recursao tenha profundidade ``⌈log2 n⌉``.

**Subproblemas.** ``resolver(lo, m)`` devolve o melhor intervalo inteiramente
contido na metade esquerda e ``resolver(m+1, hi)``, o melhor inteiramente
contido na direita.

**Caso que atravessa a divisao.** Um intervalo otimo pode comecar a esquerda de
``m`` e terminar a direita — nenhuma das duas chamadas recursivas o enxerga.
Ele e tratado a parte: qualquer intervalo cruzado se escreve como
``[i, m] ∪ [m+1, j]``. Como as duas partes sao independentes, basta maximizar
cada uma separadamente: varre-se de ``m`` para a esquerda acumulando a soma e
guardando o melhor sufixo, e de ``m+1`` para a direita guardando o melhor
prefixo. Custo ``Θ(hi - lo + 1)``, ou seja, linear no tamanho do bloco.

**Combinacao.** ``max`` dos tres candidatos (esquerdo, direito, cruzado),
comparando pelo valor da soma. Em caso de empate mantem-se o intervalo mais a
esquerda, tornando a saida deterministica.

**Recorrencia de custo.** ``T(n) = 2T(n/2) + Θ(n)``, ``T(1) = Θ(1)``. Pelo
Teorema Mestre (caso 2, ``a = 2``, ``b = 2``, ``f(n) = Θ(n) = Θ(n^{log_b a})``),
``T(n) = Θ(n log n)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

try:
    from .brute_force import ResultadoIntervalo, _validar
except ImportError:  # pragma: no cover
    from brute_force import ResultadoIntervalo, _validar  # type: ignore


@dataclass
class NoRecursao:
    """No da arvore de recursao (usado na Figura 2 da Questao 2)."""

    lo: int
    hi: int
    nivel: int
    melhor: Tuple[int, int, float]
    origem: str = ""            # "base", "esquerda", "direita" ou "cruzado"
    filhos: List["NoRecursao"] = field(default_factory=list)

    @property
    def tamanho(self) -> int:
        return self.hi - self.lo + 1


class _Contador:
    """Contador mutavel de operacoes e profundidade maxima da recursao."""

    def __init__(self) -> None:
        self.operacoes = 0
        self.chamadas = 0
        self.profundidade = 0


# --------------------------------------------------------------------------- #
def _cruzado(valores: Sequence[float], lo: int, m: int, hi: int,
             cnt: _Contador) -> Tuple[int, int, float]:
    """SOLVE CROSSING CASE — melhor intervalo que contem ``m`` e ``m+1``."""
    soma = 0.0
    melhor_esq = float("-inf")
    i_esq = m
    for i in range(m, lo - 1, -1):        # sufixo da metade esquerda
        soma += valores[i]
        cnt.operacoes += 1
        if soma > melhor_esq:
            melhor_esq, i_esq = soma, i

    soma = 0.0
    melhor_dir = float("-inf")
    j_dir = m + 1
    for j in range(m + 1, hi + 1):        # prefixo da metade direita
        soma += valores[j]
        cnt.operacoes += 1
        if soma > melhor_dir:
            melhor_dir, j_dir = soma, j

    return (i_esq, j_dir, melhor_esq + melhor_dir)


def _resolver(valores: Sequence[float], lo: int, hi: int, nivel: int,
              cnt: _Contador, arvore: Optional[List[NoRecursao]],
              max_nivel_arvore: int) -> Tuple[Tuple[int, int, float], Optional[NoRecursao]]:
    cnt.chamadas += 1
    cnt.profundidade = max(cnt.profundidade, nivel)

    if lo == hi:                                            # CASO-BASE
        melhor = (lo, lo, float(valores[lo]))
        cnt.operacoes += 1
        no = (NoRecursao(lo, hi, nivel, melhor, "base")
              if arvore is not None and nivel <= max_nivel_arvore else None)
        return melhor, no

    m = (lo + hi) // 2                                      # DIVIDE
    esq, no_esq = _resolver(valores, lo, m, nivel + 1, cnt, arvore, max_nivel_arvore)
    dir_, no_dir = _resolver(valores, m + 1, hi, nivel + 1, cnt, arvore, max_nivel_arvore)
    cru = _cruzado(valores, lo, m, hi, cnt)                 # CROSSING

    melhor, origem = esq, "esquerda"                        # COMBINE
    if dir_[2] > melhor[2]:
        melhor, origem = dir_, "direita"
    if cru[2] > melhor[2]:
        melhor, origem = cru, "cruzado"
    cnt.operacoes += 2

    no = None
    if arvore is not None and nivel <= max_nivel_arvore:
        no = NoRecursao(lo, hi, nivel, melhor, origem)
        no.filhos = [f for f in (no_esq, no_dir) if f is not None]
    return melhor, no


def intervalo_critico_dc(
    valores: Sequence[float], construir_arvore: bool = False, max_nivel_arvore: int = 3
) -> Tuple[ResultadoIntervalo, Optional[NoRecursao]]:
    """Resolve o intervalo critico por Divide and Conquer.

    Returns
    -------
    (resultado, raiz)
        ``resultado`` com indices/valor/operacoes; ``raiz`` e a arvore de
        recursao truncada em ``max_nivel_arvore`` niveis (ou ``None``).

    Complexidade
    ------------
    Tempo ``Θ(n log n)`` (recorrencia acima). Espaco ``Θ(log n)``: nao ha
    vetores auxiliares — apenas indices — e o que cresce e a pilha de chamadas,
    cuja profundidade e ``⌈log2 n⌉``.
    """
    _validar(valores)
    cnt = _Contador()
    arvore: Optional[List[NoRecursao]] = [] if construir_arvore else None
    (i, j, v), raiz = _resolver(valores, 0, len(valores) - 1, 0, cnt,
                                arvore, max_nivel_arvore)
    resultado = ResultadoIntervalo(i, j, v, cnt.operacoes, "divide_conquer")
    return resultado, raiz


def estatisticas_recursao(valores: Sequence[float]) -> dict:
    """Numero de chamadas e profundidade — confronta a teoria com a pratica."""
    _validar(valores)
    cnt = _Contador()
    _resolver(valores, 0, len(valores) - 1, 0, cnt, None, 0)
    import math
    return {
        "n": len(valores),
        "chamadas": cnt.chamadas,
        "chamadas_teoricas_2n_1": 2 * len(valores) - 1,
        "profundidade": cnt.profundidade,
        "profundidade_teorica_log2n": math.ceil(math.log2(len(valores))) if len(valores) > 1 else 0,
        "operacoes": cnt.operacoes,
    }
