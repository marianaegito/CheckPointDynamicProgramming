"""Questao 2 — funcao de criticidade.

Definicao adotada pelo grupo
============================
Para cada medicao ``i`` com consumo ``x_i``, capacidade ``k_i``, prioridade
``p_i`` e custo ``$_i``, define-se a **utilizacao** ``u_i = x_i / k_i`` e

.. math::

    c_i = 100(u_i - \\theta)
        + 100\\gamma \\max(0,\\, u_i - 1)
        + w_p (p_i - \\bar{p})
        + w_c \\left(\\frac{\\$_i}{\\$_{med}} - 1\\right) \\cdot 10

com ``θ = 0.85`` (utilizacao considerada segura), ``γ = 1.5``, ``w_p = 6``,
``w_c = 4`` e ``$_med`` = mediana dos custos da propria serie.

Por que essa forma
------------------
* **Centrada em θ.** O primeiro termo e *negativo* quando a unidade opera com
  folga e *positivo* quando passa de 85% da capacidade. Isso e essencial: se
  todos os ``c_i`` fossem positivos, o "intervalo de maior criticidade
  acumulada" seria trivialmente a serie inteira e o problema perderia o
  sentido. Com valores de sinais mistos, o problema vira genuinamente
  "subsequencia contigua de soma maxima".
* **Penalidade por excesso.** ``max(0, u-1)`` so age acima de 100% da
  capacidade, quando ha corte de carga ou acionamento de termica de ponta —
  um regime qualitativamente diferente de simplesmente estar carregado.
* **Prioridade centrada em ``p̄ = 3``.** Regioes acima da prioridade media
  agravam o intervalo; abaixo, aliviam.
* **Custo relativo a mediana.** Torna o termo adimensional e robusto a
  outliers (a mediana nao e arrastada pelos picos, ao contrario da media).

Aditividade
-----------
``c_i`` depende apenas da medicao ``i``. Logo a criticidade de um intervalo e
``C(i, j) = Σ_{t=i..j} c_t`` — soma simples. Essa **aditividade** e a
propriedade que permite tanto as somas de prefixo quanto a combinacao do
Divide and Conquer.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import List, Sequence

try:
    from .estruturas import Registro
except ImportError:  # pragma: no cover
    from estruturas import Registro  # type: ignore


@dataclass(frozen=True)
class ParametrosCriticidade:
    """Pesos da funcao de criticidade (documentados para reprodutibilidade)."""

    theta: float = 0.85     # utilizacao segura
    gama: float = 1.5       # peso da penalidade por excesso
    w_prioridade: float = 6.0
    prioridade_media: float = 3.0
    w_custo: float = 4.0


PADRAO = ParametrosCriticidade()


def calcular_criticidade(
    serie: Sequence[Registro], params: ParametrosCriticidade = PADRAO
) -> List[float]:
    """Converte a serie de registros no vetor ``c`` de criticidades.

    Complexidade: ``O(n)`` de tempo (duas passadas: mediana + calculo) e
    ``O(n)`` de espaco para o vetor resultante.
    """
    if not serie:
        raise ValueError("serie vazia")
    if any(r.capacidade <= 0 for r in serie):
        raise ValueError("capacidade deve ser positiva em todos os registros")

    custo_med = median(r.custo for r in serie) or 1.0
    valores: List[float] = []
    for r in serie:
        u = r.consumo / r.capacidade
        c = (100.0 * (u - params.theta)
             + 100.0 * params.gama * max(0.0, u - 1.0)
             + params.w_prioridade * (r.prioridade - params.prioridade_media)
             + params.w_custo * (r.custo / custo_med - 1.0) * 10.0)
        valores.append(round(c, 4))
    return valores


def descrever(valores: Sequence[float]) -> str:  # pragma: no cover - apresentacao
    positivos = sum(1 for v in valores if v > 0)
    return (f"n={len(valores)} | positivos={positivos} "
            f"({positivos / len(valores) * 100:.1f}%) | "
            f"min={min(valores):.2f} | max={max(valores):.2f} | "
            f"soma={sum(valores):.2f}")
