"""Questao 1 — Parte B: estrategia gulosa de atendimento.

Funcao de prioridade
====================
Para cada ponto ``i`` alcancavel a partir do centro de distribuicao definimos

.. math::

    s_i = \\frac{\\beta_i \\cdot w(p_i)}{d_i} \\cdot
          \\frac{1}{1 + \\lambda \\cdot (\\delta_i / \\bar{\\delta})}

onde

======  ====================================================================
 β_i    beneficio esperado do atendimento do ponto i
 p_i    nivel de prioridade (1..5) e ``w(p) = p^γ`` o peso da prioridade
 d_i    demanda de recursos (unidades de carga consumidas do veiculo)
 δ_i    distancia minima real do centro ate i (Dijkstra, vias bloqueadas
        ja excluidas)
 δ̄      distancia media dos pontos alcancaveis
 λ, γ   parametros de calibracao (default λ = 0.6, γ = 1.5)
======  ====================================================================

Por que esta decisao e *localmente* vantajosa
---------------------------------------------
1. **Nucleo β/d — densidade de beneficio.** O problema de carga e um
   knapsack 0/1: maximizar ``Σ β_i x_i`` sujeito a ``Σ d_i x_i ≤ C``. Na
   *relaxacao fracionaria* desse problema (permitindo ``0 ≤ x_i ≤ 1``), o
   teorema de Dantzig garante que ordenar por ``β_i/d_i`` decrescente e
   **otimo**. Logo, a cada passo, escolher o maior ``β_i/d_i`` maximiza o
   ganho por unidade de capacidade consumida — exatamente a definicao de
   decisao localmente otima. O limite superior fracionario obtido por essa
   ordenacao e usado adiante como *bound* para medir o gap do guloso.
2. **Peso da prioridade ``w(p)=p^γ`` com γ > 1.** A prioridade e uma escala
   ordinal (1..5) de risco de vida; γ = 1.5 torna a razao entre prioridade 5 e
   prioridade 1 igual a 11.2 em vez de 5, garantindo que uma regiao critica
   pequena nao seja atropelada por uma regiao grande e pouco prioritaria.
3. **Desconto logistico ``1/(1 + λ δ_i/δ̄)``.** A capacidade nao e o unico
   recurso escasso: tempo de deslocamento tambem e. Um ponto duas vezes mais
   distante que a media vale, com λ = 0.6, cerca de 45% menos. A normalizacao
   por δ̄ deixa o fator adimensional, de modo que a formula nao muda de
   comportamento se a cidade for medida em km ou em metros.
4. **Monotonicidade.** ``s_i`` e crescente em β_i e p_i e decrescente em d_i e
   δ_i, que e a semantica desejada: atender mais gente critica, perto, com
   pouca carga.

Limitacao conhecida (explorada na Parte D): a ordenacao por densidade e otima
para a relaxacao fracionaria, mas **nao** para o problema 0/1 — a capacidade
residual pode ficar ociosa. E dai que nasce o contraexemplo do grupo.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .estruturas import INF, GrafoPonderado
    from .solucao import Solucao
except ImportError:  # pragma: no cover
    from estruturas import INF, GrafoPonderado  # type: ignore
    from solucao import Solucao  # type: ignore

LAMBDA_DIST = 0.6
GAMA_PRIORIDADE = 1.5


# --------------------------------------------------------------------------- #
def pontos_alcancaveis(grafo: GrafoPonderado, centro: int = 0) -> Dict[int, float]:
    """Distancia minima do centro a cada ponto atendivel (exclui bloqueios).

    Retorna apenas pontos com distancia finita: se todas as vias de acesso a
    uma regiao estao interditadas, ela nao pode ser atendida por terra e e
    removida do universo de decisao dos dois algoritmos.
    """
    dist, _ = grafo.dijkstra(centro)
    return {v: d for v, d in dist.items() if d < INF and v != centro}


def score(
    grafo: GrafoPonderado,
    ponto_id: int,
    distancia: float,
    distancia_media: float,
    lam: float = LAMBDA_DIST,
    gama: float = GAMA_PRIORIDADE,
) -> float:
    """Valor da funcao de prioridade ``s_i`` descrita no cabecalho do modulo."""
    p = grafo.pontos[ponto_id]
    if p.demanda <= 0:
        raise ValueError(f"ponto {ponto_id} com demanda nao positiva")
    if distancia_media <= 0:
        raise ValueError("distancia media deve ser positiva")
    densidade = (p.beneficio * (p.prioridade ** gama)) / p.demanda
    desconto = 1.0 / (1.0 + lam * (distancia / distancia_media))
    return densidade * desconto


def ranking_guloso(
    grafo: GrafoPonderado, centro: int = 0, **kwargs
) -> List[Tuple[float, int]]:
    """Pontos ordenados por ``s_i`` decrescente.

    Implementado com **heap** (``heapq.heapify`` + ``heappop``): construir o
    heap custa O(N) e cada extracao O(log N). Como o laco de selecao pode
    parar cedo — assim que a capacidade acaba —, o heap evita pagar a
    ordenacao completa O(N log N) quando so os primeiros k itens sao usados.
    """
    distancias = pontos_alcancaveis(grafo, centro)
    if not distancias:
        return []
    media = sum(distancias.values()) / len(distancias)
    heap = [(-score(grafo, v, d, media, **kwargs), v) for v, d in distancias.items()]
    heapq.heapify(heap)
    ordem: List[Tuple[float, int]] = []
    while heap:
        s_neg, v = heapq.heappop(heap)
        ordem.append((-s_neg, v))
    return ordem


# --------------------------------------------------------------------------- #
def selecao_gulosa(
    grafo: GrafoPonderado,
    capacidade: int,
    centro: int = 0,
    candidatos: Optional[Sequence[int]] = None,
    **kwargs,
) -> Solucao:
    """Escolhe pontos por ordem de ``s_i`` enquanto couber carga.

    Regra: percorre o ranking uma unica vez; um ponto e atendido se
    ``carga + d_i <= C``. Nao ha retrocesso — e justamente essa ausencia de
    revisao que caracteriza o guloso e que abre espaco para solucoes
    subotimas.

    Complexidade: ``O((V + E) log V)`` do Dijkstra + ``O(N log N)`` do ranking.
    Espaco: ``O(V + E)``.
    """
    if capacidade < 0:
        raise ValueError("capacidade nao pode ser negativa")
    ordem = ranking_guloso(grafo, centro, **kwargs)
    if candidatos is not None:
        permitidos = set(candidatos)
        ordem = [(s, v) for s, v in ordem if v in permitidos]

    escolhidos: List[int] = []
    carga = 0
    beneficio = 0.0
    descartados_por_carga: List[int] = []
    for s, v in ordem:
        p = grafo.pontos[v]
        if carga + p.demanda <= capacidade:
            escolhidos.append(v)
            carga += p.demanda
            beneficio += p.beneficio
        else:
            descartados_por_carga.append(v)

    sol = Solucao(
        metodo="greedy",
        selecionados=escolhidos,
        beneficio=round(beneficio, 4),
        carga=carga,
        capacidade=capacidade,
        detalhes={
            "ranking": ordem,
            "descartados_por_carga": descartados_por_carga,
        },
    )
    sol.rota, sol.distancia_rota = rota_por_vizinho_mais_proximo(grafo, escolhidos, centro)
    return sol


# --------------------------------------------------------------------------- #
def rota_por_vizinho_mais_proximo(
    grafo: GrafoPonderado, selecionados: Sequence[int], centro: int = 0
) -> Tuple[List[int], float]:
    """Sequencia de atendimento pelo vizinho mais proximo (heuristica gulosa).

    Definir *quais* pontos atender e um knapsack; definir *em que ordem*
    visita-los e um TSP, NP-dificil. Usamos a heuristica do vizinho mais
    proximo sobre a matriz de caminhos minimos: a cada passo o veiculo segue
    para o ponto pendente mais proximo do ponto atual. Cada distancia vem de um
    Dijkstra (custo ``O(k (V+E) log V)`` para k pontos selecionados) e nao da
    reta euclidiana — vias bloqueadas obrigam desvios reais.
    """
    if not selecionados:
        return [centro], 0.0
    pendentes = set(selecionados)
    atual = centro
    rota = [centro]
    total = 0.0
    cache: Dict[int, Dict[int, float]] = {}
    while pendentes:
        if atual not in cache:
            cache[atual] = grafo.dijkstra(atual)[0]
        dist = cache[atual]
        proximo = min(pendentes, key=lambda v: dist[v])
        if dist[proximo] == INF:      # ficou isolado apos bloqueios
            break
        total += dist[proximo]
        rota.append(proximo)
        pendentes.discard(proximo)
        atual = proximo
    return rota, round(total, 2)


# --------------------------------------------------------------------------- #
def limite_fracionario(
    grafo: GrafoPonderado, capacidade: int, candidatos: Optional[Sequence[int]] = None
) -> float:
    """Limite superior de Dantzig (knapsack fracionario) para o beneficio.

    Ordena por ``β_i/d_i`` e permite fracionar o ultimo item. Serve como
    referencia teorica: ``beneficio_DP ≤ limite_fracionario``. A distancia da
    solucao gulosa a esse limite mede a perda causada pela regra local.
    """
    ids = list(candidatos) if candidatos is not None else [
        i for i in grafo.pontos if not grafo.pontos[i].centro
    ]
    itens = sorted(
        ((grafo.pontos[i].beneficio / grafo.pontos[i].demanda, i) for i in ids),
        reverse=True,
    )
    restante = float(capacidade)
    total = 0.0
    for razao, i in itens:
        d = grafo.pontos[i].demanda
        if d <= restante:
            total += grafo.pontos[i].beneficio
            restante -= d
        else:
            total += razao * restante
            break
    return round(total, 4)
