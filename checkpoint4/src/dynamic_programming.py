"""Questao 1 — Parte C: Programacao Dinamica (mochila 0/1).

Formulacao completa
===================

**Estado.** ``DP[i][c]`` = maior beneficio total obtivel considerando apenas os
``i`` primeiros pontos candidatos (``0 ≤ i ≤ N``) quando o veiculo dispoe de
``c`` unidades de capacidade (``0 ≤ c ≤ C``). O estado e (quantidade de pontos
ja examinados, capacidade ainda livre) — as duas unicas informacoes que
influenciam o futuro da decisao. Nada mais do passado importa: qualquer
subconjunto dos primeiros ``i`` pontos com a mesma carga e igualmente bom para
o restante. Essa e a **subestrutura otima** que legitima a DP.

**Decisao.** Para o ponto ``i`` (1-indexado, com demanda ``d_i`` e beneficio
``β_i``) ha exatamente duas acoes:

* **nao atender** — o estado herda ``DP[i-1][c]``;
* **atender** — so e viavel se ``d_i ≤ c``, e vale ``β_i + DP[i-1][c - d_i]``.

Nao existe atendimento parcial: a demanda minima do enunciado torna o problema
0/1, e nao fracionario. E exatamente por isso que a ordenacao gulosa por
densidade deixa de ser otima.

**Caso-base.**
``DP[0][c] = 0`` para todo ``c`` (nenhum ponto considerado, nenhum beneficio) e
``DP[i][0] = 0`` para todo ``i`` (sem capacidade, nada pode ser carregado).

**Recorrencia.**

.. math::

    DP[i][c] =
    \\begin{cases}
      DP[i-1][c], & \\text{se } d_i > c \\\\[4pt]
      \\max\\big(DP[i-1][c],\\; \\beta_i + DP[i-1][c-d_i]\\big), & \\text{caso contrario}
    \\end{cases}

**Reconstrucao.** A tabela guarda valores, nao conjuntos. Partindo de
``(i, c) = (N, C)`` caminhamos para tras: se ``DP[i][c] != DP[i-1][c]``, o
ponto ``i`` **so pode** ter sido incluido (foi ele que aumentou o valor), entao
registramos ``i`` e vamos para ``(i-1, c-d_i)``; caso contrario o ponto ficou
de fora e vamos para ``(i-1, c)``. O laco executa exatamente ``N`` passos,
custando ``O(N)`` — a reconstrucao nao muda a complexidade assintotica.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .estruturas import GrafoPonderado
    from .greedy import pontos_alcancaveis, rota_por_vizinho_mais_proximo
    from .solucao import Solucao
except ImportError:  # pragma: no cover
    from estruturas import GrafoPonderado  # type: ignore
    from greedy import pontos_alcancaveis, rota_por_vizinho_mais_proximo  # type: ignore
    from solucao import Solucao  # type: ignore


# --------------------------------------------------------------------------- #
def construir_tabela(
    demandas: Sequence[int], beneficios: Sequence[float], capacidade: int
) -> List[List[float]]:
    """Preenche ``DP[(N+1) x (C+1)]`` conforme a recorrencia do modulo.

    Complexidade: tempo ``Θ(N·C)`` — dois lacos aninhados, trabalho ``O(1)`` por
    celula; espaco ``Θ(N·C)`` (a tabela inteira e mantida porque a reconstrucao
    e a Figura 3 precisam dela).
    """
    if len(demandas) != len(beneficios):
        raise ValueError("demandas e beneficios com tamanhos diferentes")
    if capacidade < 0:
        raise ValueError("capacidade negativa")
    if any(d < 0 for d in demandas):
        raise ValueError("demanda negativa")

    n = len(demandas)
    # linha 0 e coluna 0 = casos-base (zeros)
    dp = [[0.0] * (capacidade + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d_i, b_i = demandas[i - 1], beneficios[i - 1]
        linha_ant, linha = dp[i - 1], dp[i]
        for c in range(capacidade + 1):
            sem_item = linha_ant[c]
            if d_i > c:                       # nao cabe: unica opcao
                linha[c] = sem_item
            else:
                com_item = b_i + linha_ant[c - d_i]
                linha[c] = com_item if com_item > sem_item else sem_item
    return dp


def reconstruir(
    dp: Sequence[Sequence[float]], demandas: Sequence[int], capacidade: int
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Recupera os indices escolhidos e as celulas visitadas na volta.

    Retorna ``(indices_escolhidos, trilha)`` onde ``trilha`` e a lista de
    celulas ``(i, c)`` percorridas — usada na Figura 3 para desenhar o caminho
    de reconstrucao sobre o heatmap.
    """
    escolhidos: List[int] = []
    trilha: List[Tuple[int, int]] = []
    c = capacidade
    for i in range(len(demandas), 0, -1):
        trilha.append((i, c))
        if dp[i][c] != dp[i - 1][c]:      # valor mudou => item i entrou
            escolhidos.append(i - 1)
            c -= demandas[i - 1]
    trilha.append((0, c))
    escolhidos.reverse()
    trilha.reverse()
    return escolhidos, trilha


def mochila_otimizada(
    demandas: Sequence[int], beneficios: Sequence[float], capacidade: int
) -> float:
    """Mesma recorrencia com **uma unica linha** — espaco ``Θ(C)``.

    O laco de capacidade e percorrido em ordem **decrescente** para que
    ``linha[c - d_i]`` ainda contenha o valor da iteracao ``i-1`` (se fosse
    crescente, o mesmo item poderia ser usado varias vezes, virando mochila
    ilimitada). Devolve apenas o valor otimo: sem as linhas anteriores nao ha
    como reconstruir o conjunto — este e o *trade-off* memoria x rastreabilidade
    discutido no README.
    """
    linha = [0.0] * (capacidade + 1)
    for d_i, b_i in zip(demandas, beneficios):
        for c in range(capacidade, d_i - 1, -1):
            cand = b_i + linha[c - d_i]
            if cand > linha[c]:
                linha[c] = cand
    return linha[capacidade]


# --------------------------------------------------------------------------- #
def selecao_dp(
    grafo: GrafoPonderado,
    capacidade: int,
    centro: int = 0,
    candidatos: Optional[Sequence[int]] = None,
    guardar_tabela: bool = True,
) -> Solucao:
    """Resolve o atendimento otimo por Programacao Dinamica.

    Somente pontos alcancaveis entram na mochila: uma regiao cujas vias estao
    todas interditadas nao e uma decisao disponivel, e nao um item de valor
    zero.
    """
    if capacidade < 0:
        raise ValueError("capacidade nao pode ser negativa")
    alcancaveis = pontos_alcancaveis(grafo, centro)
    ids = [v for v in sorted(alcancaveis)
           if candidatos is None or v in set(candidatos)]
    if not ids:
        return Solucao("dp", [], 0.0, 0, capacidade, [centro], 0.0)

    demandas = [grafo.pontos[v].demanda for v in ids]
    beneficios = [grafo.pontos[v].beneficio for v in ids]

    dp = construir_tabela(demandas, beneficios, capacidade)
    indices, trilha = reconstruir(dp, demandas, capacidade)
    escolhidos = [ids[i] for i in indices]

    carga = sum(demandas[i] for i in indices)
    beneficio = round(sum(beneficios[i] for i in indices), 4)

    sol = Solucao(
        metodo="dp",
        selecionados=escolhidos,
        beneficio=beneficio,
        carga=carga,
        capacidade=capacidade,
        detalhes={
            "ids_candidatos": ids,
            "demandas": demandas,
            "beneficios": beneficios,
            "trilha": trilha,
            "tabela": dp if guardar_tabela else None,
        },
    )
    sol.rota, sol.distancia_rota = rota_por_vizinho_mais_proximo(grafo, escolhidos, centro)
    return sol


def curva_beneficio_por_capacidade(
    grafo: GrafoPonderado, capacidade_max: int, centro: int = 0
) -> Tuple[List[int], List[float]]:
    """``(capacidades, beneficio otimo)`` para c = 0..capacidade_max.

    Aproveita a propria tabela DP: a ultima linha ``DP[N][c]`` ja contem o
    otimo para **toda** capacidade intermediaria — nao e preciso resolver o
    problema novamente para cada c. Custo total: um unico ``Θ(N·C)``.
    """
    alcancaveis = sorted(pontos_alcancaveis(grafo, centro))
    demandas = [grafo.pontos[v].demanda for v in alcancaveis]
    beneficios = [grafo.pontos[v].beneficio for v in alcancaveis]
    dp = construir_tabela(demandas, beneficios, capacidade_max)
    return list(range(capacidade_max + 1)), dp[-1]
