"""Questao 1 — Parte D: Greedy x Programacao Dinamica.

Contem tambem o **contraexemplo proprio do grupo**: uma instancia pequena,
construida a mao, em que a regra gulosa comprovadamente nao e otima.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

try:
    from .dynamic_programming import selecao_dp
    from .estruturas import GrafoPonderado, Ponto
    from .greedy import limite_fracionario, selecao_gulosa
    from .solucao import Solucao
except ImportError:  # pragma: no cover
    from dynamic_programming import selecao_dp  # type: ignore
    from estruturas import GrafoPonderado, Ponto  # type: ignore
    from greedy import limite_fracionario, selecao_gulosa  # type: ignore
    from solucao import Solucao  # type: ignore


# --------------------------------------------------------------------------- #
def comparar(grafo: GrafoPonderado, capacidade: int, centro: int = 0) -> Dict[str, object]:
    """Executa os dois algoritmos sobre os mesmos dados e mede tempo/gap."""
    t0 = time.perf_counter()
    g_sol = selecao_gulosa(grafo, capacidade, centro)
    t_greedy = time.perf_counter() - t0

    t0 = time.perf_counter()
    d_sol = selecao_dp(grafo, capacidade, centro, guardar_tabela=False)
    t_dp = time.perf_counter() - t0

    gap = d_sol.beneficio - g_sol.beneficio
    return {
        "capacidade": capacidade,
        "greedy": g_sol,
        "dp": d_sol,
        "tempo_greedy_s": t_greedy,
        "tempo_dp_s": t_dp,
        "gap_absoluto": round(gap, 4),
        "gap_percentual": round(gap / d_sol.beneficio * 100, 4) if d_sol.beneficio else 0.0,
        "limite_fracionario": limite_fracionario(grafo, capacidade),
        "greedy_otimo": abs(gap) < 1e-9,
    }


def varredura_capacidades(
    grafo: GrafoPonderado, c_min: int = 10, c_max: int = 200, passo: int = 5
) -> List[Dict[str, object]]:
    """Compara Greedy x DP para varias capacidades.

    Serve para responder empiricamente *quando* o guloso acerta: capacidades
    muito pequenas (poucos itens cabem) e muito grandes (quase tudo cabe)
    tendem a favorecer o guloso; a regiao intermediaria e onde o desperdicio de
    capacidade residual aparece.
    """
    if c_min > c_max or passo <= 0:
        raise ValueError("intervalo de capacidades invalido")
    return [comparar(grafo, c) for c in range(c_min, c_max + 1, passo)]


def resumo_varredura(resultados: List[Dict[str, object]]) -> Dict[str, object]:
    """Estatisticas agregadas da varredura."""
    total = len(resultados)
    otimos = sum(1 for r in resultados if r["greedy_otimo"])
    gaps = [float(r["gap_percentual"]) for r in resultados]
    pior = max(resultados, key=lambda r: float(r["gap_percentual"]))
    return {
        "n_capacidades": total,
        "greedy_otimo_em": otimos,
        "percentual_otimo": round(otimos / total * 100, 2),
        "gap_medio_pct": round(sum(gaps) / total, 4),
        "gap_maximo_pct": round(max(gaps), 4),
        "capacidade_pior_caso": pior["capacidade"],
    }


# --------------------------------------------------------------------------- #
# CONTRAEXEMPLO DO GRUPO
# --------------------------------------------------------------------------- #
def grafo_contraexemplo() -> Tuple[GrafoPonderado, int]:
    """Instancia minima em que Greedy != otimo.

    Construcao (todos os pontos com prioridade 3 e a **mesma** distancia 10 do
    centro, de modo que o fator de prioridade e o desconto logistico se anulam
    e a funcao ``s_i`` se reduz a densidade ``β_i / d_i``)::

        ponto   demanda   beneficio   densidade
        A          6         12.0        2.00
        B          5          9.5        1.90
        C          5          9.5        1.90

    Capacidade do veiculo: **C = 10**.

    * Greedy: escolhe A (maior densidade, consome 6). Restam 4 unidades; nem B
      nem C cabem (5 > 4). Resultado: **12.0** com 4 unidades ociosas.
    * Otimo (DP): B + C = 10 unidades exatas -> **19.0**.

    O guloso perde 36.8% do beneficio. A razao e estrutural: a densidade e o
    criterio otimo do knapsack *fracionario*, onde as 4 unidades restantes
    poderiam ser preenchidas com 80% do item B. No problema 0/1 esse
    preenchimento e proibido, e o item de maior densidade produz um "buraco" de
    capacidade que nenhuma decisao futura consegue aproveitar. Como o guloso
    nunca revisa a decisao ja tomada, ele nao percebe que trocar A por {B, C}
    aumentaria o beneficio.
    """
    g = GrafoPonderado()
    g.add_ponto(Ponto(0, "CD", 0, 3, 0, 0.0, 0.0, 0.0, centro=True))
    g.add_ponto(Ponto(1, "A", 600, 3, 6, 12.0, 10.0, 0.0))
    g.add_ponto(Ponto(2, "B", 500, 3, 5, 9.5, 0.0, 10.0))
    g.add_ponto(Ponto(3, "C", 500, 3, 5, 9.5, -10.0, 0.0))
    for i in (1, 2, 3):
        g.add_aresta(0, i, 10.0)
    return g, 10


def relatorio_contraexemplo() -> Dict[str, object]:
    """Roda o contraexemplo e devolve os numeros para o README/notebook."""
    g, cap = grafo_contraexemplo()
    guloso = selecao_gulosa(g, cap)
    otimo = selecao_dp(g, cap)
    return {
        "capacidade": cap,
        "greedy_pontos": [g.pontos[i].nome for i in guloso.selecionados],
        "greedy_beneficio": guloso.beneficio,
        "greedy_carga": guloso.carga,
        "dp_pontos": [g.pontos[i].nome for i in otimo.selecionados],
        "dp_beneficio": otimo.beneficio,
        "dp_carga": otimo.carga,
        "perda_percentual": round(
            (otimo.beneficio - guloso.beneficio) / otimo.beneficio * 100, 2
        ),
        "grafo": g,
    }


if __name__ == "__main__":  # pragma: no cover
    rel = relatorio_contraexemplo()
    print("Contraexemplo:")
    print("  Greedy:", rel["greedy_pontos"], "->", rel["greedy_beneficio"],
          f"(carga {rel['greedy_carga']}/{rel['capacidade']})")
    print("  DP    :", rel["dp_pontos"], "->", rel["dp_beneficio"],
          f"(carga {rel['dp_carga']}/{rel['capacidade']})")
    print("  Perda do guloso:", rel["perda_percentual"], "%")
