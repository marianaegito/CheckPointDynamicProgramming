"""Testes da Questao 1 — grafo, Greedy e Programacao Dinamica.

Rodar da raiz do projeto:  pytest -q
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.comparacao import grafo_contraexemplo, relatorio_contraexemplo
from src.dynamic_programming import (construir_tabela, mochila_otimizada,
                                     reconstruir, selecao_dp)
from src.estruturas import INF, GrafoPonderado, Ponto
from src.gerar_dados import gerar_grafo
from src.greedy import limite_fracionario, ranking_guloso, selecao_gulosa


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def grafo() -> GrafoPonderado:
    return gerar_grafo(config.SEED)


# --------------------------------------------------------------------------- #
# Estruturas de dados
# --------------------------------------------------------------------------- #
def test_instancia_atende_ao_enunciado(grafo: GrafoPonderado) -> None:
    assert len(grafo) == config.N_PONTOS + 1          # 20 pontos + 1 centro
    assert grafo.n_arestas >= 35
    assert any(b for *_x, b in grafo.arestas())        # existem vias bloqueadas
    # o grafo NAO e completo
    v = len(grafo)
    assert grafo.n_arestas < v * (v - 1) / 2


def test_geracao_e_reprodutivel() -> None:
    a = gerar_grafo(123)
    b = gerar_grafo(123)
    c = gerar_grafo(124)
    assert sorted(a.arestas()) == sorted(b.arestas())
    assert sorted(a.arestas()) != sorted(c.arestas())


def test_entradas_invalidas_sao_rejeitadas() -> None:
    g = GrafoPonderado()
    g.add_ponto(Ponto(0, "CD", 0, 3, 0, 0.0, 0.0, 0.0, centro=True))
    g.add_ponto(Ponto(1, "A", 10, 1, 5, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        g.add_ponto(Ponto(0, "dup", 0, 3, 0, 0.0, 0.0, 0.0))
    with pytest.raises(KeyError):
        g.add_aresta(0, 99, 1.0)
    with pytest.raises(ValueError):
        g.add_aresta(0, 1, -3.0)          # Dijkstra exige peso positivo
    with pytest.raises(ValueError):
        g.add_aresta(0, 0, 1.0)           # laco
    with pytest.raises(ValueError):
        Ponto(2, "X", 10, 9, 5, 1.0, 0.0, 0.0)   # prioridade fora de 1..5


def test_dijkstra_em_grafo_conhecido() -> None:
    """Grafo de referencia calculado a mao.

        0 --4-- 1 --1-- 3
        |       |
        2       2
        |       |
        2 --5-- 3          menor caminho 0->3 = 0-2? nao: 0-1-3 = 5
    """
    g = GrafoPonderado()
    for i in range(4):
        g.add_ponto(Ponto(i, f"P{i}", 10, 1, 1, 1.0, float(i), 0.0, centro=(i == 0)))
    g.add_aresta(0, 1, 4.0)
    g.add_aresta(0, 2, 2.0)
    g.add_aresta(1, 3, 1.0)
    g.add_aresta(2, 3, 5.0)
    g.add_aresta(1, 2, 2.0)
    dist, ant = g.dijkstra(0)
    assert dist[0] == 0.0
    assert dist[2] == 2.0
    assert dist[1] == 4.0
    assert dist[3] == 5.0                       # 0-1-3
    assert g.reconstruir_caminho(ant, 3) == [0, 1, 3]


def test_dijkstra_respeita_bloqueio() -> None:
    g = GrafoPonderado()
    for i in range(3):
        g.add_ponto(Ponto(i, f"P{i}", 10, 1, 1, 1.0, float(i), 0.0, centro=(i == 0)))
    g.add_aresta(0, 1, 1.0)
    g.add_aresta(1, 2, 1.0)
    assert g.dijkstra(0)[0][2] == 2.0
    g.bloquear(1, 2)
    assert g.dijkstra(0)[0][2] == INF            # ficou inalcancavel
    g.desbloquear(1, 2)
    assert g.dijkstra(0)[0][2] == 2.0


# --------------------------------------------------------------------------- #
# Greedy
# --------------------------------------------------------------------------- #
def test_greedy_respeita_capacidade_e_e_valido(grafo: GrafoPonderado) -> None:
    for cap in (0, 1, 30, 120, 500):
        sol = selecao_gulosa(grafo, cap)
        assert sol.carga <= cap
        assert sol.valida(grafo)


def test_greedy_ordena_por_score_decrescente(grafo: GrafoPonderado) -> None:
    ranking = ranking_guloso(grafo)
    valores = [s for s, _v in ranking]
    assert valores == sorted(valores, reverse=True)
    assert len(ranking) <= len(grafo) - 1


def test_greedy_rejeita_capacidade_negativa(grafo: GrafoPonderado) -> None:
    with pytest.raises(ValueError):
        selecao_gulosa(grafo, -5)


def test_rota_comeca_no_centro(grafo: GrafoPonderado) -> None:
    sol = selecao_gulosa(grafo, 120)
    assert sol.rota[0] == 0
    assert set(sol.rota[1:]) == set(sol.selecionados)


# --------------------------------------------------------------------------- #
# Programacao Dinamica
# --------------------------------------------------------------------------- #
def test_casos_base_da_tabela() -> None:
    dp = construir_tabela([3, 4], [10.0, 12.0], 5)
    assert all(v == 0.0 for v in dp[0])           # DP[0][c] = 0
    assert all(linha[0] == 0.0 for linha in dp)   # DP[i][0] = 0


def test_recorrencia_em_exemplo_manual() -> None:
    """d = [2, 3, 4], b = [3, 4, 5], C = 5 -> otimo = itens 1 e 2 (b = 7)."""
    dp = construir_tabela([2, 3, 4], [3.0, 4.0, 5.0], 5)
    assert dp[3][5] == 7.0
    escolhidos, trilha = reconstruir(dp, [2, 3, 4], 5)
    assert escolhidos == [0, 1]
    assert trilha[0][0] == 0 and trilha[-1] == (3, 5)


def test_versao_otimizada_bate_com_a_tabela() -> None:
    rng = random.Random(7)
    for _ in range(40):
        n = rng.randint(1, 12)
        d = [rng.randint(1, 10) for _ in range(n)]
        b = [round(rng.uniform(1, 50), 2) for _ in range(n)]
        c = rng.randint(0, 30)
        assert mochila_otimizada(d, b, c) == pytest.approx(construir_tabela(d, b, c)[n][c])


def test_dp_e_igual_a_busca_exaustiva_em_instancia_pequena() -> None:
    """Prova de corretude: compara a DP com a enumeracao de TODOS os subconjuntos."""
    rng = random.Random(11)
    for _ in range(25):
        n = rng.randint(1, 10)
        d = [rng.randint(1, 8) for _ in range(n)]
        b = [round(rng.uniform(1, 30), 2) for _ in range(n)]
        cap = rng.randint(0, 20)
        melhor = 0.0
        for r in range(n + 1):
            for comb in itertools.combinations(range(n), r):
                if sum(d[i] for i in comb) <= cap:
                    melhor = max(melhor, sum(b[i] for i in comb))
        assert construir_tabela(d, b, cap)[n][cap] == pytest.approx(melhor)


def test_dp_nunca_e_pior_que_greedy(grafo: GrafoPonderado) -> None:
    for cap in range(10, 200, 15):
        g = selecao_gulosa(grafo, cap)
        d = selecao_dp(grafo, cap, guardar_tabela=False)
        assert d.beneficio >= g.beneficio - 1e-9
        assert d.valida(grafo)


def test_dp_nao_ultrapassa_limite_fracionario(grafo: GrafoPonderado) -> None:
    for cap in (40, 90, 150):
        d = selecao_dp(grafo, cap, guardar_tabela=False)
        assert d.beneficio <= limite_fracionario(grafo, cap) + 1e-6


def test_pontos_inalcancaveis_ficam_fora(grafo: GrafoPonderado) -> None:
    g = gerar_grafo(config.SEED)
    alvo = 7
    for u, v, _p, _b in g.arestas():
        if alvo in (u, v):
            g.bloquear(u, v)
    sol = selecao_dp(g, 300, guardar_tabela=False)
    assert alvo not in sol.selecionados
    assert alvo not in selecao_gulosa(g, 300).selecionados


# --------------------------------------------------------------------------- #
# Contraexemplo (Parte D)
# --------------------------------------------------------------------------- #
def test_contraexemplo_greedy_nao_e_otimo() -> None:
    rel = relatorio_contraexemplo()
    assert rel["greedy_beneficio"] == 12.0
    assert rel["dp_beneficio"] == 19.0
    assert rel["greedy_pontos"] == ["A"]
    assert sorted(rel["dp_pontos"]) == ["B", "C"]
    assert rel["perda_percentual"] > 30


def test_contraexemplo_greedy_deixa_capacidade_ociosa() -> None:
    g, cap = grafo_contraexemplo()
    guloso = selecao_gulosa(g, cap)
    otimo = selecao_dp(g, cap, guardar_tabela=False)
    assert guloso.folga == 4          # 4 unidades desperdicadas
    assert otimo.folga == 0
