"""Testes da Questao 2 — estruturas, Forca Bruta e Divide and Conquer."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.brute_force import forca_bruta, forca_bruta_cubica, top_intervalos
from src.criticidade import ParametrosCriticidade, calcular_criticidade
from src.divide_conquer import estatisticas_recursao, intervalo_critico_dc
from src.escalabilidade import medir
from src.estruturas import IndiceConsumo, Registro
from src.gerar_dados import gerar_serie


@pytest.fixture(scope="module")
def serie():
    return gerar_serie(1200, config.SEED)


@pytest.fixture(scope="module")
def criticidades(serie):
    return calcular_criticidade(serie)


# --------------------------------------------------------------------------- #
# Dados e estruturas
# --------------------------------------------------------------------------- #
def test_serie_atende_ao_enunciado(serie) -> None:
    assert len(serie) >= 1000
    campos = ("timestamp", "regiao", "consumo", "capacidade", "prioridade", "custo")
    assert all(hasattr(serie[0], c) for c in campos)


def test_geracao_reprodutivel() -> None:
    assert gerar_serie(50, 99) == gerar_serie(50, 99)
    assert gerar_serie(50, 99) != gerar_serie(50, 100)


def test_registro_e_imutavel(serie) -> None:
    with pytest.raises(Exception):
        serie[0].consumo = 0.0        # dataclass frozen


def test_indice_por_regiao_equivale_a_varredura(serie) -> None:
    idx = IndiceConsumo(serie)
    for regiao in idx.regioes:
        esperado = sum(r.consumo for r in serie if r.regiao == regiao)
        assert idx.consumo_regiao(regiao) == pytest.approx(esperado)


def test_soma_por_prefixo_equivale_a_soma_direta(serie) -> None:
    idx = IndiceConsumo(serie)
    rng = random.Random(3)
    for _ in range(50):
        i = rng.randrange(len(serie))
        j = rng.randrange(i, len(serie))
        direto = sum(r.consumo for r in serie[i:j + 1])
        assert idx.soma_intervalo(i, j) == pytest.approx(direto)


def test_top_k_com_heap_bate_com_ordenacao(serie) -> None:
    idx = IndiceConsumo(serie)
    esperado = sorted(((r.consumo, p) for p, r in enumerate(serie)), reverse=True)[:5]
    assert [v for v, _ in idx.top_k_picos(5)] == [v for v, _ in esperado]


def test_indice_rejeita_consultas_invalidas(serie) -> None:
    idx = IndiceConsumo(serie)
    with pytest.raises(KeyError):
        idx.consumo_regiao("Atlantida")
    with pytest.raises(ValueError):
        idx.consumo_hora(99)
    with pytest.raises(IndexError):
        idx.soma_intervalo(10, 5)
    with pytest.raises(ValueError):
        IndiceConsumo([])


# --------------------------------------------------------------------------- #
# Criticidade
# --------------------------------------------------------------------------- #
def test_criticidade_tem_sinais_mistos(criticidades) -> None:
    """Se todos os valores fossem positivos o problema seria trivial."""
    assert any(v > 0 for v in criticidades)
    assert any(v < 0 for v in criticidades)


def test_criticidade_cresce_com_o_consumo() -> None:
    base = dict(timestamp="2026-01-01 10:00:00", regiao="Sul",
                capacidade=1000.0, prioridade=3, custo=400.0)
    baixo = Registro(indice=0, consumo=500.0, **base)
    alto = Registro(indice=1, consumo=1500.0, **base)
    c = calcular_criticidade([baixo, alto])
    assert c[1] > c[0]
    assert c[0] < 0 < c[1]            # folga negativa, sobrecarga positiva


def test_criticidade_rejeita_capacidade_invalida() -> None:
    r = Registro(0, "2026-01-01 00:00:00", "Sul", 100.0, 0.0, 3, 10.0)
    with pytest.raises(ValueError):
        calcular_criticidade([r])
    with pytest.raises(ValueError):
        calcular_criticidade([])


def test_parametros_alteram_o_resultado(serie) -> None:
    padrao = calcular_criticidade(serie[:100])
    severo = calcular_criticidade(serie[:100], ParametrosCriticidade(theta=0.5))
    assert all(b > a for a, b in zip(padrao, severo))


# --------------------------------------------------------------------------- #
# Forca Bruta x Divide and Conquer
# --------------------------------------------------------------------------- #
def test_exemplo_manual() -> None:
    v = [-2.0, 1.0, -3.0, 4.0, -1.0, 2.0, 1.0, -5.0, 4.0]
    bf = forca_bruta(v)
    dc, _ = intervalo_critico_dc(v)
    assert (bf.inicio, bf.fim, bf.valor) == (3, 6, 6.0)
    assert (dc.inicio, dc.fim, dc.valor) == (3, 6, 6.0)


def test_um_unico_elemento() -> None:
    assert forca_bruta([5.0]).como_tupla() == (0, 0, 5.0)
    assert intervalo_critico_dc([5.0])[0].como_tupla() == (0, 0, 5.0)


def test_todos_negativos_devolve_o_maior() -> None:
    v = [-9.0, -2.0, -7.0]
    assert forca_bruta(v).como_tupla() == (1, 1, -2.0)
    assert intervalo_critico_dc(v)[0].como_tupla() == (1, 1, -2.0)


def test_vetor_vazio_e_rejeitado() -> None:
    with pytest.raises(ValueError):
        forca_bruta([])
    with pytest.raises(ValueError):
        intervalo_critico_dc([])


def test_tres_algoritmos_concordam_em_entradas_aleatorias() -> None:
    """Fuzzing: n^3, n^2 e n log n devem devolver o mesmo valor otimo."""
    rng = random.Random(2026)
    for _ in range(200):
        n = rng.randint(1, 25)
        v = [round(rng.uniform(-20, 20), 3) for _ in range(n)]
        a = forca_bruta_cubica(v)
        b = forca_bruta(v)
        c, _ = intervalo_critico_dc(v)
        assert a.valor == pytest.approx(b.valor)
        assert b.valor == pytest.approx(c.valor)


def test_concordancia_na_serie_real(criticidades) -> None:
    bf = forca_bruta(criticidades)
    dc, _ = intervalo_critico_dc(criticidades)
    assert bf.como_tupla() == pytest.approx(dc.como_tupla())


def test_divide_conquer_faz_menos_operacoes(criticidades) -> None:
    bf = forca_bruta(criticidades)
    dc, _ = intervalo_critico_dc(criticidades)
    assert dc.operacoes < bf.operacoes / 10


def test_estrutura_da_recursao(criticidades) -> None:
    """Arvore binaria completa sobre n folhas: 2n-1 chamadas, altura ceil(log2 n)."""
    est = estatisticas_recursao(criticidades)
    assert est["chamadas"] == est["chamadas_teoricas_2n_1"]
    assert est["profundidade"] == est["profundidade_teorica_log2n"]


def test_arvore_de_recursao_tem_tres_niveis(criticidades) -> None:
    _res, raiz = intervalo_critico_dc(criticidades, construir_arvore=True,
                                      max_nivel_arvore=3)
    assert raiz is not None and raiz.nivel == 0
    n1 = raiz.filhos
    n2 = [f for no in n1 for f in no.filhos]
    n3 = [f for no in n2 for f in no.filhos]
    assert len(n1) == 2 and len(n2) == 4 and len(n3) == 8
    assert raiz.lo == 0 and raiz.hi == len(criticidades) - 1
    assert n1[0].hi + 1 == n1[1].lo          # a divisao nao perde nem repete indices


def test_top_intervalos_disjuntos(criticidades) -> None:
    tops = top_intervalos(criticidades[:300], k=3)
    assert len(tops) == 3
    for a, b in zip(tops, tops[1:]):
        assert a.valor >= b.valor
    faixas = sorted((t.inicio, t.fim) for t in tops)
    for (i1, f1), (i2, _f2) in zip(faixas, faixas[1:]):
        assert f1 < i2                        # nao ha sobreposicao


def test_medicao_registra_tempo_e_memoria(criticidades) -> None:
    m = medir(forca_bruta, criticidades[:200], repeticoes=2)
    assert m["tempo_medio_s"] > 0
    assert m["operacoes"] == 200 * 201 / 2    # n(n+1)/2 intervalos avaliados
    with pytest.raises(ValueError):
        medir(forca_bruta, criticidades[:10], repeticoes=0)
