"""Geracao reprodutivel das instancias do Checkpoint 4.

Todo o processo depende exclusivamente de ``config.SEED``. Rodar duas vezes com
a mesma semente produz arquivos identicos; trocar a semente produz outra
instancia do problema (exigencia de reprodutibilidade do enunciado).

Uso:
    python -m src.gerar_dados
"""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Sequence, Tuple

try:  # permite `python src/gerar_dados.py` e `python -m src.gerar_dados`
    from . import config
    from .estruturas import GrafoPonderado, Ponto, Registro
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from estruturas import GrafoPonderado, Ponto, Registro  # type: ignore


# ===========================================================================
# QUESTAO 1
# ===========================================================================
_NOMES = [
    "Vila Alta", "Jardim Rio", "Morro Verde", "Centro Sul", "Parque Norte",
    "Bairro Industrial", "Vila Ponte", "Sitio Novo", "Cohab Leste", "Beira Rio",
    "Alto da Serra", "Vila Encosta", "Distrito Oeste", "Nova Aurora", "Campo Belo",
    "Vila Lagoa", "Recanto Azul", "Bela Vista", "Porto Velho Bairro", "Chacara Sul",
]


def gerar_grafo(seed: int = config.SEED) -> GrafoPonderado:
    """Constroi a rede de emergencia (1 centro + 20 pontos, >= 35 vias).

    Passos
    ------
    1. Sorteia coordenadas — o peso da via e a distancia euclidiana, o que
       garante desigualdade triangular e torna o grafo geograficamente
       plausivel.
    2. Gera uma **arvore geradora aleatoria** ligando cada novo ponto a um ja
       inserido: assegura que o grafo bruto seja conexo.
    3. Acrescenta arestas extras entre pontos proximos ate atingir o minimo
       exigido. O grafo permanece **esparso** (longe do grafo completo, que
       teria 210 arestas).
    4. Declara ~15% das vias indisponiveis (enchentes/deslizamentos),
       preservando a conectividade de pelo menos 80% dos pontos para que o
       cenario continue interessante.
    """
    rng = random.Random(seed)
    g = GrafoPonderado()

    g.add_ponto(Ponto(0, "Centro de Distribuicao", 0, 3, 0, 0.0, 50.0, 50.0, centro=True))
    for i in range(1, config.N_PONTOS + 1):
        pessoas = rng.randint(150, 4000)
        prioridade = rng.randint(1, 5)
        # demanda cresce com a populacao, com ruido: ~1 unidade / 90 pessoas
        demanda = max(5, int(pessoas / 90) + rng.randint(-4, 8))
        # beneficio: pessoas x prioridade, com ruido de 15% (nao e proporcional
        # exato a demanda — e isso que torna o problema nao trivial)
        beneficio = round(pessoas * prioridade * rng.uniform(0.85, 1.15) / 100, 2)
        g.add_ponto(
            Ponto(
                id=i,
                nome=_NOMES[i - 1],
                pessoas=pessoas,
                prioridade=prioridade,
                demanda=demanda,
                beneficio=beneficio,
                x=round(rng.uniform(0, 100), 2),
                y=round(rng.uniform(0, 100), 2),
            )
        )

    def dist(a: int, b: int) -> float:
        pa, pb = g.pontos[a], g.pontos[b]
        return round(math.hypot(pa.x - pb.x, pa.y - pb.y), 2)

    # 2) arvore geradora aleatoria -> conectividade garantida
    ids = list(g.pontos)
    for novo in ids[1:]:
        antigo = rng.choice([i for i in ids if i < novo])
        g.add_aresta(novo, antigo, dist(novo, antigo))

    # 3) arestas extras entre pares proximos
    candidatos: List[Tuple[float, int, int]] = []
    for a in ids:
        for b in ids:
            if a < b and b not in g._adj[a]:
                candidatos.append((dist(a, b), a, b))
    candidatos.sort()
    # embaralha levemente os 60 mais curtos para nao gerar sempre a mesma malha
    curtos = candidatos[:60]
    rng.shuffle(curtos)
    for d, a, b in curtos:
        if g.n_arestas >= config.N_ARESTAS_MIN + 8:
            break
        g.add_aresta(a, b, d)

    # 4) bloqueios
    todas = g.arestas()
    n_bloq = max(1, int(len(todas) * config.FRACAO_BLOQUEADA))
    ordem = todas[:]
    rng.shuffle(ordem)
    bloqueadas = 0
    for u, v, _w, _b in ordem:
        if bloqueadas >= n_bloq:
            break
        g.bloquear(u, v)
        alcancaveis = sum(1 for d in g.dijkstra(0)[0].values() if d < float("inf"))
        if alcancaveis < int(0.8 * len(g)):
            g.desbloquear(u, v)          # bloqueio isolaria a cidade: desfaz
        else:
            bloqueadas += 1
    return g


def salvar_grafo(g: GrafoPonderado, prefixo: str = "problema1") -> Tuple[str, str]:
    """Grava pontos e arestas em CSV."""
    f_pontos = config.DATA_DIR / f"{prefixo}_pontos.csv"
    f_arestas = config.DATA_DIR / f"{prefixo}_arestas.csv"

    with open(f_pontos, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "nome", "pessoas", "prioridade", "demanda",
                    "beneficio", "x", "y", "centro"])
        for p in g.pontos.values():
            w.writerow([p.id, p.nome, p.pessoas, p.prioridade, p.demanda,
                        p.beneficio, p.x, p.y, int(p.centro)])

    with open(f_arestas, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["origem", "destino", "peso", "bloqueada"])
        for u, v, peso, bloq in sorted(g.arestas()):
            w.writerow([u, v, peso, int(bloq)])
    return str(f_pontos), str(f_arestas)


def carregar_grafo(prefixo: str = "problema1") -> GrafoPonderado:
    """Le a instancia a partir dos CSV gerados."""
    g = GrafoPonderado()
    with open(config.DATA_DIR / f"{prefixo}_pontos.csv", encoding="utf-8") as fh:
        for linha in csv.DictReader(fh):
            g.add_ponto(Ponto(
                id=int(linha["id"]), nome=linha["nome"],
                pessoas=int(linha["pessoas"]), prioridade=int(linha["prioridade"]),
                demanda=int(linha["demanda"]), beneficio=float(linha["beneficio"]),
                x=float(linha["x"]), y=float(linha["y"]),
                centro=bool(int(linha["centro"])),
            ))
    with open(config.DATA_DIR / f"{prefixo}_arestas.csv", encoding="utf-8") as fh:
        for linha in csv.DictReader(fh):
            g.add_aresta(int(linha["origem"]), int(linha["destino"]),
                         float(linha["peso"]), bool(int(linha["bloqueada"])))
    return g


# ===========================================================================
# QUESTAO 2
# ===========================================================================
_FATOR_REGIAO = {
    "Sudeste": 1.00, "Sul": 0.72, "Nordeste": 0.85,
    "Norte": 0.55, "Centro-Oeste": 0.65,
}
_CAPACIDADE_REGIAO = {
    "Sudeste": 1450.0, "Sul": 1050.0, "Nordeste": 1200.0,
    "Norte": 800.0, "Centro-Oeste": 950.0,
}


def gerar_serie(n: int = config.N_OBSERVACOES, seed: int = config.SEED) -> List[Registro]:
    """Gera ``n`` medicoes horarias sinteticas de consumo.

    Modelo de geracao (documentado para reprodutibilidade)::

        consumo = base_regiao
                  * sazonalidade_diaria(hora)
                  * tendencia_lenta(t)
                  * (1 + ruido)          , ruido ~ U(-0.08, 0.08)
                  + evento_de_pico       , ~4% dos instantes

    A ``sazonalidade_diaria`` reproduz o perfil brasileiro: vale de madrugada e
    ponta entre 18h e 21h. Os eventos de pico (onda de calor, falha de geracao)
    criam janelas contiguas de sobrecarga — sao elas que o algoritmo de
    intervalo critico precisa localizar.
    """
    if n <= 0:
        raise ValueError("n deve ser positivo")
    rng = random.Random(seed + 7)
    inicio = datetime(2026, 1, 1, 0, 0, 0)
    serie: List[Registro] = []

    # janelas de pico: comeco e duracao sorteados uma unica vez
    n_eventos = max(1, n // 250)
    eventos: List[Tuple[int, int, float]] = []
    for _ in range(n_eventos):
        ini = rng.randint(0, max(1, n - 40))
        eventos.append((ini, rng.randint(8, 30), rng.uniform(0.25, 0.55)))

    for t in range(n):
        ts = inicio + timedelta(hours=t)
        hora = ts.hour
        regiao = config.REGIOES[t % len(config.REGIOES)]
        base = 900.0 * _FATOR_REGIAO[regiao]

        # sazonalidade diaria (dois harmonicos: vale 3h-5h, ponta 18h-21h)
        saz = (1.0
               + 0.22 * math.sin((hora - 8) / 24 * 2 * math.pi)
               + 0.30 * math.exp(-((hora - 19.5) ** 2) / 8.0)
               - 0.18 * math.exp(-((hora - 4.0) ** 2) / 8.0))
        tendencia = 1.0 + 0.00008 * t          # crescimento lento da demanda
        ruido = rng.uniform(-0.08, 0.08)
        consumo = base * saz * tendencia * (1 + ruido)

        for ini, dur, amp in eventos:
            if ini <= t < ini + dur:
                consumo *= (1 + amp)

        capacidade = _CAPACIDADE_REGIAO[regiao]
        prioridade = 5 if regiao == "Sudeste" else rng.randint(1, 4)
        # custo marginal sobe quando a utilizacao passa de 1 (termica cara)
        u = consumo / capacidade
        custo = round(consumo * (0.42 + 0.55 * max(0.0, u - 1.0)), 2)

        serie.append(Registro(
            indice=t,
            timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
            regiao=regiao,
            consumo=round(consumo, 2),
            capacidade=capacidade,
            prioridade=prioridade,
            custo=custo,
        ))
    return serie


def salvar_serie(serie: Sequence[Registro], nome: str = "problema2.csv") -> str:
    caminho = config.DATA_DIR / nome
    with open(caminho, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["indice", "timestamp", "regiao", "consumo",
                    "capacidade", "prioridade", "custo"])
        for r in serie:
            w.writerow([r.indice, r.timestamp, r.regiao, r.consumo,
                        r.capacidade, r.prioridade, r.custo])
    return str(caminho)


def carregar_serie(nome: str = "problema2.csv") -> List[Registro]:
    serie: List[Registro] = []
    with open(config.DATA_DIR / nome, encoding="utf-8") as fh:
        for linha in csv.DictReader(fh):
            serie.append(Registro(
                indice=int(linha["indice"]), timestamp=linha["timestamp"],
                regiao=linha["regiao"], consumo=float(linha["consumo"]),
                capacidade=float(linha["capacidade"]),
                prioridade=int(linha["prioridade"]), custo=float(linha["custo"]),
            ))
    return serie


# ===========================================================================
def main() -> None:
    g = gerar_grafo()
    p1, p2 = salvar_grafo(g)
    dist, _ = g.dijkstra(0)
    inalcancaveis = [i for i, d in dist.items() if d == float("inf")]
    print(f"[Q1] seed={config.SEED} | {len(g)} vertices | {g.n_arestas} arestas "
          f"| {len(g._bloqueadas)} bloqueadas | inalcancaveis: {inalcancaveis}")
    print(f"     -> {p1}\n     -> {p2}")

    serie = gerar_serie()
    caminho = salvar_serie(serie)
    print(f"[Q2] {len(serie)} observacoes -> {caminho}")


if __name__ == "__main__":
    main()
