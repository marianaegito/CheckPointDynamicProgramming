"""Questao 2 — Parte D: experimento de escalabilidade.

Mede, para varios tamanhos de entrada:

* tempo medio de execucao (media de ``repeticoes`` execucoes);
* numero de operacoes relevantes contadas pelos proprios algoritmos;
* pico de memoria alocada (``tracemalloc``), isolando a memoria do algoritmo
  da memoria ocupada pela entrada.

    python -m src.escalabilidade
"""

from __future__ import annotations

import csv
import time
import tracemalloc
from typing import Callable, Dict, List, Sequence

try:
    from . import config
    from .brute_force import ResultadoIntervalo, forca_bruta
    from .criticidade import calcular_criticidade
    from .divide_conquer import intervalo_critico_dc
    from .gerar_dados import gerar_serie
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from brute_force import ResultadoIntervalo, forca_bruta  # type: ignore
    from criticidade import calcular_criticidade  # type: ignore
    from divide_conquer import intervalo_critico_dc  # type: ignore
    from gerar_dados import gerar_serie  # type: ignore

TAMANHOS_PADRAO = (100, 250, 500, 1000, 2000, 5000)


def _executar_dc(valores: Sequence[float]) -> ResultadoIntervalo:
    return intervalo_critico_dc(valores)[0]


ALGORITMOS: Dict[str, Callable[[Sequence[float]], ResultadoIntervalo]] = {
    "forca_bruta": forca_bruta,
    "divide_conquer": _executar_dc,
}


def medir(
    funcao: Callable[[Sequence[float]], ResultadoIntervalo],
    valores: Sequence[float],
    repeticoes: int = 3,
) -> Dict[str, float]:
    """Tempo medio, desvio grosseiro, operacoes e pico de memoria."""
    if repeticoes < 1:
        raise ValueError("repeticoes deve ser >= 1")
    tempos: List[float] = []
    resultado = None
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        resultado = funcao(valores)
        tempos.append(time.perf_counter() - t0)

    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    funcao(valores)
    pico = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    media = sum(tempos) / len(tempos)
    return {
        "tempo_medio_s": media,
        "tempo_min_s": min(tempos),
        "tempo_max_s": max(tempos),
        "operacoes": float(resultado.operacoes),   # type: ignore[union-attr]
        "memoria_pico_kb": (pico - base) / 1024,
        "valor": float(resultado.valor),           # type: ignore[union-attr]
        "inicio": float(resultado.inicio),         # type: ignore[union-attr]
        "fim": float(resultado.fim),               # type: ignore[union-attr]
    }


def experimento(
    tamanhos: Sequence[int] = TAMANHOS_PADRAO,
    repeticoes: int = 3,
    seed: int = config.SEED,
) -> List[Dict[str, object]]:
    """Roda os dois algoritmos para cada tamanho e devolve as linhas do CSV.

    Os dois algoritmos recebem **exatamente** o mesmo vetor em cada tamanho, de
    modo que a diferenca observada seja de algoritmo e nao de instancia.
    """
    linhas: List[Dict[str, object]] = []
    for n in tamanhos:
        valores = calcular_criticidade(gerar_serie(n, seed))
        # entradas pequenas sao mais sensiveis a ruido do relogio: repete mais
        reps = repeticoes if n >= 1000 else repeticoes * 5
        for nome, fn in ALGORITMOS.items():
            m = medir(fn, valores, reps)
            linhas.append({"n": n, "algoritmo": nome, **m})
            print(f"n={n:>5} {nome:<15} t={m['tempo_medio_s']:.5f}s "
                  f"ops={int(m['operacoes']):>10} mem={m['memoria_pico_kb']:.1f}KB")
    return linhas


def salvar(linhas: List[Dict[str, object]], nome: str = "escalabilidade.csv") -> str:
    caminho = config.DATA_DIR / nome
    with open(caminho, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    return str(caminho)


def razoes_empiricas(linhas: List[Dict[str, object]]) -> Dict[str, List[Dict[str, float]]]:
    """Razao de tempo entre tamanhos consecutivos x razao prevista pela teoria.

    Para ``n_{k+1}/n_k = r``, espera-se ``t`` multiplicado por ``r^2`` na forca
    bruta e por ``r·log(n_{k+1})/log(n_k)`` no divide and conquer. Comparar a
    razao observada com a prevista e a forma correta de ligar o experimento a
    analise assintotica quando os tamanhos nao dobram exatamente.
    """
    import math

    previsto = {
        "forca_bruta": lambda a, b: (b / a) ** 2,
        "divide_conquer": lambda a, b: (b / a) * (math.log2(b) / math.log2(a)),
    }
    saida: Dict[str, List[Dict[str, float]]] = {}
    for alg in ALGORITMOS:
        pontos = sorted((int(l["n"]), float(l["tempo_medio_s"]))
                        for l in linhas if l["algoritmo"] == alg)
        saida[alg] = [
            {
                "de": na, "para": nb,
                "razao_observada": round(tb / ta, 2),
                "razao_prevista": round(previsto[alg](na, nb), 2),
            }
            for (na, ta), (nb, tb) in zip(pontos, pontos[1:]) if ta > 0
        ]
    return saida


if __name__ == "__main__":  # pragma: no cover
    linhas = experimento()
    print("CSV:", salvar(linhas))
    print("razoes t(n_{k+1})/t(n_k):", razoes_empiricas(linhas))
