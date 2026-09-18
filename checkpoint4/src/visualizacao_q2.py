"""Questao 2 — Parte E: figuras obrigatorias.

    python -m src.visualizacao_q2
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

try:
    from . import config
    from .brute_force import ResultadoIntervalo
    from .divide_conquer import NoRecursao
    from .estruturas import Registro
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from brute_force import ResultadoIntervalo  # type: ignore
    from divide_conquer import NoRecursao  # type: ignore
    from estruturas import Registro  # type: ignore


# --------------------------------------------------------------------------- #
def figura_serie(
    serie: Sequence[Registro],
    criticidades: Sequence[float],
    resultado: ResultadoIntervalo,
    caminho: Optional[str] = None,
) -> str:
    """Figura 1 — consumo x tempo com o intervalo critico destacado."""
    t = [r.indice for r in serie]
    consumo = [r.consumo for r in serie]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(t, consumo, lw=0.5, color="#4c72b0", alpha=0.45,
             label="consumo por medicao (MWh)")
    # media movel de 24 medicoes: revela a tendencia sob o rodizio de regioes
    janela = 24
    if len(consumo) > janela:
        acum = [0.0]
        for c in consumo:
            acum.append(acum[-1] + c)
        media = [(acum[k + janela] - acum[k]) / janela
                 for k in range(len(consumo) - janela + 1)]
        ax1.plot(range(janela - 1, len(consumo)), media, lw=1.8, color="#1f3b73",
                 label=f"media movel ({janela} medicoes)")
    ax1.axvspan(resultado.inicio, resultado.fim, color="#d62728", alpha=0.18,
                label="intervalo critico encontrado")
    ax1.set_ylabel("consumo / capacidade")
    ax1.set_title(
        f"Figura 1 — Serie de consumo e intervalo critico | "
        f"[{resultado.inicio}, {resultado.fim}] "
        f"({serie[resultado.inicio].timestamp} -> {serie[resultado.fim].timestamp}) | "
        f"criticidade acumulada = {resultado.valor:.1f}", fontsize=11)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(alpha=0.2)

    cores = ["#d62728" if v > 0 else "#7fb3d5" for v in criticidades]
    ax2.bar(t, criticidades, width=1.0, color=cores, linewidth=0)
    ax2.axvspan(resultado.inicio, resultado.fim, color="#d62728", alpha=0.18)
    ax2.axhline(0, color="black", lw=0.7)
    ax2.set_xlabel("indice temporal (horas desde o inicio da serie)")
    ax2.set_ylabel("criticidade c_i")
    ax2.set_title("criticidade por instante — o algoritmo busca a janela contigua "
                  "de maior soma (valores negativos = folga)", fontsize=10)
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    destino = caminho or str(config.FIG_Q2 / "fig1_serie_temporal.png")
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino


# --------------------------------------------------------------------------- #
def _coletar(no: NoRecursao, nivel_max: int, por_nivel: Dict[int, List[NoRecursao]]) -> None:
    if no.nivel > nivel_max:
        return
    por_nivel.setdefault(no.nivel, []).append(no)
    for f in no.filhos:
        _coletar(f, nivel_max, por_nivel)


def figura_recursao(
    raiz: NoRecursao, resultado: ResultadoIntervalo,
    niveis: int = 3, caminho: Optional[str] = None,
) -> str:
    """Figura 2 — decomposicao do Divide and Conquer (>= 3 niveis reais).

    Cada caixa mostra o intervalo ``[lo, hi]`` realmente processado, o melhor
    valor devolvido por aquela chamada e de onde ele veio (esquerda, direita ou
    caso cruzado). A borda vermelha marca as chamadas cuja resposta coincide
    com a resposta final — evidenciando por onde a solucao "subiu" na arvore.
    """
    por_nivel: Dict[int, List[NoRecursao]] = {}
    _coletar(raiz, niveis, por_nivel)

    fig, ax = plt.subplots(figsize=(16, 8.5))
    n_niveis = max(por_nivel) + 1
    posicoes: Dict[int, float] = {}

    for nivel in sorted(por_nivel):
        nos = sorted(por_nivel[nivel], key=lambda x: x.lo)
        largura_total = 100.0
        largura = largura_total / len(nos)
        y = (n_niveis - nivel - 1) * 1.6
        for k, no in enumerate(nos):
            x = k * largura + largura / 2
            posicoes[id(no)] = x
            venceu = abs(no.melhor[2] - resultado.valor) < 1e-9
            caixa = FancyBboxPatch(
                (x - largura * 0.44, y - 0.42), largura * 0.88, 0.84,
                boxstyle="round,pad=0.02",
                facecolor="#ffe8e8" if venceu else "#eef3f8",
                edgecolor="#d62728" if venceu else "#4c72b0",
                linewidth=2.0 if venceu else 1.0, zorder=3)
            ax.add_patch(caixa)
            fonte = max(5.5, min(9.0, 60.0 / len(nos)))
            ax.text(x, y + 0.16, f"[{no.lo}, {no.hi}]  n={no.tamanho}",
                    ha="center", va="center", fontsize=fonte, fontweight="bold", zorder=4)
            ax.text(x, y - 0.16,
                    f"melhor={no.melhor[2]:.0f}\n({no.origem})",
                    ha="center", va="center", fontsize=fonte - 0.7, zorder=4)
            for f in no.filhos:
                if id(f) in posicoes:
                    ax.plot([x, posicoes[id(f)]], [y - 0.44, y - 1.6 + 0.44],
                            color="#666666", lw=1.6, zorder=1)
        ax.text(-3.5, y, f"nivel {nivel}\n({len(nos)} chamadas)",
                ha="right", va="center", fontsize=9, color="#333")

    ax.set_xlim(-16, 104)
    ax.set_ylim(-1.0, n_niveis * 1.6)
    ax.axis("off")
    ax.set_title(
        "Figura 2 — Decomposicao Divide and Conquer sobre os dados reais\n"
        f"raiz = [{raiz.lo}, {raiz.hi}] | resposta final = "
        f"[{resultado.inicio}, {resultado.fim}] com valor {resultado.valor:.1f} | "
        "borda vermelha = chamada que carrega a resposta otima", fontsize=12)
    fig.tight_layout()

    destino = caminho or str(config.FIG_Q2 / "fig2_recursao.png")
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino


# --------------------------------------------------------------------------- #
def figura_escalabilidade(
    linhas: Sequence[Dict[str, object]], caminho: Optional[str] = None
) -> str:
    """Figura 3 — tamanho da entrada x tempo, com curvas de referencia."""
    dados: Dict[str, List] = {}
    for l in linhas:
        dados.setdefault(str(l["algoritmo"]), []).append(
            (int(l["n"]), float(l["tempo_medio_s"]), float(l["operacoes"])))
    for k in dados:
        dados[k].sort()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    estilos = {"forca_bruta": ("#d62728", "o-", "Forca Bruta  O(n^2)"),
               "divide_conquer": ("#2ca02c", "s-", "Divide & Conquer  O(n log n)")}
    for alg, pontos in dados.items():
        cor, marca, rotulo = estilos.get(alg, ("#333", "^-", alg))
        ns = [p[0] for p in pontos]
        ts = [p[1] for p in pontos]
        ax1.plot(ns, ts, marca, color=cor, label=rotulo, ms=6)

    # curvas de referencia ancoradas no primeiro ponto de cada serie
    if "forca_bruta" in dados:
        n0, t0, _ = dados["forca_bruta"][0]
        ns = [p[0] for p in dados["forca_bruta"]]
        ax1.plot(ns, [t0 * (n / n0) ** 2 for n in ns], "--", color="#d62728",
                 alpha=0.45, lw=1.2, label="referencia c·n²")
    if "divide_conquer" in dados:
        n0, t0, _ = dados["divide_conquer"][0]
        ns = [p[0] for p in dados["divide_conquer"]]
        ax1.plot(ns, [t0 * (n * math.log2(n)) / (n0 * math.log2(n0)) for n in ns],
                 "--", color="#2ca02c", alpha=0.45, lw=1.2, label="referencia c·n log n")

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("tamanho da entrada n (escala log)")
    ax1.set_ylabel("tempo medio de execucao (s, escala log)")
    ax1.set_title("Figura 3a — Escalabilidade empirica\n"
                  "em log-log a inclinacao e o expoente do polinomio", fontsize=11)
    ax1.grid(alpha=0.25, which="both"); ax1.legend(fontsize=9)

    for alg, pontos in dados.items():
        cor, marca, rotulo = estilos.get(alg, ("#333", "^-", alg))
        ax2.plot([p[0] for p in pontos], [p[2] for p in pontos], marca,
                 color=cor, label=rotulo, ms=6)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("tamanho da entrada n (escala log)")
    ax2.set_ylabel("operacoes relevantes contadas")
    ax2.set_title("Figura 3b — Operacoes contadas pelo proprio algoritmo\n"
                  "n(n+1)/2 x ~n log n", fontsize=11)
    ax2.grid(alpha=0.25, which="both"); ax2.legend(fontsize=9)

    fig.tight_layout()
    destino = caminho or str(config.FIG_Q2 / "fig3_escalabilidade.png")
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino


# --------------------------------------------------------------------------- #
def figura_perfil_regiao(serie: Sequence[Registro], caminho: Optional[str] = None) -> str:
    """Figura extra — consumo medio por hora e por regiao (usa os indices dict)."""
    try:
        from .estruturas import IndiceConsumo
    except ImportError:  # pragma: no cover
        from estruturas import IndiceConsumo  # type: ignore

    idx = IndiceConsumo(serie)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for regiao in sorted(idx.regioes):
        posicoes = idx.por_regiao[regiao]
        por_hora: Dict[int, List[float]] = {}
        for p in posicoes:
            r = idx.serie[p]
            por_hora.setdefault(r.hora, []).append(r.consumo)
        horas = sorted(por_hora)
        ax.plot(horas, [sum(por_hora[h]) / len(por_hora[h]) for h in horas],
                "o-", ms=3.5, label=regiao)
    ax.set_xlabel("hora do dia")
    ax.set_ylabel("consumo medio (MWh)")
    ax.set_title("Figura extra — perfil horario por regiao "
                 "(consulta O(1) pelo indice invertido dict)", fontsize=11)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout()

    destino = caminho or str(config.FIG_Q2 / "fig4_perfil_regiao.png")
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino


# --------------------------------------------------------------------------- #
def main() -> None:  # pragma: no cover
    try:
        from .criticidade import calcular_criticidade
        from .divide_conquer import intervalo_critico_dc
        from .escalabilidade import experimento, salvar
        from .gerar_dados import carregar_serie
    except ImportError:
        from criticidade import calcular_criticidade  # type: ignore
        from divide_conquer import intervalo_critico_dc  # type: ignore
        from escalabilidade import experimento, salvar  # type: ignore
        from gerar_dados import carregar_serie  # type: ignore

    serie = carregar_serie()
    valores = calcular_criticidade(serie)
    resultado, raiz = intervalo_critico_dc(valores, construir_arvore=True, max_nivel_arvore=3)

    print(figura_serie(serie, valores, resultado))
    print(figura_recursao(raiz, resultado))
    print(figura_perfil_regiao(serie))
    linhas = experimento()
    salvar(linhas)
    print(figura_escalabilidade(linhas))


if __name__ == "__main__":
    main()
