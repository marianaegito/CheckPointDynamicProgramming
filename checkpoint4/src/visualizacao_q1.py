"""Questao 1 — Parte E: figuras obrigatorias.

Todas as figuras sao geradas pelo proprio codigo, a partir da instancia gerada
pela semente do grupo. Nenhuma imagem e externa.

    python -m src.visualizacao_q1
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from . import config
    from .dynamic_programming import curva_beneficio_por_capacidade, selecao_dp
    from .estruturas import GrafoPonderado
    from .greedy import selecao_gulosa
    from .solucao import Solucao
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from dynamic_programming import curva_beneficio_por_capacidade, selecao_dp  # type: ignore
    from estruturas import GrafoPonderado  # type: ignore
    from greedy import selecao_gulosa  # type: ignore
    from solucao import Solucao  # type: ignore

CMAP_PRIORIDADE = {1: "#c7e9c0", 2: "#a1d99b", 3: "#fdd0a2", 4: "#fc9272", 5: "#de2d26"}


# --------------------------------------------------------------------------- #
def figura_grafo(grafo: GrafoPonderado, caminho: Optional[str] = None) -> str:
    """Figura 1 — rede completa: centro, pontos, pesos e vias bloqueadas."""
    fig, ax = plt.subplots(figsize=(12, 9))

    for u, v, peso, bloqueada in grafo.arestas():
        pu, pv = grafo.pontos[u], grafo.pontos[v]
        estilo = dict(color="#d62728", linestyle="--", linewidth=1.8, alpha=0.9) if bloqueada \
            else dict(color="#9aa0a6", linestyle="-", linewidth=1.1, alpha=0.8)
        ax.plot([pu.x, pv.x], [pu.y, pv.y], zorder=1, **estilo)
        mx, my = (pu.x + pv.x) / 2, (pu.y + pv.y) / 2
        ax.text(mx, my, f"{peso:.0f}", fontsize=6.5, color="#444",
                ha="center", va="center", zorder=2,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))

    for p in grafo.pontos.values():
        if p.centro:
            ax.scatter(p.x, p.y, s=520, marker="*", c="#1f77b4",
                       edgecolors="black", linewidths=1.2, zorder=4)
            ax.text(p.x, p.y - 4.5, "CD", fontsize=10, fontweight="bold", ha="center")
        else:
            ax.scatter(p.x, p.y, s=90 + p.pessoas / 25,
                       c=CMAP_PRIORIDADE[p.prioridade],
                       edgecolors="black", linewidths=0.8, zorder=3)
            ax.text(p.x, p.y + 3.0, f"{p.id}", fontsize=8, ha="center", fontweight="bold")

    legenda = [Line2D([], [], color="#9aa0a6", lw=1.5, label="via disponivel"),
               Line2D([], [], color="#d62728", lw=1.8, ls="--", label="via bloqueada"),
               Line2D([], [], marker="*", color="w", markerfacecolor="#1f77b4",
                      markeredgecolor="k", markersize=16, label="centro de distribuicao")]
    legenda += [Line2D([], [], marker="o", color="w", markerfacecolor=c,
                       markeredgecolor="k", markersize=9, label=f"prioridade {k}")
                for k, c in sorted(CMAP_PRIORIDADE.items())]
    ax.legend(handles=legenda, loc="upper left", fontsize=8, framealpha=0.95)
    ax.set_title(
        f"Figura 1 — Rede de emergencia (seed={config.SEED})\n"
        f"{len(grafo)} vertices, {grafo.n_arestas} arestas, "
        f"{sum(1 for *_ , b in grafo.arestas() if b)} vias bloqueadas | "
        "tamanho do circulo = pessoas afetadas",
        fontsize=12)
    ax.set_xlabel("coordenada x (km)")
    ax.set_ylabel("coordenada y (km)")
    ax.grid(alpha=0.15)
    fig.tight_layout()

    destino = caminho or str(config.FIG_Q1 / "fig1_grafo.png")
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino


# --------------------------------------------------------------------------- #
def _expandir_rota(grafo: GrafoPonderado, rota: Sequence[int]) -> List[int]:
    """Converte a sequencia de visitas nos vertices realmente percorridos."""
    completo: List[int] = []
    for a, b in zip(rota, rota[1:]):
        _, anterior = grafo.dijkstra(a)
        trecho = grafo.reconstruir_caminho(anterior, b)
        completo.extend(trecho if not completo else trecho[1:])
    return completo or list(rota)


def figura_solucao(
    grafo: GrafoPonderado, solucao: Solucao, caminho: Optional[str] = None
) -> str:
    """Figura 2 — atendidos, nao atendidos e sequencia de atendimento."""
    atendidos = set(solucao.selecionados)
    fig, ax = plt.subplots(figsize=(12, 9))

    for u, v, _peso, bloqueada in grafo.arestas():
        pu, pv = grafo.pontos[u], grafo.pontos[v]
        if bloqueada:
            ax.plot([pu.x, pv.x], [pu.y, pv.y], color="#d62728", ls="--", lw=1.4, alpha=0.55)
        else:
            ax.plot([pu.x, pv.x], [pu.y, pv.y], color="#d0d0d0", lw=0.9, alpha=0.8)

    # trajeto real do veiculo (caminhos minimos encadeados)
    trajeto = _expandir_rota(grafo, solucao.rota)
    xs = [grafo.pontos[v].x for v in trajeto]
    ys = [grafo.pontos[v].y for v in trajeto]
    ax.plot(xs, ys, color="#1f77b4", lw=2.4, alpha=0.85, zorder=2, label="trajeto do veiculo")
    for i, (a, b) in enumerate(zip(trajeto, trajeto[1:])):
        pa, pb = grafo.pontos[a], grafo.pontos[b]
        ax.annotate("", xy=(pb.x, pb.y), xytext=(pa.x, pa.y),
                    arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=1.2, alpha=0.8))

    for p in grafo.pontos.values():
        if p.centro:
            ax.scatter(p.x, p.y, s=520, marker="*", c="#1f77b4",
                       edgecolors="black", zorder=5)
            continue
        atendido = p.id in atendidos
        ax.scatter(p.x, p.y, s=90 + p.pessoas / 25,
                   c="#2ca02c" if atendido else "#e8e8e8",
                   edgecolors="black" if atendido else "#888",
                   linewidths=1.0, zorder=4)
        rot = f"{p.id}\nb={p.beneficio:.0f}/d={p.demanda}"
        ax.text(p.x, p.y + 3.2, rot, fontsize=6.5, ha="center",
                color="black" if atendido else "#666")

    # numeros de ordem de atendimento
    for ordem, v in enumerate(solucao.rota[1:], start=1):
        p = grafo.pontos[v]
        ax.text(p.x - 3.4, p.y - 3.4, str(ordem), fontsize=10, fontweight="bold",
                color="white", ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.22", fc="#1f77b4", ec="none"))

    legenda = [Line2D([], [], marker="o", color="w", markerfacecolor="#2ca02c",
                      markeredgecolor="k", markersize=10, label="atendido"),
               Line2D([], [], marker="o", color="w", markerfacecolor="#e8e8e8",
                      markeredgecolor="#888", markersize=10, label="nao atendido"),
               Line2D([], [], color="#1f77b4", lw=2.4, label="sequencia de atendimento"),
               Line2D([], [], color="#d62728", lw=1.4, ls="--", label="via bloqueada")]
    ax.legend(handles=legenda, loc="upper left", fontsize=9)
    ax.set_title(
        f"Figura 2 — Solucao {solucao.metodo.upper()} | beneficio={solucao.beneficio:.1f} | "
        f"carga={solucao.carga}/{solucao.capacidade} | "
        f"distancia percorrida={solucao.distancia_rota:.1f}",
        fontsize=12)
    ax.set_xlabel("coordenada x (km)")
    ax.set_ylabel("coordenada y (km)")
    ax.grid(alpha=0.15)
    fig.tight_layout()

    destino = caminho or str(config.FIG_Q1 / f"fig2_solucao_{solucao.metodo}.png")
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino


# --------------------------------------------------------------------------- #
def figura_dp(
    grafo: GrafoPonderado, solucao_dp: Solucao, caminho: Optional[str] = None
) -> str:
    """Figura 3 — heatmap da tabela DP, trilha de reconstrucao e curva β(C)."""
    tabela = solucao_dp.detalhes.get("tabela")
    if tabela is None:
        raise ValueError("solucao sem tabela: rode selecao_dp(..., guardar_tabela=True)")
    trilha: List[Tuple[int, int]] = list(solucao_dp.detalhes["trilha"])  # type: ignore
    ids: List[int] = list(solucao_dp.detalhes["ids_candidatos"])          # type: ignore
    demandas: List[int] = list(solucao_dp.detalhes["demandas"])           # type: ignore

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [1.55, 1]})

    im = ax1.imshow(tabela, aspect="auto", origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax1, label="DP[i][c] = beneficio maximo (R$ x 100)")
    tx = [c for _i, c in trilha]
    ty = [i for i, _c in trilha]
    ax1.plot(tx, ty, color="white", lw=2.0, marker="o", ms=4.5,
             markerfacecolor="#d62728", markeredgecolor="white",
             label="trilha de reconstrucao (N,C) -> (0,c)")
    for i, c in trilha[:-1]:
        if i > 0 and tabela[i][c] != tabela[i - 1][c]:
            ax1.scatter([c], [i], s=150, facecolors="none",
                        edgecolors="#ffd400", linewidths=2.2, zorder=5)
    ax1.set_xlabel("c — capacidade disponivel")
    ax1.set_ylabel("i — pontos considerados (ordem crescente de id)")
    ax1.set_title("Figura 3a — Tabela DP[i][c]\ncirculos amarelos = item incluido "
                  "(DP[i][c] != DP[i-1][c])", fontsize=11)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_yticks(range(0, len(ids) + 1, max(1, len(ids) // 10)))

    capacidades, curva = curva_beneficio_por_capacidade(grafo, solucao_dp.capacidade)
    ax2.plot(capacidades, curva, color="#1f77b4", lw=2.0, label="beneficio otimo DP[N][c]")
    ax2.fill_between(capacidades, curva, color="#1f77b4", alpha=0.12)
    saltos = [c for c in range(1, len(curva)) if curva[c] > curva[c - 1] + 1e-9]
    ax2.scatter(saltos, [curva[c] for c in saltos], s=12, color="#d62728", zorder=3,
                label="capacidades em que a solucao muda")
    ax2.axvline(solucao_dp.carga, color="#2ca02c", ls="--",
                label=f"carga usada = {solucao_dp.carga}")
    ax2.set_xlabel("capacidade C do veiculo")
    ax2.set_ylabel("beneficio total maximo")
    ax2.set_title("Figura 3b — Evolucao do beneficio otimo com a capacidade\n"
                  "(a curva e a ultima linha da mesma tabela DP)", fontsize=11)
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=8, loc="lower right")

    incluidos = [(ids[i - 1], demandas[i - 1]) for i, c in trilha[:-1]
                 if i > 0 and tabela[i][c] != tabela[i - 1][c]]
    texto = "itens incluidos (id, demanda): " + ", ".join(
        f"{v}({d})" for v, d in sorted(incluidos))
    fig.suptitle(texto, y=0.02, fontsize=9, color="#333")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    destino = caminho or str(config.FIG_Q1 / "fig3_dp.png")
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino


# --------------------------------------------------------------------------- #
def figura_comparacao(resultados: List[Dict[str, object]],
                      caminho: Optional[str] = None) -> str:
    """Figura extra — beneficio Greedy x DP em funcao da capacidade."""
    caps = [r["capacidade"] for r in resultados]
    g = [r["greedy"].beneficio for r in resultados]      # type: ignore
    d = [r["dp"].beneficio for r in resultados]          # type: ignore
    gap = [r["gap_percentual"] for r in resultados]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(caps, d, "o-", color="#1f77b4", ms=4, label="Programacao Dinamica (otimo)")
    ax1.plot(caps, g, "s--", color="#ff7f0e", ms=4, label="Greedy")
    ax1.set_ylabel("beneficio total")
    ax1.set_title("Greedy x Programacao Dinamica sobre a mesma instancia")
    ax1.legend(); ax1.grid(alpha=0.2)

    ax2.bar(caps, gap, width=3.2, color=["#2ca02c" if x < 1e-9 else "#d62728" for x in gap])
    ax2.set_xlabel("capacidade do veiculo")
    ax2.set_ylabel("gap do Greedy (%)")
    ax2.grid(alpha=0.2)
    ax2.set_title("verde = guloso otimo | vermelho = guloso subotimo", fontsize=10)
    fig.tight_layout()

    destino = caminho or str(config.FIG_Q1 / "fig4_greedy_vs_dp.png")
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino


# --------------------------------------------------------------------------- #
def main() -> None:  # pragma: no cover
    try:
        from .gerar_dados import carregar_grafo
        from .comparacao import varredura_capacidades
    except ImportError:
        from gerar_dados import carregar_grafo  # type: ignore
        from comparacao import varredura_capacidades  # type: ignore

    grafo = carregar_grafo()
    cap = config.CAPACIDADE_VEICULO
    g_sol = selecao_gulosa(grafo, cap)
    d_sol = selecao_dp(grafo, cap)

    print(figura_grafo(grafo))
    print(figura_solucao(grafo, g_sol))
    print(figura_solucao(grafo, d_sol))
    print(figura_dp(grafo, d_sol))
    print(figura_comparacao(varredura_capacidades(grafo, 20, 200, 5)))


if __name__ == "__main__":
    main()
