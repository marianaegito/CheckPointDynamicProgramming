"""Representacao de uma solucao de atendimento (usada por Greedy e por DP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

try:
    from .estruturas import GrafoPonderado
except ImportError:  # pragma: no cover
    from estruturas import GrafoPonderado  # type: ignore


@dataclass
class Solucao:
    """Conjunto de pontos atendidos por uma viagem do veiculo.

    Attributes
    ----------
    metodo : str
        Nome do algoritmo que produziu a solucao ("greedy" ou "dp").
    selecionados : list[int]
        Ids dos pontos atendidos.
    beneficio : float
        Soma dos beneficios dos pontos atendidos (funcao objetivo).
    carga : int
        Soma das demandas — nunca pode exceder a capacidade do veiculo.
    capacidade : int
        Capacidade usada na execucao.
    rota : list[int]
        Sequencia de visitas a partir do centro de distribuicao.
    distancia_rota : float
        Custo total da rota (soma de caminhos minimos entre visitas).
    """

    metodo: str
    selecionados: List[int]
    beneficio: float
    carga: int
    capacidade: int
    rota: List[int] = field(default_factory=list)
    distancia_rota: float = 0.0
    detalhes: Dict[str, object] = field(default_factory=dict)

    @property
    def folga(self) -> int:
        return self.capacidade - self.carga

    def valida(self, grafo: GrafoPonderado) -> bool:
        """Verifica invariantes: sem repeticao, dentro da capacidade e coerente."""
        if len(set(self.selecionados)) != len(self.selecionados):
            return False
        if any(i not in grafo.pontos for i in self.selecionados):
            return False
        carga = sum(grafo.pontos[i].demanda for i in self.selecionados)
        beneficio = sum(grafo.pontos[i].beneficio for i in self.selecionados)
        return (carga == self.carga
                and carga <= self.capacidade
                and abs(beneficio - self.beneficio) < 1e-6)

    def __str__(self) -> str:  # pragma: no cover - apresentacao
        return (f"[{self.metodo}] beneficio={self.beneficio:.2f} "
                f"carga={self.carga}/{self.capacidade} "
                f"({len(self.selecionados)} pontos) "
                f"rota={self.distancia_rota:.1f}")


def resumo_comparativo(a: Solucao, b: Solucao) -> str:  # pragma: no cover
    """Texto curto comparando duas solucoes."""
    dif = b.beneficio - a.beneficio
    rel = (dif / a.beneficio * 100) if a.beneficio else 0.0
    return (f"{a.metodo}: {a.beneficio:.2f} | {b.metodo}: {b.beneficio:.2f} "
            f"| diferenca: {dif:+.2f} ({rel:+.2f}%)")
