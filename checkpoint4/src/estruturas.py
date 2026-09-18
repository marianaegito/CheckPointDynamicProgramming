"""Estruturas de dados do Checkpoint 4.

Este modulo concentra as estruturas usadas pelas duas questoes. Cada classe
documenta *por que* a estrutura escolhida e adequada as operacoes executadas
pelos algoritmos (e nao apenas que ela "funciona").

Questao 1
---------
- ``Ponto``            : tupla nomeada imutavel (dataclass frozen) por registro.
- ``GrafoPonderado``   : lista de adjacencia em ``dict[int, dict[int, float]]``.
- ``dijkstra``         : fila de prioridade (heap binario, ``heapq``).

Questao 2
---------
- ``Registro``         : dataclass frozen (tupla) por medicao horaria.
- ``IndiceConsumo``    : dict de indices invertidos + set + heap para top-k.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

INF = float("inf")


# ===========================================================================
# QUESTAO 1 — grafo ponderado
# ===========================================================================
@dataclass(frozen=True)
class Ponto:
    """Ponto de atendimento (ou centro de distribuicao) da rede.

    Usa-se ``dataclass(frozen=True)`` — ou seja, uma tupla nomeada imutavel —
    porque os atributos de uma regiao sao lidos milhares de vezes pelos
    algoritmos e nunca alterados. A imutabilidade evita que o Greedy corrompa
    os dados usados depois pela Programacao Dinamica e torna o objeto
    hashavel, permitindo seu uso em ``set``/``dict``.
    """

    id: int
    nome: str
    pessoas: int
    prioridade: int      # 1 (baixa) .. 5 (critica)
    demanda: int         # unidades de carga necessarias
    beneficio: float     # beneficio esperado do atendimento
    x: float             # coordenada apenas para desenho
    y: float
    centro: bool = False

    def __post_init__(self) -> None:
        if self.demanda < 0:
            raise ValueError(f"demanda negativa no ponto {self.id}")
        if not 1 <= self.prioridade <= 5:
            raise ValueError(f"prioridade fora de 1..5 no ponto {self.id}")
        if self.beneficio < 0:
            raise ValueError(f"beneficio negativo no ponto {self.id}")


class GrafoPonderado:
    """Grafo nao dirigido e ponderado em lista de adjacencia.

    Representacao: ``dict[int, dict[int, float]]``.

    Justificativa
    -------------
    * O grafo do problema e **esparso** (21 vertices, ~40 arestas). Uma matriz
      de adjacencia gastaria ``O(V^2)`` posicoes com ~90% de zeros; a lista de
      adjacencia gasta ``O(V + E)``.
    * O ``dict`` externo da acesso ``O(1)`` medio ao vertice, e o ``dict``
      interno da acesso ``O(1)`` medio ao peso de uma aresta especifica
      (``grafo.peso(u, v)``), operacao usada na validacao e no desenho.
    * Dijkstra percorre apenas os vizinhos reais de cada vertice: com lista de
      adjacencia o laco interno custa ``deg(u)``, somando ``O(E)`` no total, e
      nao ``O(V^2)`` como na matriz.
    * As vias indisponiveis ficam num ``set`` de ``frozenset({u, v})``: teste
      de bloqueio em ``O(1)`` medio, sem duplicar a aresta nos dois sentidos e
      sem remover a aresta da estrutura (queremos desenha-la como bloqueada).
    """

    def __init__(self) -> None:
        self._pontos: Dict[int, Ponto] = {}
        self._adj: Dict[int, Dict[int, float]] = {}
        self._bloqueadas: Set[frozenset] = set()

    # ------------------------------------------------------------------ #
    # Construcao
    # ------------------------------------------------------------------ #
    def add_ponto(self, ponto: Ponto) -> None:
        """Insere um ponto. O(1) medio."""
        if ponto.id in self._pontos:
            raise ValueError(f"ponto {ponto.id} duplicado")
        self._pontos[ponto.id] = ponto
        self._adj.setdefault(ponto.id, {})

    def add_aresta(self, u: int, v: int, peso: float, bloqueada: bool = False) -> None:
        """Insere uma via nao dirigida de custo ``peso``. O(1) medio."""
        if u not in self._pontos or v not in self._pontos:
            raise KeyError(f"aresta ({u}, {v}) referencia ponto inexistente")
        if u == v:
            raise ValueError("laco nao e permitido nesta rede")
        if peso <= 0:
            raise ValueError("Dijkstra exige pesos positivos")
        self._adj[u][v] = peso
        self._adj[v][u] = peso
        if bloqueada:
            self._bloqueadas.add(frozenset((u, v)))

    def bloquear(self, u: int, v: int) -> None:
        """Declara uma via indisponivel (deslizamento/alagamento)."""
        if v not in self._adj.get(u, {}):
            raise KeyError(f"via ({u}, {v}) nao existe")
        self._bloqueadas.add(frozenset((u, v)))

    def desbloquear(self, u: int, v: int) -> None:
        self._bloqueadas.discard(frozenset((u, v)))

    # ------------------------------------------------------------------ #
    # Consulta
    # ------------------------------------------------------------------ #
    @property
    def pontos(self) -> Dict[int, Ponto]:
        return self._pontos

    def __len__(self) -> int:
        return len(self._pontos)

    @property
    def n_arestas(self) -> int:
        return sum(len(viz) for viz in self._adj.values()) // 2

    def esta_bloqueada(self, u: int, v: int) -> bool:
        return frozenset((u, v)) in self._bloqueadas

    def peso(self, u: int, v: int) -> float:
        return self._adj[u][v]

    def vizinhos(self, u: int) -> Iterator[Tuple[int, float]]:
        """Itera apenas sobre vizinhos alcancaveis (ignora vias bloqueadas)."""
        for v, w in self._adj[u].items():
            if frozenset((u, v)) not in self._bloqueadas:
                yield v, w

    def arestas(self, incluir_bloqueadas: bool = True) -> List[Tuple[int, int, float, bool]]:
        """Lista ``(u, v, peso, bloqueada)`` sem repetir a aresta simetrica."""
        vistas: Set[frozenset] = set()
        saida: List[Tuple[int, int, float, bool]] = []
        for u, viz in self._adj.items():
            for v, w in viz.items():
                chave = frozenset((u, v))
                if chave in vistas:
                    continue
                vistas.add(chave)
                bloq = chave in self._bloqueadas
                if bloq and not incluir_bloqueadas:
                    continue
                saida.append((u, v, w, bloq))
        return saida

    # ------------------------------------------------------------------ #
    # Caminhos minimos
    # ------------------------------------------------------------------ #
    def dijkstra(self, origem: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        """Caminhos minimos a partir de ``origem`` (implementacao propria).

        Usa fila de prioridade (heap binario) com a tecnica de *lazy deletion*:
        em vez de diminuir a chave dentro do heap (operacao que o ``heapq`` nao
        oferece), empurra-se um novo par ``(dist, v)`` e descarta-se a entrada
        obsoleta ao desempilhar.

        Por que heap? A operacao dominante do algoritmo e "retirar o vertice
        aberto de menor distancia". Numa lista essa busca custa ``O(V)`` por
        extracao (``O(V^2)`` no total); no heap custa ``O(log V)``, levando a
        ``O((V + E) log V)``.

        Complexidade: tempo ``O((V + E) log V)``, espaco ``O(V + E)``.
        """
        if origem not in self._pontos:
            raise KeyError(f"origem {origem} inexistente")

        dist: Dict[int, float] = {v: INF for v in self._pontos}
        anterior: Dict[int, Optional[int]] = {v: None for v in self._pontos}
        dist[origem] = 0.0
        finalizados: Set[int] = set()
        heap: List[Tuple[float, int]] = [(0.0, origem)]

        while heap:
            d_u, u = heapq.heappop(heap)
            if u in finalizados:          # entrada obsoleta (lazy deletion)
                continue
            finalizados.add(u)
            for v, w in self.vizinhos(u):
                nova = d_u + w
                if nova < dist[v]:
                    dist[v] = nova
                    anterior[v] = u
                    heapq.heappush(heap, (nova, v))
        return dist, anterior

    @staticmethod
    def reconstruir_caminho(anterior: Dict[int, Optional[int]], destino: int) -> List[int]:
        """Reconstroi o caminho origem->destino andando de tras para frente."""
        caminho: List[int] = []
        atual: Optional[int] = destino
        while atual is not None:
            caminho.append(atual)
            atual = anterior[atual]
        caminho.reverse()
        return caminho


# ===========================================================================
# QUESTAO 2 — serie temporal de consumo
# ===========================================================================
@dataclass(frozen=True)
class Registro:
    """Medicao horaria de consumo.

    ``frozen=True`` transforma o registro numa tupla nomeada imutavel: os
    algoritmos de Forca Bruta e Divide-and-Conquer leem os mesmos dados e
    precisam ver exatamente a mesma serie, sob pena de a comparacao entre eles
    perder o sentido.
    """

    indice: int
    timestamp: str
    regiao: str
    consumo: float
    capacidade: float
    prioridade: int
    custo: float

    @property
    def hora(self) -> int:
        return int(self.timestamp[11:13])


class IndiceConsumo:
    """Indices auxiliares sobre a serie de medicoes.

    Estruturas e a operacao em que cada uma e vantajosa
    --------------------------------------------------
    ``list``  : ``self.serie`` mantem a **ordem temporal**. Todo o problema do
                intervalo critico depende de contiguidade e de acesso por
                indice em ``O(1)``; um ``dict`` nao preserva adjacencia
                temporal e um ``set`` destruiria a ordem.
    ``dict``  : indice invertido ``regiao -> [posicoes]`` e ``hora -> [posicoes]``.
                Consultar "consumo do Sudeste" passa de varredura ``O(n)`` para
                acesso ``O(1)`` + leitura apenas dos elementos relevantes.
    ``tuple`` : cada ``Registro`` e imutavel/hashavel (ver acima) e o resultado
                de um intervalo e devolvido como tupla ``(i, j, valor)``,
                barata de copiar e segura de compartilhar.
    ``set``   : ``self.regioes`` e ``self.horas_criticas`` respondem
                "pertence?" em ``O(1)`` medio, contra ``O(n)`` numa lista, e
                eliminam duplicatas sem ordenacao.
    ``heap``  : ``top_k_picos`` mantem um heap de tamanho k e resolve o top-k
                em ``O(n log k)``, em vez de ordenar tudo em ``O(n log n)``.
    """

    def __init__(self, serie: Sequence[Registro]) -> None:
        if not serie:
            raise ValueError("serie vazia")
        self.serie: List[Registro] = list(serie)                       # list
        self.por_regiao: Dict[str, List[int]] = {}                     # dict
        self.por_hora: Dict[int, List[int]] = {}                       # dict
        self.regioes: Set[str] = set()                                 # set
        for pos, reg in enumerate(self.serie):
            self.por_regiao.setdefault(reg.regiao, []).append(pos)
            self.por_hora.setdefault(reg.hora, []).append(pos)
            self.regioes.add(reg.regiao)

        # soma de prefixos: soma de qualquer intervalo em O(1)
        self.prefixo: List[float] = [0.0]
        for reg in self.serie:
            self.prefixo.append(self.prefixo[-1] + reg.consumo)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.serie)

    def consumo_regiao(self, regiao: str) -> float:
        """Consumo total de uma regiao. O(n_regiao) via indice invertido."""
        if regiao not in self.regioes:
            raise KeyError(f"regiao desconhecida: {regiao}")
        return sum(self.serie[p].consumo for p in self.por_regiao[regiao])

    def consumo_hora(self, hora: int) -> float:
        """Consumo total de um horario do dia. O(n_hora)."""
        if not 0 <= hora <= 23:
            raise ValueError("hora deve estar entre 0 e 23")
        return sum(self.serie[p].consumo for p in self.por_hora.get(hora, []))

    def soma_intervalo(self, i: int, j: int) -> float:
        """Soma de consumo em ``[i, j]`` em O(1) usando somas de prefixo."""
        if not 0 <= i <= j < len(self.serie):
            raise IndexError("intervalo invalido")
        return self.prefixo[j + 1] - self.prefixo[i]

    def top_k_picos(self, k: int = 5) -> List[Tuple[float, int]]:
        """k maiores consumos, em ``O(n log k)`` com heap de tamanho fixo."""
        if k <= 0:
            raise ValueError("k deve ser positivo")
        heap: List[Tuple[float, int]] = []                              # heap
        for pos, reg in enumerate(self.serie):
            if len(heap) < k:
                heapq.heappush(heap, (reg.consumo, pos))
            elif reg.consumo > heap[0][0]:
                heapq.heapreplace(heap, (reg.consumo, pos))
            # heap[0] guarda o menor dos k maiores: comparacao O(1)
        return sorted(heap, reverse=True)

    def horas_criticas(self, limiar_percentual: float = 0.9) -> Set[int]:
        """Horas cujo consumo medio ultrapassa ``limiar`` da capacidade."""
        criticas: Set[int] = set()
        for hora, posicoes in self.por_hora.items():
            for p in posicoes:
                r = self.serie[p]
                if r.consumo > limiar_percentual * r.capacidade:
                    criticas.add(hora)
                    break
        return criticas
