"""Configuracao central do Checkpoint 4.

Concentra a semente de reprodutibilidade e os caminhos do projeto, de modo que
qualquer script/notebook produza exatamente a mesma instancia do problema.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Identificacao do grupo
# ---------------------------------------------------------------------------
GRUPO_ID = "ESPW-RESILIENCIA"
INTEGRANTES = [
    # (Nome completo, RM)
    ("Kaua Gabriel Moreira E Silva", "RM566043"),
    ("Mariana Silva do Egito Moreira", "RM562544"),
]

# Semente unica do grupo: soma dos RMs dos integrantes (566043 + 562544).
# Regra documentada e verificavel — grupos diferentes tem RMs diferentes e,
# portanto, instancias diferentes do problema. Trocar este numero muda TODA a
# instancia gerada (grafo da Questao 1 e serie temporal da Questao 2).
SEED = 1128587

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ / "data"
FIG_DIR = RAIZ / "figures"
FIG_Q1 = FIG_DIR / "questao1"
FIG_Q2 = FIG_DIR / "questao2"

for _d in (DATA_DIR, FIG_Q1, FIG_Q2):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Parametros do problema 1 (logistica de emergencia)
# ---------------------------------------------------------------------------
N_PONTOS = 20            # pontos de atendimento (alem do centro de distribuicao)
N_ARESTAS_MIN = 35       # conexoes minimas exigidas pelo enunciado
FRACAO_BLOQUEADA = 0.15  # fracao das vias declaradas indisponiveis
CAPACIDADE_VEICULO = 120 # unidades de carga transportaveis por viagem

# ---------------------------------------------------------------------------
# Parametros do problema 2 (consumo de energia)
# ---------------------------------------------------------------------------
N_OBSERVACOES = 1200     # >= 1000 exigidas
REGIOES = ("Sudeste", "Sul", "Nordeste", "Norte", "Centro-Oeste")
