# Checkpoint 4 — Algoritmos e Estruturas de Dados (Turma W)

**FIAP — Dynamic Programming**

## Integrantes

| Nome | RM |
|---|---|
| Kauã Gabriel Moreira E Silva | RM566043 |
| Mariana Silva do Egito Moreira | RM562544 |

**Nome do grupo:** Resiliência · **Identificador:** `ESPW-RESILIENCIA`
**Semente de reprodutibilidade:** `SEED = 1128587` (definida em `src/config.py`)

A seed é a **soma dos RMs dos integrantes** (566043 + 562544). A regra é
documentada e verificável, e garante que grupos distintos — que têm RMs
distintos — trabalhem com instâncias distintas do problema.

> Trocar `SEED` gera outra instância completa — outro grafo na Questão 1 e outra
> série temporal na Questão 2. Rodar duas vezes com a mesma semente produz
> exatamente os mesmos arquivos, figuras e números.

---

## 1. Problema

**Questão 1 — Logística de emergência.** Depois de um evento climático severo,
uma equipe de Defesa Civil precisa distribuir recursos limitados (água,
medicamentos, alimentos, kits de higiene, cobertores) a partir de um centro de
distribuição para regiões afetadas. Algumas vias estão interditadas por
alagamento ou deslizamento. O veículo tem capacidade limitada. Duas decisões
precisam ser tomadas: **quais** regiões atender e **em que ordem**.

**Questão 2 — Consumo de energia.** Dado um histórico horário de medições de
consumo por região, encontrar o intervalo contínuo de tempo com maior condição
crítica acumulada — a janela em que o sistema esteve sob maior estresse
simultâneo de carga, capacidade, prioridade e custo.

## 2. Modelo adotado

### Questão 1

* **Grafo ponderado não dirigido** `G = (V, E)`, com `V = 21` (1 centro + 20
  pontos de atendimento) e `E = 43` conexões. O peso de cada via é a distância
  euclidiana entre os pontos, o que garante desigualdade triangular. O grafo é
  **esparso** (43 de 210 arestas possíveis) e **não completo**, como exige o
  enunciado. Seis vias estão declaradas indisponíveis.
* Cada ponto carrega: pessoas afetadas, prioridade (1–5), demanda de recursos,
  benefício esperado e coordenadas.
* **Alcançabilidade** vem de um Dijkstra próprio a partir do centro, ignorando
  vias bloqueadas. Um ponto isolado sai do universo de decisão *antes* de
  qualquer cálculo de carga — ele não é um item de valor zero, é uma decisão
  indisponível.
* **Seleção de atendimentos** = mochila 0/1: maximizar `Σ βᵢxᵢ` sujeito a
  `Σ dᵢxᵢ ≤ C`.
* **Ordem de visita** = TSP, resolvido por heurística de vizinho mais próximo
  sobre a matriz de caminhos mínimos reais (com desvios impostos pelos
  bloqueios), não sobre a distância em linha reta.

### Questão 2

* Série de **1.200 medições horárias** (mínimo exigido: 1.000), geradas por um
  modelo documentado e reprodutível:

  ```
  consumo = base_regiao × sazonalidade_diaria(hora) × tendencia(t) × (1 + ruído)
            + evento_de_pico
  ```

  A sazonalidade reproduz o perfil brasileiro (vale de madrugada, ponta entre
  18h e 21h); os eventos de pico criam janelas contíguas de sobrecarga.
* **Função de criticidade** (definida pelo grupo), com `uᵢ = consumo/capacidade`:

  ```
  cᵢ = 100(uᵢ − 0,85) + 150·max(0, uᵢ − 1) + 6(pᵢ − 3) + 40(custoᵢ/custo_mediano − 1)
  ```

  A centragem em 0,85 é o ponto crucial: abaixo de 85% de utilização a parcela é
  **negativa**. Se todos os `cᵢ` fossem positivos, o intervalo de maior soma
  seria trivialmente a série inteira. Com sinais mistos (30,8% positivos nesta
  instância), o problema vira genuinamente "subsequência contígua de soma
  máxima". Como `cᵢ` depende só da medição `i`, a criticidade é **aditiva** — e é
  essa aditividade que sustenta tanto as somas de prefixo quanto a fase COMBINE
  do divide and conquer.

## 3. Estruturas de dados e por que cada uma

| Estrutura | Onde | Operação que a justifica |
|---|---|---|
| `dict[int, dict[int, float]]` | lista de adjacência do grafo | grafo esparso: `Θ(V+E)` de memória contra `Θ(V²)` da matriz; peso de aresta em `O(1)`; o laço de Dijkstra visita apenas vizinhos reais, dando `O(E)` no total em vez de `O(V²)` |
| `set[frozenset]` | vias bloqueadas | teste "esta via está interditada?" em `O(1)` médio; `frozenset` torna a chave simétrica sem duplicação, e a aresta continua na estrutura para poder ser desenhada como bloqueada |
| `dataclass(frozen=True)` | `Ponto`, `Registro` | atributos lidos milhares de vezes e nunca alterados; a imutabilidade impede que o Greedy corrompa os dados que a DP vai usar depois, e torna o objeto hashável |
| `heap` (`heapq`) | Dijkstra, ranking guloso, top-k de picos | a operação dominante é "extrair o menor": `O(log n)` no heap contra `O(n)` na lista. No top-k, um heap de tamanho fixo resolve em `O(n log k)` em vez de `O(n log n)` |
| `list[list[float]]` | tabela DP | acesso `O(1)` a `(i, c)`; a matriz inteira é retida porque a reconstrução e a Figura 3 dependem dela |
| `list` | série temporal, vetor de criticidades | preserva **ordem** e adjacência temporal — o problema do intervalo contíguo é indefinível sobre `dict` ou `set` |
| `dict` (índice invertido) | `região → posições`, `hora → posições` | consulta por região/horário em `O(1)` + leitura só dos relevantes, contra `O(n)` de varredura |
| `list` de somas de prefixo | consumo acumulado | soma de qualquer intervalo em `O(1)` após pré-processamento `Θ(n)` |
| `tuple` | resultados `(i, j, valor)` | barata de copiar, segura de compartilhar entre algoritmos |

## 4. Algoritmos

### 4.1 Greedy (Questão 1, Parte B)

Função de prioridade construída pelo grupo:

```
        βᵢ · pᵢ^1,5              1
sᵢ =  ─────────────── × ────────────────────
            dᵢ            1 + 0,6·(δᵢ / δ̄)
```

`βᵢ` = benefício, `pᵢ` = prioridade (1–5), `dᵢ` = demanda, `δᵢ` = distância
mínima real do centro (Dijkstra), `δ̄` = distância média.

**Por que a decisão é localmente vantajosa.** O núcleo `βᵢ/dᵢ` não é intuição: o
problema de carga é um knapsack 0/1 e, na sua **relaxação fracionária**, o
teorema de Dantzig garante que ordenar por `βᵢ/dᵢ` é *exatamente ótimo*. Logo,
escolher a cada passo o maior `βᵢ/dᵢ` maximiza o ganho por unidade de capacidade
consumida — a definição formal de decisão localmente ótima. O expoente 1,5 na
prioridade faz a razão entre prioridade 5 e 1 valer 11,2 em vez de 5, porque
risco de vida não é uma escala linear. O fator `1/(1 + λδᵢ/δ̄)` desconta o tempo
de deslocamento, o segundo recurso escasso; a normalização por `δ̄` deixa o fator
adimensional, de modo que a fórmula não muda de comportamento se a cidade for
medida em km ou em metros. `sᵢ` é crescente em `βᵢ` e `pᵢ` e decrescente em `dᵢ`
e `δᵢ` — exatamente a semântica desejada.

Uma regra como `maior_prioridade_primeiro()` ignora o custo de capacidade: um
ponto de prioridade 5 que consome metade do caminhão pode valer menos que três
pontos de prioridade 4 que cabem juntos.

### 4.2 Programação dinâmica (Questão 1, Parte C)

* **Estado:** `DP[i][c]` = maior benefício considerando os `i` primeiros pontos
  candidatos com `c` unidades de capacidade livres. São as duas únicas
  informações que afetam o futuro — qualquer subconjunto dos primeiros `i`
  pontos com a mesma carga é igualmente bom daí em diante (subestrutura ótima).
* **Decisão:** não atender (herda `DP[i-1][c]`) ou atender, viável só se
  `dᵢ ≤ c`, valendo `βᵢ + DP[i-1][c-dᵢ]`. Não há atendimento parcial: a demanda
  mínima torna o problema 0/1.
* **Caso-base:** `DP[0][c] = 0` e `DP[i][0] = 0`.
* **Recorrência:**
  ```
  DP[i][c] = DP[i-1][c]                                    se dᵢ > c
  DP[i][c] = max(DP[i-1][c], βᵢ + DP[i-1][c-dᵢ])           caso contrário
  ```
* **Reconstrução:** de `(N, C)` para trás — se `DP[i][c] ≠ DP[i-1][c]`, o ponto
  `i` **só pode** ter entrado (foi ele que aumentou o valor); registra-se `i` e
  segue-se para `(i-1, c-dᵢ)`. Caso contrário, para `(i-1, c)`. Exatamente `N`
  passos.

Há também uma variante de **linha única** (`mochila_otimizada`), com `Θ(C)` de
memória, que percorre a capacidade em ordem decrescente para não reutilizar o
mesmo item; ela devolve só o valor ótimo, sem o conjunto.

### 4.3 Força bruta (Questão 2, Parte B)

* `forca_bruta_cubica` — enumera todos os pares `(i, j)` e **recalcula** cada
  soma: `Θ(n³)`. Tradução literal do enunciado, mantida como referência de
  correção.
* `forca_bruta` — enumera os mesmos `n(n+1)/2` intervalos, mas estende a soma em
  `O(1)`: `Θ(n²)`. É a versão levada ao experimento de escalabilidade.

### 4.4 Divide and conquer (Questão 2, Parte C)

```
DIVIDE (m = (lo+hi)//2) → SOLVE LEFT → SOLVE RIGHT → SOLVE CROSSING → COMBINE
```

* **Caso-base:** `lo == hi` → a resposta é `c[lo]`. O intervalo vazio não é
  permitido, então numa série toda negativa a resposta é o maior elemento — o
  "período menos folgado", que é o comportamento correto para o problema real.
* **Divisão:** por posição, no meio. Metades sempre balanceadas → profundidade
  `⌈log₂ n⌉`.
* **Caso que atravessa:** um intervalo ótimo pode começar antes de `m` e
  terminar depois — nenhuma das duas recursões o enxerga. Todo intervalo cruzado
  se escreve como `[i, m] ∪ [m+1, j]`; como as duas partes são independentes,
  maximiza-se cada uma separadamente: varre-se de `m` para a esquerda guardando
  o melhor sufixo, e de `m+1` para a direita guardando o melhor prefixo. Custo
  linear no bloco.
* **Combinação:** `max` dos três candidatos; empates resolvidos pelo intervalo
  mais à esquerda, o que torna a saída determinística.

Nenhuma função pronta de busca é usada em nenhum dos dois problemas — nem
`networkx`, nem `scipy.optimize`, nem `pulp`. As bibliotecas aparecem só para
leitura de dados, gráficos, medição e testes.

## 5. Como executar

```bash
git clone <url-do-repositorio>
cd checkpoint4
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.gerar_dados          # gera data/*.csv a partir da SEED
python -m src.visualizacao_q1      # figuras da Questão 1
python -m src.visualizacao_q2      # figuras da Questão 2 + experimento de escalabilidade
python -m src.escalabilidade       # só o experimento, com o resumo no terminal
python -m src.comparacao           # contraexemplo Greedy ≠ ótimo
pytest -q                          # 40 testes
```

Notebooks (já executados, com saídas salvas):

```bash
jupyter notebook notebooks/questao1.ipynb
jupyter notebook notebooks/questao2.ipynb
```

## 6. Resultados

### Questão 1 (seed 1128587, capacidade C = 120)

| | Greedy | DP |
|---|---|---|
| Benefício total | 463,88 | **484,01** |
| Carga | 119/120 | 120/120 |
| Pontos atendidos | 6 | 5 |
| Distância percorrida | 546,8 | 352,0 |
| Gap | 4,16% abaixo do ótimo | — |

Cota superior de Dantzig (knapsack fracionário): 499,71 — o ótimo 0/1 fica
abaixo dela, como deve ser, e a diferença mede a perda causada pela
indivisibilidade do atendimento. Note que a DP fecha a carga exata (120/120)
enquanto o guloso deixa uma unidade ociosa atendendo um ponto a mais.

Varredura de 37 capacidades (20 a 200): o guloso encontrou o ótimo em **21
casos (56,8%)**, com gap médio de **2,09%** e **pior caso de 35,11%** (em
`C = 20`). O contraste é instrutivo: o guloso acerta na maioria das
capacidades, mas quando erra pode errar muito — e não há como saber de antemão
em qual dos dois regimes se está sem resolver o problema exato.

### Contraexemplo do grupo — Greedy ≠ ótimo

Três pontos com a mesma prioridade e a mesma distância do centro (os fatores de
prioridade e distância se cancelam e `sᵢ` vira exatamente `βᵢ/dᵢ`), capacidade
`C = 10`:

| ponto | demanda | benefício | densidade `β/d` |
|---|---|---|---|
| A | 6 | 12,0 | **2,00** |
| B | 5 | 9,5 | 1,90 |
| C | 5 | 9,5 | 1,90 |

* **Greedy:** escolhe A (maior densidade, consome 6). Restam 4 unidades; nem B
  nem C cabem. Resultado: **12,0**, com 4 unidades ociosas.
* **Ótimo (DP):** B + C fecha a carga exata de 10 → **19,0**.
* **Perda do guloso: 36,84%.**

**Por quê.** A densidade é o critério ótimo do knapsack *fracionário*: nele, as 4
unidades restantes seriam preenchidas com 80% de B (total 19,6). No problema 0/1
esse preenchimento é proibido, então o item de maior densidade abre um **buraco
de capacidade** que nenhuma decisão futura aproveita. Como o guloso nunca revisa
uma decisão tomada, ele não enxerga que trocar A por {B, C} seria melhor. A DP
avalia explicitamente o estado "capacidade 10 sem o item A" e encontra a troca.

### Questão 2 (seed 1128587, n = 1.200)

* Intervalo crítico: **`[1070, 1077]`** — 8 horas, de `2026-02-14 14:00` a
  `2026-02-14 21:00`, criticidade acumulada **490,42**. A janela cai exatamente
  sobre a ponta vespertina (14h–21h), o que é coerente com o perfil de consumo
  modelado.
* Força bruta e divide and conquer devolvem o mesmo intervalo, com 720.600 e
  15.950 operações respectivamente (45,2× menos).
* Estrutura da recursão medida: **2.399 chamadas** (`= 2n − 1`) e profundidade
  **11** (`= ⌈log₂ 1200⌉`) — exatamente o previsto pela teoria.

| n | Força bruta (s) | D&C (s) | ganho |
|---|---|---|---|
| 100 | 0,00019 | 0,00013 | 1,5× |
| 500 | 0,00490 | 0,00071 | 6,9× |
| 1.000 | 0,01982 | 0,00165 | 12,0× |
| 2.000 | 0,08287 | 0,00326 | 25,4× |
| 5.000 | 0,52883 | 0,00862 | 61,4× |

Razão observada × prevista (1.000 → 2.000): força bruta **4,18** contra 4,00
previsto; divide and conquer **1,98** contra 2,20 previsto. O experimento não
apenas mostra qual algoritmo é mais rápido — confirma o expoente.

## 7. Complexidade

Resumo (dedução completa em [`docs/analise_complexidade.md`](docs/analise_complexidade.md)):

| Algoritmo | Tempo | Espaço |
|---|---|---|
| Dijkstra | `O((V + E) log V)` | `Θ(V + E)` |
| Greedy (seleção) | `O((V+E) log V + N log N)` | `Θ(V + E + N)` |
| DP (mochila 0/1) | `Θ(N · C)` | `Θ(N · C)` — ou `Θ(C)` sem reconstrução |
| Reconstrução da DP | `Θ(N)` | `O(N)` |
| Rota (vizinho mais próximo) | `O(k(V+E) log V + k²)` | `Θ(V + E)` |
| Força bruta (Q2) | `Θ(n²)` | `O(1)` |
| Força bruta cúbica | `Θ(n³)` | `O(1)` |
| Divide and conquer | `Θ(n log n)` | `Θ(log n)` |

**Se o conjunto crescer de 1.000 para 1.000.000 de registros?** Extrapolando de
`n = 5.000`: a força bruta multiplica o tempo por `(10⁶/5.000)² = 40.000`,
levando `0,53 s × 40.000 ≈ 5,9 horas`. O divide and conquer multiplica por
`200 × log₂10⁶/log₂5.000 ≈ 324`, levando `≈ 2,8 segundos`. **Só o divide and
conquer continua viável.** A memória não é o gargalo em nenhum dos dois (a pilha
chega a 20 molduras); o gargalo é o número de intervalos que a força bruta
insiste em enumerar: `5 × 10¹¹` contra `≈ 2 × 10⁷` operações.

## 8. Limitações

1. **A DP é pseudopolinomial.** `Θ(N·C)` cresce com o *valor* da capacidade, não
   com seu tamanho em bits. Com `C = 120` a tabela tem 2.541 células; com
   capacidade na casa dos milhões, a abordagem exata deixaria de caber em
   memória e o guloso (ou uma DP sobre benefícios, ou um esquema FPTAS) voltaria
   a ser a escolha.
2. **A rota é heurística.** Definir *quais* pontos atender é resolvido de forma
   exata; definir *em que ordem* visitá-los é um TSP, NP-difícil, resolvido aqui
   por vizinho mais próximo. A sequência produzida não é garantidamente a mais
   curta.
3. **Uma única viagem, um único veículo.** O modelo não contempla frota,
   múltiplas viagens, janelas de tempo, recarga no centro nem validade dos
   suprimentos.
4. **Benefício e prioridade são exógenos.** Vêm do gerador de dados; num cenário
   real precisariam ser calibrados com a Defesa Civil, e a escolha de `γ = 1,5` e
   `λ = 0,6` é uma decisão de projeto, não um resultado.
5. **Vias bloqueadas são estáticas.** Na prática o bloqueio muda durante a
   operação, o que exigiria recomputar caminhos mínimos de forma incremental.
6. **A Questão 2 assume medições regulares e sem falhas.** Não há tratamento de
   lacunas, timestamps fora de ordem ou sensores defeituosos.
7. **O modelo de criticidade é linear nos seus termos.** Interações (por
   exemplo, sobrecarga prolongada sendo pior que a soma de sobrecargas isoladas)
   não são capturadas.
8. **Medições de tempo dependem da máquina.** Os números da seção 6 vêm de uma
   execução específica; as *razões* entre tamanhos são o que se deve comparar com
   a teoria, não os valores absolutos.

---

## Estrutura do repositório

```
checkpoint4/
├── README.md
├── requirements.txt
├── data/
│   ├── problema1_pontos.csv
│   ├── problema1_arestas.csv
│   ├── problema2.csv
│   └── escalabilidade.csv
├── src/
│   ├── config.py                # SEED e parâmetros
│   ├── estruturas.py            # grafo, heap, índices, registros
│   ├── gerar_dados.py           # geração reprodutível
│   ├── solucao.py               # dataclass de solução + validação
│   ├── greedy.py                # Questão 1 — Parte B
│   ├── dynamic_programming.py   # Questão 1 — Parte C
│   ├── comparacao.py            # Questão 1 — Parte D + contraexemplo
│   ├── criticidade.py           # Questão 2 — função de criticidade
│   ├── brute_force.py           # Questão 2 — Parte B
│   ├── divide_conquer.py        # Questão 2 — Parte C
│   ├── escalabilidade.py        # Questão 2 — Parte D
│   ├── visualizacao_q1.py       # Questão 1 — Parte E
│   └── visualizacao_q2.py       # Questão 2 — Parte E
├── notebooks/
│   ├── questao1.ipynb
│   └── questao2.ipynb
├── figures/
│   ├── questao1/  (fig1_grafo, fig2_solucao_greedy, fig2_solucao_dp, fig3_dp, fig4_greedy_vs_dp)
│   └── questao2/  (fig1_serie_temporal, fig2_recursao, fig3_escalabilidade, fig4_perfil_regiao)
├── tests/
│   ├── test_questao1.py
│   └── test_questao2.py
└── docs/
    └── analise_complexidade.md
```

---

## Pergunta final obrigatória

**Qual foi a decisão algorítmica mais importante tomada pelo grupo?**

Foi definir o **estado da programação dinâmica como `(pontos considerados,
capacidade livre)` e manter a tabela `(N+1)×(C+1)` inteira em memória**, em vez
de usar a variante de linha única, `Θ(C)`.

A alternativa descartada era exatamente essa versão otimizada, implementada e
testada em `mochila_otimizada`: ela devolve o mesmo valor ótimo (484,01)
percorrendo a capacidade em ordem decrescente, gastando 121 células em vez de
2.541 — 95% menos memória, com tempo idêntico, `Θ(N·C)`, já que a recorrência é
a mesma. Em benefício da solução as duas empatam.

O que decidiu a escolha foi a **reconstrução**. Sem as linhas anteriores é
impossível recuperar *quais* pontos formam o ótimo: o teste
`DP[i][c] ≠ DP[i-1][c]` precisa da linha `i-1`. E o produto deste projeto não é
o número 484,01 — é a lista de regiões a atender. Guardar a tabela custou 2.541
floats (≈ 20 KB), memória irrelevante nesta escala, e devolveu três coisas que a
versão enxuta não entrega: o conjunto atendido, a trilha de decisões usada na
Figura 3a, e a curva `DP[N][c]` da Figura 3b — que é a última linha da mesma
tabela e dá o ótimo para *toda* capacidade intermediária sem recomputar nada.

O contraexemplo confirma que a decisão valeu: greedy 12,0 contra ótimo 19,0
(perda de 36,8%), e na instância principal a DP entrega 4,3% mais benefício —
com pior caso medido de 35,11% na varredura de capacidades. O
trade-off inverteria se `C` fosse muito grande — por ser pseudopolinomial, a
tabela cresce com o *valor* da capacidade. Para `C = 10⁶` seriam 2×10⁷ células, e
aí a versão `Θ(C)`, ou o próprio guloso com seu gap médio de 2,09%, voltariam a
ser a escolha certa.
