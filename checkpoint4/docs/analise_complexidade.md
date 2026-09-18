# Análise de complexidade — Checkpoint 4

Parâmetros usados em todo o documento:

| Símbolo | Significado | Valor nesta instância |
|---|---|---|
| `V` | vértices do grafo | 21 (1 centro + 20 pontos) |
| `E` | arestas do grafo | 43 |
| `N` | regiões candidatas (alcançáveis) | 20 |
| `C` | capacidade do veículo | 120 |
| `k` | pontos efetivamente selecionados | 4 a 6 |
| `n` | observações da série de energia | 1.200 (até 5.000 no experimento) |

---

## Questão 1

### 1.1 Dijkstra (`estruturas.GrafoPonderado.dijkstra`)

**De onde vem o custo.** A inicialização de `dist` e `anterior` percorre todos
os vértices: `Θ(V)`. Depois, cada vértice é finalizado no máximo uma vez
(`finalizados`), e ao ser finalizado percorre sua lista de adjacência. Somando
sobre todos os vértices, `Σ deg(u) = 2E`, ou seja `Θ(E)` relaxações. Cada
relaxação bem-sucedida faz um `heappush` de custo `O(log |heap|)`, e o heap tem
no máximo `E` entradas (usamos *lazy deletion*: entradas obsoletas são
descartadas ao desempilhar, não removidas do meio do heap). Cada `heappop`
custa `O(log E)` e há no máximo `E + 1` pops.

Como `E ≤ V²`, `log E ≤ 2 log V`, então:

```
T(V, E) = Θ(V) + O(E log V) + O(E log V) = O((V + E) log V)
```

**Espaço.** `S(V, E) = Θ(V + E)`: adjacência `Θ(V + E)`, `dist` e `anterior`
`Θ(V)` cada, heap `O(E)`, conjunto `finalizados` `O(V)`.

Por que não a matriz de adjacência: com `V = 21` e `E = 43`, a matriz teria 441
posições, das quais 355 seriam zeros; e o laço de relaxação passaria a custar
`Θ(V)` por vértice, levando o algoritmo a `Θ(V²)`. Para grafos esparsos
(`E = O(V)`), a lista de adjacência com heap é assintoticamente melhor.

### 1.2 Greedy (`greedy.selecao_gulosa`)

| Etapa | Custo |
|---|---|
| Dijkstra a partir do centro | `O((V + E) log V)` |
| Cálculo de `s_i` para cada candidato | `Θ(N)` |
| `heapify` do ranking | `Θ(N)` |
| `N` extrações do heap | `O(N log N)` |
| Varredura única de seleção | `Θ(N)` |
| Rota por vizinho mais próximo (`k` Dijkstras) | `O(k (V + E) log V + k²)` |

```
T = O((V + E) log V + N log N + k (V + E) log V)
S = Θ(V + E + N)
```

O termo `N log N` domina o `Θ(N)` da avaliação da função de prioridade; o termo
de rota domina o resto quando `k` cresce. Não há tabela auxiliar proporcional à
capacidade — essa é a vantagem estrutural do guloso.

### 1.3 Programação dinâmica (`dynamic_programming.construir_tabela`)

Dois laços aninhados: `i` de `1` a `N`, `c` de `0` a `C`. Cada célula faz uma
comparação e no máximo uma soma, `O(1)`. Portanto:

```
T(N, C) = Θ(N · C)      -> 20 × 121 = 2.420 células nesta instância
S(N, C) = Θ(N · C)      -> tabela (N+1) × (C+1) = 2.541 células
```

Reconstrução: `Θ(N)` — exatamente um passo por linha, de `(N, C)` até
`(0, c)`.

Versão de linha única (`mochila_otimizada`): mesmo `Θ(N · C)` de tempo com
`Θ(C)` de espaço, ao custo de perder a reconstrução do conjunto.

**Observação importante.** `Θ(N · C)` é **pseudopolinomial**: o custo cresce com
o *valor* de `C`, não com o número de bits usados para escrevê-lo. Dobrar a
capacidade dobra o tempo; acrescentar um dígito a `C` multiplica o tempo por
dez. O experimento no notebook confirma: dobrar `C` de 120 para 240, 480 e 960
dobra o tempo a cada passo, com custo por célula praticamente constante.

Verificação empírica (notebook `questao1.ipynb`):

| N | C | N·C | tempo |
|---|---|---|---|
| 10 | 120 | 1.200 | ~0,1 ms |
| 20 | 120 | 2.400 | ~0,2 ms |
| 20 | 240 | 4.800 | ~0,4 ms |
| 20 | 480 | 9.600 | ~0,8 ms |
| 20 | 960 | 19.200 | ~1,6 ms |

### 1.4 Comparação

| | Greedy | DP |
|---|---|---|
| Tempo | `O((V+E) log V + N log N)` | `Θ(N · C)` + `O((V+E) log V)` |
| Espaço | `Θ(V + E + N)` | `Θ(N · C)` (ou `Θ(C)` sem reconstrução) |
| Garantia | nenhuma | ótimo global |
| Resultado (C = 120) | 463,88 | **484,01** (+4,3%) |
| Pior gap medido (C de 20 a 200) | 35,11% | — |
| Gap médio | 2,09% | — |
| Vezes em que foi ótimo | 21 de 37 capacidades (56,8%) | 37 de 37 |

Cota superior teórica (limite fracionário de Dantzig) para `C = 120`: **499,71**.
O ótimo 0/1 (484,01) fica abaixo dela, como esperado, e a distância entre as
duas mede exatamente a perda causada pela indivisibilidade do atendimento.

---

## Questão 2

### 2.1 Força bruta — `brute_force.forca_bruta`

O laço externo roda `n` vezes; para cada `i`, o laço interno roda `n - i` vezes.

```
Σ_{i=0}^{n-1} (n - i) = n + (n-1) + ... + 1 = n(n+1)/2 ∈ Θ(n²)
```

Cada iteração faz uma soma e uma comparação, `O(1)`, porque a soma é estendida
incrementalmente (`soma += valores[j]`). O contador interno do algoritmo devolve
exatamente `n(n+1)/2` — para `n = 1200`, 720.600 operações, confirmado pelo
teste `test_medicao_registra_tempo_e_memoria`.

```
T(n) = Θ(n²)
S(n) = O(1)   — cinco escalares além da entrada; sem recursão, sem vetores auxiliares
```

A versão `forca_bruta_cubica` recalcula cada soma do zero, acrescentando um
terceiro laço: `T(n) = Θ(n³)`, `S(n) = O(1)`. É mantida apenas como referência
de correção para `n` pequeno.

### 2.2 Divide and conquer — `divide_conquer.intervalo_critico_dc`

**Recorrência.** Uma chamada sobre `n` elementos faz duas chamadas sobre `n/2`
mais uma varredura linear (caso cruzado: um laço de `m` até `lo` e outro de
`m+1` até `hi`, somando `n` iterações):

```
T(n) = 2 T(n/2) + Θ(n),    T(1) = Θ(1)
```

**Teorema Mestre.** `a = 2`, `b = 2`, `f(n) = Θ(n)`. Como
`n^{log_b a} = n^{log_2 2} = n = Θ(f(n))`, aplica-se o caso 2:

```
T(n) = Θ(n log n)
```

**Pela árvore de recursão.** Há `⌈log₂ n⌉ + 1` níveis; em cada nível, a soma dos
tamanhos dos blocos é `n`, e o caso cruzado percorre cada bloco linearmente.
Logo cada nível custa `Θ(n)` e o total é `Θ(n log n)`. O número de chamadas é
`2n - 1` (árvore binária cheia com `n` folhas) e a profundidade máxima é
`⌈log₂ n⌉`. Ambos são verificados em tempo de teste por
`test_estrutura_da_recursao`: para `n = 1200`, 2.399 chamadas e profundidade 11
(`⌈log₂ 1200⌉ = 11`).

**Espaço.** Não há vetores auxiliares — só índices `lo`, `m`, `hi` e escalares.
O que consome memória é a pilha de chamadas: uma moldura por nível ativo, e há
no máximo `⌈log₂ n⌉` níveis ativos simultaneamente.

```
S(n) = Θ(log n)
```

### 2.3 Resultados do experimento (seed 1128587)

| n | Força bruta (s) | operações | D&C (s) | operações | ganho de tempo |
|---|---|---|---|---|---|
| 100 | 0,00019 | 5.050 | 0,00013 | 970 | 1,5× |
| 250 | 0,00110 | 31.375 | 0,00034 | 2.742 | 3,2× |
| 500 | 0,00490 | 125.250 | 0,00071 | 5.986 | 6,9× |
| 1.000 | 0,01982 | 500.500 | 0,00165 | 12.974 | 12,0× |
| 2.000 | 0,08287 | 2.001.000 | 0,00326 | 27.950 | 25,4× |
| 5.000 | 0,52883 | 12.502.500 | 0,00862 | 76.806 | 61,4× |

Razão observada × razão prevista pela teoria:

| transição | força bruta obs. | previsto `(n₂/n₁)²` | D&C obs. | previsto `(n₂/n₁)·log n₂/log n₁` |
|---|---|---|---|---|
| 100 → 250 | 5,69 | 6,25 | 2,67 | 3,00 |
| 250 → 500 | 4,46 | 4,00 | 2,08 | 2,25 |
| 500 → 1.000 | 4,05 | 4,00 | 2,31 | 2,22 |
| 1.000 → 2.000 | 4,18 | 4,00 | 1,98 | 2,20 |
| 2.000 → 5.000 | 6,38 | 6,25 | 2,65 | 2,80 |

As colunas observada e prevista coincidem dentro do ruído de medição. O
experimento não apenas mostra que um algoritmo é mais rápido — ele **confirma o
expoente**: a força bruta quadruplica o tempo quando `n` dobra, o divide and
conquer pouco mais que dobra.

O gráfico log-log (Figura 3a) mostra a mesma coisa geometricamente: a
inclinação da reta é o expoente do polinômio, ≈ 2 para a força bruta e ≈ 1 para
o `n log n` (a curvatura suave é o fator logarítmico).

### 2.4 De 1.000 para 1.000.000 de registros

Extrapolando a partir de `n = 5.000`:

**Força bruta.** Fator `(10⁶ / 5.000)² = 40.000`:

```
0,52883 s × 40.000 ≈ 21.153 s ≈ 5,9 horas
```

E, se a implementação fosse a cúbica literal, o fator seria `2 × 10⁸` — algo na
casa de milhares de anos.

**Divide and conquer.** Fator `200 × (log₂10⁶ / log₂5000) = 200 × 1,62 ≈ 324`:

```
0,00862 s × 324 ≈ 2,8 segundos
```

**Conclusão.** Apenas o divide and conquer permanece viável. A memória não é o
gargalo em nenhum dos dois casos: a força bruta usa `O(1)` e o divide and
conquer usa uma pilha de `⌈log₂ 10⁶⌉ = 20` molduras. O gargalo é exclusivamente
o número de intervalos que a força bruta insiste em enumerar —
`5 × 10¹¹` para `n = 10⁶`, contra ≈ `2 × 10⁷` operações do divide and conquer.

Para escalas ainda maiores, o passo seguinte natural seria o algoritmo de
Kadane, `Θ(n)` de tempo e `O(1)` de espaço, que resolveria `10⁶` registros em
frações de segundo. Ele não foi adotado como algoritmo avaliado porque o
enunciado pede especificamente a comparação força bruta × divide and conquer,
mas está registrado aqui como a evolução recomendada.
