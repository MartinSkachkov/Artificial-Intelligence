# Genetic Algorithm for the Knapsack Problem

## 🧩 Problem Description

You are given: - **M** -- maximum allowed weight of the knapsack\

- **N** -- number of items\
- Each item has: - weight **w** - value **v**

Goal:\
**Select a subset of items so that:** - Total weight ≤ M\

- Total value is maximized

This is a classical **NP-hard optimization problem**.\
We solve it using a **Genetic Algorithm (GA)**.

---

# 🚀 Genetic Algorithm Overview

GA simulates biological evolution:

1.  **Population** -- collection of candidate solutions (chromosomes)
2.  **Fitness** -- score measuring how good each solution is
3.  **Selection** -- choose strong candidates for reproduction
4.  **Crossover** -- mix genes between parents
5.  **Mutation** -- introduce randomness
6.  **Elitism & Repair** -- stabilize and improve evolution
7.  **Repeat for many generations**

---

# 🧬 Chromosome Representation

Each solution is a binary list of length **N**:

    [1, 0, 1, 1, 0, ...]

Where: - `1` → the item **is included** - `0` → the item **is NOT
included**

Example:

    items = [(weight=3, value=10),
             (weight=5, value=20),
             (weight=2, value=5)]
    chromosome = [1,0,1]

Total weight = 3 + 2 = 5\
Total value = 10 + 5 = 15

---

# 🏋️ Fitness Function

The fitness evaluates the chromosome's total value. If weight \> M → the
solution is invalid → fitness = 0.

### Why avoid early break?

Bad (incorrect evolution):

```python
for i, bit in enumerate(chromosome):
    if bit:
        weight += items[i][0]
        value  += items[i][1]
        if weight > M:
            value = 0
            break
```

Because **early break makes invalid solutions look the same**\
→ Genetic Algorithm learns nothing.

Good:

```python
total_weight = sum(...)
total_value  = sum(...)
if overweight: return 0
```

---

# 🛠 Repair Function (Critical Part)

Mutation or crossover may produce overweight solutions.\
Repair fixes them **intelligently**.

### Strategy

Remove the _worst items first_, based on:

    value / weight   (efficiency)

This is the key idea that stabilizes evolution.

Example: \| Item \| w \| v \| v/w \| \|------\|---\|----\|-----\| \| A
\| 10 \| 100 \| 10 \| \| B \| 5 \| 10 \| 2 \| \| C \| 20 \| 40 \| 2 \|
\| D \| 1 \| 100 \| 100 \|

If overweight → remove **B and C first**, because they add little value
per kg.

### Repair code:

```python
total_weight = sum(items[i][0] for i, bit in enumerate(chromosome) if bit)
if total_weight <= max_weight:
    return

indexed = [(i, items[i][1] / items[i][0])
           for i, bit in enumerate(chromosome) if bit]

indexed.sort(key=lambda x: x[1])  # lowest efficiency first

for i, _ in indexed:
    chromosome[i] = 0
    total_weight -= items[i][0]
    if total_weight <= max_weight:
        break
```

---

# 🎯 Selection -- Tournament

Randomly sample a few individuals (e.g., 5).\
The one with best fitness wins the tournament.

Example:

    population = [[1,0,1], [0,1,1], [1,1,0]]
    fitnesses  = [100, 250, 180]

Tournament of size 2 might pick individuals: - (250) - (180)

Winner → 250

---

# 🔗 One‑Point Crossover

```python
point = random.randint(1, n-1)
child1 = parent1[:point] + parent2[point:]
child2 = parent2[:point] + parent1[point:]
```

Example:

    P1 = [1,1,0,0]
    P2 = [0,0,1,1]
    Cut at index 2

    C1 = [1,1 | 1,1] = [1,1,1,1]
    C2 = [0,0 | 0,0] = [0,0,0,0]

---

# 🔧 Mutation

Randomly flips bits with probability **0.02**.

Example:

    Before: [1,0,1,1,0]
    After:  [1,1,1,1,0]   # bit 1 flipped

---

# 🌟 Elitism

We always copy the top 5 individuals unchanged into the next
generation.\
This ensures: - Best solutions are never lost - Evolution is stable

---

# 📈 Logging Best Values

We print the best fitness at: - first generation\

- last generation\
- every 1/9 of total generations

These checkpoints help us see improvement over time.

---

# 📜 Full Code Explanation Included

The code contains: - input parser - population generator - repair -
fitness - tournament selection - crossover - mutation - genetic loop -
elitism - measuring time

Everything is combined into one working system.

---

# ✔ Summary of Key Tricks

---

Feature Why It's Important

---

Repair Prevents overweight chromosomes from ruining
evolution

Efficiency v/w Best strategy for item removal

No early break in Prevents wrong fitness ranking
fitness

Elitism Guarantees best solutions survive

Tournament Good balance between exploration & exploitation
selection

One‑point crossover Works well for binary knapsack

Mutation Prevents stagnation

---

---

# 📦 Final Notes

This README explains: - the knapsack problem\

- the GA representation\
- the tricky parts that matter\
- examples for clarity\
- why certain design choices are mandatory

It is suitable for university coursework, homework submissions, and
project documentation.
