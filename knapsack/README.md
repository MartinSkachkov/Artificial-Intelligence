# [Генетичен алгоритъм за задачата „Раница“ (Knapsack) — Пояснена документация](https://www.youtube.com/watch?v=MacVqujSXWE)

## Описание на задачата

Имаме:

- **M** — максимална допустима тежест на раницата.
- **N** — брой предмети.
- Всеки предмет има:
  - тегло `w`
  - стойност `v`

Цел: Избери подмножество от предмети така, че:

- общото тегло ≤ M
- общата стойност да бъде максимална

Това е класически NP-труден оптимизационен проблем. Ще го решим с генетичен алгоритъм (GA).

---

## Кратък преглед на генетичния алгоритъм

Стъпки:

1. **Представяне (хромозома)** — всеки кандидат-решение е списък от 0/1 с дължина N.
2. **Популация** — множество от хромозоми.
3. **Фитнес (fitness)** — оценка на хромозомата (обща стойност, ако теглото ≤ M; иначе 0).
4. **Селекция** — избираме родители за размножаване (напр. турнирна селекция).
5. **Кросоувър** — комбинираме родители (напр. едноточков).
6. **Мутация** — добавяме малка случайност.
7. **Repair** — поправяме хромозоми, които надвишават теглото.
8. **Елитизъм** — пазим няколко най-добри индивида без промяна.

---

## Представяне (пример)

Ако имаме 5 предмета, една хромозома може да бъде:

```
[1, 0, 1, 1, 0]
```

Това означава: взимаме предмети 1, 3 и 4.

---

## Фитнес функция — защо не правим ранно `break`

```python
for i, bit in enumerate(chromosome):
    if bit:
        total_weight += items[i][0]
        total_value += items[i][1]
        if total_weight > max_weight:
            total_value = 0
            break
```

---

## Repair (поправка) — детайлно

Кросоувърът и мутацията могат да произведат хромозоми с общо тегло > M. Repair ги прави валидни.

### Стратегия (value/weight):

За всеки включен предмет изчисляваме `ratio = value / weight`. Махаме предметите с най-нисък ratio първо — това минимизира загубата на стойност при намаляване на теглото.

Примерна реализация:

```python
def repair(chromosome, items, max_weight):
    total_weight = sum(items[i][0] for i, bit in enumerate(chromosome) if bit)
    if total_weight <= max_weight:
        return
    indexed = [(i, items[i][1] / items[i][0]) for i, bit in enumerate(chromosome) if bit]
    indexed.sort(key=lambda x: x[1])  # най-нисък ratio първо
    for i, _ in indexed:
        chromosome[i] = 0
        total_weight -= items[i][0]
        if total_weight <= max_weight:
            break
```

**Защо работи добре:**

- премахваме предмети, които носят най-малко стойност на единица тегло;
- запазваме възможно най-много стойност при корекцията.

**Защо random repair е лош:**

- ако махаш произволни предмети, може да свалиш точно критичните, които дават голяма стойност, и така да развалиш добрите хромозоми.

---

## Турнирна селекция (tournament selection)

- Вземаме `k` произволни индивида от популацията.
- Победител е този с най-висок fitness.
- Турнирът се повтаря, за да изберем двама родители.

Пример:

```
population = [[1,0,1],[0,1,1],[1,1,0]]
fitnesses  = [100, 200, 150]
TOURNAMENT_SIZE = 2
random.sample -> picks e.g. ([1,0,1], 100) and ([0,1,1],200) -> победител [0,1,1]
```

---

## Кръстосване — едноточково (one-point crossover)

```python
point = random.randint(1, n-1)
child1 = parent1[:point] + parent2[point:]
child2 = parent2[:point] + parent1[point:]
```

Пример:

```
p1 = [1,1,0,0]
p2 = [0,0,1,1]
cut=2 -> c1=[1,1,1,1], c2=[0,0,0,0]
```

---

## Мутация

- Разглеждаме всеки ген и с малка вероятност (например 0.02) обръщаме бита.
- Важно за откриване на нови решения и избягване на застой.

---

## Елитизъм

- Копираме `ELITE_COUNT` най-добри хромозоми директно в следващото поколение.
- Това гарантира, че най-добрите решения не се губят поради кросоувър/мутация.

---

## Пълният код (стъпка по стъпка обяснен)

Следва опростен и добре структуриран код. Под всяка функция има обяснение.

```python
import time
import random

# Параметри
POPULATION_SIZE = 200
GENERATIONS = 100
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.02
ELITE_COUNT = 5

# Четене на вход:
def read_input():
    # Връща M, N, items (списък от (w, v))
    M, N = map(int, input().split())
    items = []
    for _ in range(N):
        w, v = map(int, input().split())
        items.append((w, v))
    return M, N, items
```

**Обяснение:** четем капацитета и броя предмети; след това N реда с тегло и стойност.

```python
def fitness(individual, items, max_weight):
    # Пресмятаме общо тегло и обща стойност
    total_weight = total_value = 0
    for i, bit in enumerate(individual):
        if bit:
            total_weight += items[i][0]
            total_value += items[i][1]
    # Връщаме стойност само ако е валидна
    return total_value if total_weight <= max_weight else 0
```

**Обяснение:** изчисляваме теглото на всяка една хромозома

```python
def repair(individual, items, max_weight):
    # Поправя хромозомата, като маха най-неефективните предмети
    total_weight = sum(items[i][0] for i, bit in enumerate(individual) if bit)
    if total_weight <= max_weight:
        return
    # Списък от (index, value/weight)
    indexed = [(i, items[i][1] / items[i][0]) for i, bit in enumerate(individual) if bit]
    indexed.sort(key=lambda x: x[1])  # най-нисък ratio първо
    for i, _ in indexed:
        individual[i] = 0
        total_weight -= items[i][0]
        if total_weight <= max_weight:
            break
```

**Обяснение:** пазим информацията за value/weight и махаме най-лошите до валидиране.

```python
def generate_individual(n, items, max_weight):
    # Създаваме случайна хромозома и я поправяме веднага
    ind = [random.randint(0, 1) for _ in range(n)]
    repair(ind, items, max_weight)
    return ind

def generate_population(size, n, items, max_weight):
    return [generate_individual(n, items, max_weight) for _ in range(size)]
```

**Обяснение:** началната популация е валидна благодарение на repair — избягваме много хромозоми с fitness=0.

```python
def tournament_selection(population, fitnesses):
    # Вземаме TOURNAMENT_SIZE произволни индивиди (с техните фитнеси) и избираме най-добрия
    selected = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]
```

**Обяснение:** прост и ефективен метод за селекция.

```python
def one_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2
```

**Обяснение:** класически едноточков кросоувър.

```python
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]
```

**Обяснение:** флипваме случайни гени.

# Главен алгоритъм

```python
def genetic_algorithm(M, N, items):
    start_time = time.time()
    population = generate_population(POPULATION_SIZE, N, items, M)
    best_values = []

    for gen in range(GENERATIONS):
        fitnesses = [fitness(ind, items, M) for ind in population]
        new_population = []

        # Елитизъм — запази най-добрите
        elite = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:ELITE_COUNT]
        new_population.extend([e[0] for e in elite])

        # Попълни останалата част чрез селекция, кръстосване, мутация и repair
        while len(new_population) < POPULATION_SIZE:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            c1, c2 = one_point_crossover(p1, p2)
            mutate(c1); mutate(c2)
            repair(c1, items, M); repair(c2, items, M)
            new_population.append(c1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(c2)

        population = new_population

        # Оценка и логване на напредъка
        fitnesses = [fitness(ind, items, M) for ind in population]
        best = max(fitnesses)
        if gen == 0 or gen == GENERATIONS - 1 or gen % (GENERATIONS // 9) == 0:
            print(best)
            best_values.append(best)

    print()
    print(best_values[-1])
    print("Time:", round(time.time() - start_time, 3), "seconds")
```

---

## Примери за вход и изход

### Пример 1 (малък)

Вход:

```
50 5
10 60
20 100
30 120
5 15
15 40
```

Изход (примерно):

```
100
180
195
... (10 стойности общо)
195

195
Time: 0.12 seconds
```

### Пример 2 (KP short test data)

- M = 5000, N = 24 → оптимум 1130 (очакван)
- С горния GA обикновено (в повечето пускания) ще достигнеш оптимума.

---

## Съвети и фини настройки

- Увеличаване на `POPULATION_SIZE` → по-голямо изследване, по-бавна работа.
- Увеличаване на `GENERATIONS` → повече време за еволюция.
- `TOURNAMENT_SIZE` → по-голям = по-силна селекция (експлоатация), по-малък = повече разнообразие (експлорация).
- `MUTATION_RATE` 0.01–0.05 е типична.
- `ELITE_COUNT` 1–10; запазва страхотни решения.

---

## Чести проблеми (и решения)

- **Алгоритъм "застава" в лош локален минимум** → опитай по-голям mutation rate или по-голяма популация.
- **Много хромозоми с fitness 0** → поправи генерирането или repair функцията.
- **Бавно изпълнение при N много голям (≈10k)** → оптимизирай fitness (zip вместо enumerate), или използвай vectorized библиотеки.

---

## Комплексност и бързи бележки

- Една генерация: O(population_size \* (cost на fitness + cost на repair + ...))
- Repair със сортиране е O(k log k), където k е броят на включените предмети в хромозомата (обикновено ≤ N).
- В практиката repair с ratio е балансирано и води до по-бързо събиране на решения (по-малко поколението са "безполезни").

---
