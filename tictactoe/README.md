# [Tic-Tac-Toe Minimax Agent](https://www.youtube.com/watch?v=SLgZhpDsrfc)

Детерминистичен агент за игра на морски шах, който играе **оптимално** използвайки алгоритъма **Minimax с Alpha-Beta отсичане**.

---

## Съдържание

1. [Как да стартирам](#как-да-стартирам)
2. [Режими на работа](#режими-на-работа)
3. [Архитектура и функции](#архитектура-и-функции)
4. [Minimax алгоритъм](#minimax-алгоритъм)
5. [Alpha-Beta отсичане](#alpha-beta-отсичане)
6. [Ролята на depth](#ролята-на-depth)
7. [Критични моменти](#критични-моменти)
8. [Примери](#примери)

---

## Как да стартирам

```bash
python tictactoe.py
```

За тестване с judge системата:

```bash
judge run --bench --problem tictactoe tictactoe.py
```

---

## Режими на работа

### Режим JUDGE

Използва се от автоматичната система за тестване.

**Вход:**

```
JUDGE
TURN X
+---+---+---+
| _ | _ | _ |
+---+---+---+
| _ | _ | _ |
+---+---+---+
| _ | _ | _ |
+---+---+---+
```

**Изход:**

- За нетерминални позиции: `row col` (1-базирана индексация)
- За терминални позиции: `-1`

**Примерен изход:** `2 2` (център на дъската)

---

### Режим GAME

Интерактивна игра човек-компютър.

**Вход:**

```
GAME
FIRST X
HUMAN O
+---+---+---+
| _ | _ | _ |
+---+---+---+
| _ | _ | _ |
+---+---+---+
| _ | _ | _ |
+---+---+---+
```

След това въвеждате ходове като `1 1`, `2 3` и т.н.

**Краен изход:** `WINNER: X`, `WINNER: O` или `DRAW`

---

## Архитектура и функции

### Константи

```python
X = 'X'      # Играч X (MAX - максимизира)
O = 'O'      # Играч O (MIN - минимизира)
EMPTY = '_'  # Празна клетка
```

---

### `terminal(state)`

**Цел:** Проверява дали играта е приключила.

**Връща:** `True` ако има победител или дъската е пълна, `False` иначе.

**Как работи:**

1. Проверява всички 8 линии (3 реда + 3 колони + 2 диагонала)
2. Ако някоя линия има 3 еднакви символа (не празни) → победа
3. Ако няма победител и няма празни клетки → равенство

```python
def terminal(state):
    lines = [
        # Редове
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        # Колони
        [state[0][0], state[1][0], state[2][0]],
        [state[0][1], state[1][1], state[2][1]],
        [state[0][2], state[1][2], state[2][2]],
        # Диагонали
        [state[0][0], state[1][1], state[2][2]],
        [state[0][2], state[1][1], state[2][0]],
    ]
    for line in lines:
        if line[0] != EMPTY and line[0] == line[1] == line[2]:
            return True
    for row in state:
        if EMPTY in row:
            return False
    return True
```

**Пример:**

```
| X | X | X |
| O | O | _ |  →  terminal() = True (X печели)
| _ | _ | _ |
```

---

### `winner(state)`

**Цел:** Определя кой е победителят.

**Връща:** `'X'`, `'O'` или `None`

```python
def winner(state):
    # Същата логика като terminal(), но връща победителя
    ...
    return line[0]  # или None ако няма победител
```

---

### `value(state, depth)`

**Цел:** Оценява терминално състояние.

**Връща:** Числова оценка в интервала [-9, 9]

```python
def value(state, depth):
    w = winner(state)
    if w == X:
        return 10 - depth  # X печели: [1, 9]
    elif w == O:
        return depth - 10  # O печели: [-9, -1]
    return 0               # Равенство
```

**Защо `10 - depth`?**

- По-бърза победа = по-висока оценка
- Победа на depth=1: `10 - 1 = 9`
- Победа на depth=5: `10 - 5 = 5`
- Агентът предпочита 9 > 5, т.е. по-бързата победа

---

### `player(state)`

**Цел:** Определя кой е на ход.

**Връща:** `'X'` или `'O'`

**Логика:** X винаги започва първи, така че:

- Ако брой X == брой O → X е на ход
- Иначе → O е на ход

```python
def player(state):
    x_count = sum(row.count(X) for row in state)
    o_count = sum(row.count(O) for row in state)
    return X if x_count == o_count else O
```

**Пример:**

```
| X | O | _ |
| _ | X | _ |  →  X=2, O=1  →  player() = 'O'
| _ | _ | _ |
```

---

### `actions(state)`

**Цел:** Връща всички възможни ходове.

**Връща:** Списък от кортежи `[(row, col), ...]` (0-базирана индексация)

```python
def actions(state):
    moves = []
    for i in range(3):
        for j in range(3):
            if state[i][j] == EMPTY:
                moves.append((i, j))
    return moves
```

**Пример:**

```
| X | O | _ |
| _ | X | _ |  →  actions() = [(0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]
| _ | _ | _ |
```

---

### `result(state, action)`

**Цел:** Прилага ход и връща ново състояние.

**Важно:** Създава **копие** на дъската, не модифицира оригинала!

```python
def result(state, action):
    i, j = action
    new_state = [row[:] for row in state]  # Deep copy
    new_state[i][j] = player(state)
    return new_state
```

**Пример:**

```
Преди:              action=(1,0)        След:
| X | O | _ |                           | X | O | _ |
| _ | X | _ |       →                   | O | X | _ |
| _ | _ | _ |                           | _ | _ | _ |
```

---

## Minimax алгоритъм

### Концепция

Minimax е алгоритъм за вземане на решения в игри с двама играчи:

- **MAX (X):** Иска да **максимизира** оценката
- **MIN (O):** Иска да **минимизира** оценката

Алгоритъмът симулира всички възможни игри и избира оптималния ход.

### Псевдокод

```
Minimax(state):
    if Terminal(state):
        return Value(state)

    if Player(state) == MAX:
        value = -∞
        for action in Actions(state):
            value = Max(value, Minimax(Result(state, action)))
        return value

    if Player(state) == MIN:
        value = +∞
        for action in Actions(state):
            value = Min(value, Minimax(Result(state, action)))
        return value
```

### Визуализация

```
              MAX (X) избира максимум
             /     |     \
           MIN   MIN    MIN   (O) избира минимум
          / \    / \    / \
         5  3   2  9   1  7
         ↓      ↓      ↓
        MIN    MIN    MIN
       избира избира избира
         3      2      1

MAX избира max(3, 2, 1) = 3
```

---

## Alpha-Beta отсичане

### Концепция

Оптимизация на Minimax, която **отрязва клонове**, които не могат да повлияят на крайното решение.

- **α (alpha):** Най-добрата стойност, която MAX може да гарантира
- **β (beta):** Най-добрата стойност, която MIN може да гарантира

**Правило за отсичане:** Ако `β ≤ α`, спираме търсенето в текущия клон.

### Имплементация

```python
def minimax(state, depth, alpha, beta, maximizing):
    if terminal(state):
        return value(state, depth), None

    best_action = None

    if maximizing:  # MAX (X)
        max_eval = -math.inf
        for action in actions(state):
            eval_score, _ = minimax(result(state, action), depth + 1, alpha, beta, False)
            if eval_score > max_eval:
                max_eval = eval_score
                best_action = action
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # β отсичане
        return max_eval, best_action

    else:  # MIN (O)
        min_eval = math.inf
        for action in actions(state):
            eval_score, _ = minimax(result(state, action), depth + 1, alpha, beta, True)
            if eval_score < min_eval:
                min_eval = eval_score
                best_action = action
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # α отсичане
        return min_eval, best_action
```

### Пример за отсичане

```
         MAX (α=3)
        /         \
      MIN         MIN
     /   \          |
    5     3        2  ← MIN ще върне ≤2
                      ← Но MAX вече има 3!
                      ← 2 < 3, така че ОТСИЧАНЕ
                      ← Не проверяваме останалите
```

---

## Ролята на depth

### Защо е необходим?

Без depth алгоритъмът не различава **бърза** от **бавна** победа.

### Пример

```
| X | X | _ |
| O | O | _ |
| _ | _ | _ |
```

X има два хода към победа:

| Ход   | Резултат           | Без depth | С depth      |
| ----- | ------------------ | --------- | ------------ |
| (1,3) | Победа веднага     | +10       | 10-1 = **9** |
| (3,1) | Победа след 4 хода | +10       | 10-5 = **5** |

**С depth:** 9 > 5, избираме незабавната победа!

### Граници на depth

- **Минимум:** 1 (първи ход)
- **Максимум:** 9 (дъската е пълна)
- **Depth > 9 е невъзможен** (има само 9 клетки)

---

## Критични моменти

### 1. Deep Copy при result()

```python
# ГРЕШНО - модифицира оригинала!
new_state = state
new_state[i][j] = player(state)

# ПРАВИЛНО - създава копие
new_state = [row[:] for row in state]
new_state[i][j] = player(state)
```

### 2. Индексация

- **Вътрешно:** 0-базирана (0, 1, 2)
- **Вход/Изход:** 1-базирана (1, 2, 3)

```python
# При четене
row, col = int(move[0]) - 1, int(move[1]) - 1

# При писане
print(f"{action[0] + 1} {action[1] + 1}")
```

### 3. Начални стойности на alpha/beta

```python
alpha = -math.inf  # MAX започва от най-лошото
beta = math.inf    # MIN започва от най-лошото
```

### 4. Проверка за терминално състояние ПРЕДИ търсене

```python
if terminal(state):
    print(-1)
    return
```

### 5. Кой е на ход - базирано на броя символи

```python
# X винаги започва, така че:
# X==O означава X е на ход
# X>O означава O е на ход
return X if x_count == o_count else O
```

---

## Примери

### Пример 1: Оптимален първи ход

**Вход:**

```
JUDGE
TURN X
+---+---+---+
| _ | _ | _ |
+---+---+---+
| _ | _ | _ |
+---+---+---+
| _ | _ | _ |
+---+---+---+
```

**Изход:** `2 2` (центърът е оптимален първи ход)

---

### Пример 2: Незабавна победа

**Вход:**

```
JUDGE
TURN X
+---+---+---+
| X | X | _ |
+---+---+---+
| O | O | _ |
+---+---+---+
| _ | _ | _ |
+---+---+---+
```

**Изход:** `1 3` (X печели веднага)

---

### Пример 3: Блокиране на противника

**Вход:**

```
JUDGE
TURN X
+---+---+---+
| O | O | _ |
+---+---+---+
| _ | X | _ |
+---+---+---+
| _ | _ | _ |
+---+---+---+
```

**Изход:** `1 3` (блокира O от победа)

---

### Пример 4: Терминална позиция

**Вход:**

```
JUDGE
TURN X
+---+---+---+
| X | X | X |
+---+---+---+
| O | O | _ |
+---+---+---+
| _ | _ | _ |
+---+---+---+
```

**Изход:** `-1` (играта вече е приключила)

---

## Сложност

| Метрика            | Без Alpha-Beta     | С Alpha-Beta        |
| ------------------ | ------------------ | ------------------- |
| Време (worst case) | O(b^d)             | O(b^(d/2))          |
| За Tic-Tac-Toe     | ~362,880 състояния | ~500-1000 състояния |

Където:

- b = branching factor (средно ~4-5 за Tic-Tac-Toe)
- d = дълбочина (максимум 9)

---
