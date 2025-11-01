# 🧩 IDA\* Solver за N-пъзел

Решение за **N-пъзела** (например 8-пъзел, 15-пъзел) чрез **IDA\*** алгоритъм — итеративна версия на A\*, която комбинира ефективност и минимална памет.

IDA\* (Iterative Deepening A\*) е търсещ алгоритъм, който:

- Използва **гранично (bounded) дълбочинно търсене (DFS)**;
- Постепенно увеличава лимита на функцията `f = g + h`;
- Използва **евристика** (Манхатън разстояние), за да ръководи търсенето;
- Гарантира **оптимално решение** (ако евристиката е допустима).

---

## 🧠 Какво представлява N-пъзелът?

Пъзелът представлява квадратна дъска `n × n`, съдържаща числата `1...N` и празно място `0`.  
Целта е да се преместят плочките така, че дъската да съвпадне с целевата конфигурация.

Пример за 8-пъзел:

```
1 2 3
4 5 6
7 0 8
```

→ цел:

```
1 2 3
4 5 6
7 8 0
```

---

## 🧩 Структура на кода

Кодът се състои от няколко логически модула:

1. Изчисление на **евристика (Манхатън разстояние)**
2. Проверка дали дадено състояние е **решимо**
3. Генериране на **възможни ходове (съседни състояния)**
4. Рекурсивен **DFS с ограничение по f = g + h**
5. Основният **IDA\*** цикъл
6. Главната програма — четене на вход, стартиране и извеждане на резултата

---

## 1️⃣ Евристика — Манхатън разстояние

```python
def manhattan(board, goal_positions, n):
    total = 0
    for i, tile in enumerate(board):
        if tile == 0:
            continue
        goal_i = goal_positions[tile]
        x1, y1 = divmod(i, n)
        x2, y2 = divmod(goal_i, n)
        total += abs(x1 - x2) + abs(y1 - y2)
    return total
```

### 🔹 Обяснение ред по ред:

- enumerate(board) обхожда всички плочки.

- Ако плочката е 0, тя се пропуска (не участва в евристиката).

- goal_positions — речник {tile: index} на целевата позиция. goal_positions[tile] дава индекса на мястото, където тази плочка трябва да бъде.

- divmod(i, n) връща (ред, колона) за дадения индекс.

- Добавяме разликите по ред и колона към total.

### 📈 Евристиката е O(N) операция, но по-късно ще видим как я оптимизираме инкрементално.

---

## 2️⃣ Проверка дали пъзелът е решим

```python
def is_solvable(board, n, zero_pos):
    tiles = [x for x in board if x != 0]
    inversions = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inversions += 1

    if n % 2 == 1:
        return inversions % 2 == 0
    else:
        zero_row = zero_pos // n
        return (inversions + zero_row) % 2 == 1
```

- Събира плочките (без 0) в списък tiles.

- Изчислява броя на инверсиите: брой двойки (i<j) с tiles[i] > tiles[j].

- Ако n е нечетно → пъзелът е решим ↔ инверсиите са четни.

- Ако n е четно → пъзелът е решим ↔ (inversions + zero_row) % 2 == 1, където zero_row = zero_pos // n (индексирано от 0, считано от горния ред).

---

## 3️⃣ Генериране на съседни състояния

```python
def get_neighbors(board, zero_idx, n):
    moves = [(-1, 0, "down"), (1, 0, "up"), (0, -1, "right"), (0, 1, "left")]
    neighbors = []

    x0, y0 = divmod(zero_idx, n)
    for dx, dy, move in moves:
        x, y = x0 + dx, y0 + dy
        if 0 <= x < n and 0 <= y < n:
            new_idx = x * n + y
            neighbors.append((new_idx, move))
    return neighbors
```

- Връща списък от съседни позиции, но не връща готови бордове — само new_idx и move (името на хода).

- Защо така: с тази версия избегваме ненужни копирания на цялата дъска в get_neighbors. В dfs() се прави копиране само при валиден ход (и еднократно), така че броят на временни обекти намалява.

Формат на резултата:

`[(new_zero_idx1, "move1"), (new_zero_idx2, "move2"), ...]`

new_zero_idx е индексът където 0 ще бъде след размяната (т.е. текущият индекс на плочката, която ще бъде преместена).

---

## 4️⃣ Рекурсивна функция DFS

```python
def dfs(path, g, bound, zero_idx, n, goal, goal_positions, current_h):
    board = path[-1]
    f = g + current_h

    if f > bound:
        return f, None

    if board == goal:
        return True, []

    min_over = float('inf')

    for new_zero, move in get_neighbors(board, zero_idx, n):
        new_board = list(board)
        moved_tile = new_board[new_zero]
        new_board[zero_idx], new_board[new_zero] = new_board[new_zero], new_board[zero_idx]
        new_board = tuple(new_board)

        if len(path) > 1 and new_board == path[-2]:
            continue

        goal_i = goal_positions[moved_tile]
        gx, gy = divmod(goal_i, n)

        old_x, old_y = divmod(zero_idx, n)
        new_x, new_y = divmod(new_zero, n)

        old_dist = abs(old_x - gx) + abs(old_y - gy)
        new_dist = abs(new_x - gx) + abs(new_y - gy)

        new_h = current_h - old_dist + new_dist

        path.append(new_board)
        result, moves = dfs(path, g + 1, bound, new_zero, n, goal, goal_positions, new_h)
        path.pop()

        if result is True:
            return True, [move] + moves

        if result < min_over:
            min_over = result

    return min_over, None
```

Това е рекурсивната DFS функция с ограничение bound. Обяснение на всички важни стъпки:

1. board = path[-1] — текущата дъска (tuple).

2. f = g + current_h.

   - Ако f > bound: връщаме f (нагоре ще се използва за изчисляване на следващото bound).

3. Ако board == goal: върни True, [] (успех; един празен списък — надолу ще се добавят ходовете).

4. min_over = inf — ще пазим най-малкото f което е по-голямо от bound (за предложеното следващо bound).

5. За всеки съсед new_zero, move in get_neighbors(...):

   - Създаваме new_board = list(board) и moved_tile = new_board[new_zero].

   - Разменяме new_board[zero_idx] и new_board[new_zero] → плочката и 0 си сменят местата.

   - new_board = tuple(new_board) — налага се immutable за сравнения и съхраняване в path.

   - Ако len(path) > 1 и new_board == path[-2] → прескочи (не връщай обратно към предишното състояние).

   - Инкрементална евристика:

     - goal_i = goal_positions[moved_tile] → целевата позиция на преместената плочка.

     - old_x,old_y = divmod(new_zero,n) — къде е била плочката преди (new_zero).

     - new_x,new_y = divmod(zero_idx,n) — къде отива тя (zero_idx).

     - old_dist и new_dist са Манхатън разстоянията спрямо целта; new_h = current_h - old_dist + new_dist.

   - path.append(new_board) — влизаме в новото състояние.

   - Рекурсивно: result, moves = dfs(path, g+1, bound, new_zero, n, goal, goal_positions, new_h).

   - path.pop() — връщане (backtracking).

   - Ако result is True → върни True, [move] + moves (сглобява пътя отдолу нагоре).

   - Ако result < min_over → min_over = result.

6. След разглеждане на всички съседи → return min_over, None.

---

## 5️⃣ Главен цикъл IDA\*

```python
def ida_star(start, goal, n, goal_positions, zero_pos):
    h0 = manhattan(start, goal_positions, n)
    bound = h0
    path = [start]

    while True:
        result, moves = dfs(path, 0, bound, zero_pos, n, goal, goal_positions, h0)
        if result is True:
            return moves
        if result == float('inf'):
            return -1
        bound = result
```

- Взема началното h0 = manhattan(start, goal_positions, n).

- bound = h0.

- path = [start].

- Цикъл:

  - result, moves = dfs(path, 0, bound, zero_pos, n, goal, goal_positions, h0).

  - Ако result is True → връщаме moves.

  - Ако result == inf → несъвместимо/нерешимо → връщаме -1.

  - Иначе bound = result (следващата граница) и повторение.

Бележка: тук подаваме h0 на първото извикване; при всяко следващо извикване на dfs() в цикъла bound се обновява и рекурсията ще актуализира current_h инкрементално.

---

## 6️⃣ Главна функция

```python
def main():
    N = int(input().strip())
    I = int(input().strip())
    n = int(math.sqrt(N + 1))

    tiles = []
    for _ in range(n):
        row = list(map(int, input().split()))
        tiles.extend(row)

    start = tuple(tiles)

    if I == -1:
        goal = tuple(list(range(1, N + 1)) + [0])
    else:
        goal = tuple(list(range(1, I + 1)) + [0] + list(range(I + 1, N + 1)))

    goal_positions = {tile: i for i, tile in enumerate(goal)}
    zero_pos = start.index(0)

    if not is_solvable(start, n, zero_pos):
        print(-1)
        return

    result = ida_star(start, goal, n, goal_positions, zero_pos)

    if result == -1:
        print(-1)
    else:
        print(len(result))
        for move in result:
            print(move)
```

---

## 7️⃣ Стартиране

```python
if __name__ == "__main__":
    main()
```

---

## Защо и как работи инкременталното обновяване на h (Manhattan)

### Основна идея

При всяка размяна `0 ↔ tile`, само една плочка (`moved_tile`) променя позицията си. Значи цялата сума `h` се променя само заради нея:

```
h_new = h_old - dist(old_pos_of_tile, goal_pos) + dist(new_pos_of_tile, goal_pos)
```

### В кода

- `new_zero` е стара позиция на плочката (преди размяната)
- `zero_idx` е новата ѝ позиция (след размяната) — това е мястото на старото 0
- Използваме `divmod` да превърнем индекси в `(row, col)`
- Пресмятаме `old_dist` и `new_dist` и актуализираме `current_h`

**Резултат:** Това прави актуализацията O(1) вместо O(N). При пъзели, които разглеждат милиони възли, това е огромна икономия.

---

## Защо правим path.append / path.pop и защо сравняваме с path[-2]

`path` съдържа активния път от началото до текущия възел. Това е нужно, за да можем:

1. Да предотвратим директен обратен ход (A→B→A), което експоненциално увеличава дървото
2. Да върнем точно пътя, когато достигнем целта (сглобяване на ходовете отдолу нагоре)

### Механизъм

- `path[-1]` е текущият възел
- `path[-2]` е родителят (от който сме дошли)

Ако генерираме `new_board == path[-2]` — това означава "връщане обратно". Такъв ход е безсмислен и се прескача (принципно A→B→A не води до оптимално решение и удвоява ненужно дървото).

```python
if len(path) > 1 and new_board == path[-2]:
    continue
```

---

## Посоките (left/up/right/down) — как са съотнесени към движението на 0 и към движението на плочката

### Важно разграничение

В кода:

```python
moves = [(-1,0,"down"), (1,0,"up"), (0,-1,"right"), (0,1,"left")]
```

- Комбинациите `(dx, dy)` описват **движението на 0** в матрицата (защото ние изчисляваме новите координати на 0: `x = x0 + dx`, `y = y0 + dy`)
- Имената `"down"` / `"up"` / `"left"` / `"right"` обаче сме избрали да представят **реалното движение на плочката**, а не на 0

Това е направено така, че да се получи по-интуитивен изход (човек гледа кой накъде е придвижен).

### Пример

Когато 0 „се мести нагоре" (`dx = -1`), това означава, че плочката, която е била над 0, всъщност е преместена **надолу** в 0 — затова име `"down"`.

**Ако искаш имената да описват движението на 0** (а не на плочките), трябва да ги обърнеш.

### Практически съвет

Ако наблюдаваш „огледални" резултати (напр. получаваш `right right`, а очакваш `left left`), провери тази асоциация на имената и обърни при нужда.

---

## Проверка за решимост (инверсии) — подробности и често срещани грешки

### Определение

`inversions` = брой двойки `(i < j)` такива, че `tile_i > tile_j` при линейно четене (row-major), **без 0**.

### Правила за решимост

#### Ако n (страна) е нечетна:

- Пъзелът е решим ↔ `inversions` е четно

#### Ако n е четна:

- Инвариант е `(inversions + row_of_blank)` (row отгоре, 0-based)
- Пъзелът е решим ↔ `(inversions + row_of_blank) % 2 == 1`

Това е правилото, което кодът използва.

### Често срещани грешки

1. **Грешно броене на реда на празното** от долу вместо отгоре (в кода е отгоре, 0-based)
2. **Въвеждане на I** (позицията на 0 в целта) 1-based вместо 0-based; ако не си сигурен за формата на I, направи валидация и/или подай debug output

---

## Какво точно връщат функциите и как се формира изходът

### `dfs` връща кортеж `(result, moves)`

- **`result is True`** → `moves` е списък от ходове (строките `"left"`, `"right"`, `"up"`, `"down"`) от текущия възел до целта. В този случай `dfs` кумулира хода при връщането (`[move] + moves`) за всяка вложена рекурсия
- **`result` е число** (минималната стойност `f` превишила `bound`) → `moves` е `None`
- **`result == float('inf')`** → означава изчерпване (няма решения при никаква граница) → връща се `-1` в `ida_star`

### `ida_star` връща `moves` или `-1`

`main` печата резултатите съответно.
