import math

# ---------------------------------------------------
# 1. Евристика - Манхатън разстояние
# ---------------------------------------------------
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


# ---------------------------------------------------
# 2. Проверка дали даден пъзел е решим
# ---------------------------------------------------
def is_solvable(board, n, zero_pos):
    # преброяване на инверсиите
    tiles = [x for x in board if x != 0]
    inversions = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inversions += 1

    # нечетна дъска
    if n % 2 == 1:
        return inversions % 2 == 0
    # четна дъска
    else:
        zero_pos_row = zero_pos // n
        return (inversions + zero_pos_row) % 2 == 1


# ---------------------------------------------------
# 3. Генериране на всички възможни ходове от текущо състояние
# ---------------------------------------------------
def get_neighbors(board, zero_idx, n):
    moves = [(-1, 0, "down"), (1, 0, "up"), (0, -1, "right"), (0, 1, "left")]
    neighbors = []

    x0, y0 = divmod(zero_idx, n)
    for dx, dy, move in moves:
        x, y = x0 + dx, y0 + dy
        if 0 <= x < n and 0 <= y < n:
            new_idx = x * n + y
            new_board = list(board)
            new_board[zero_idx], new_board[new_idx] = new_board[new_idx], new_board[zero_idx]
            neighbors.append((tuple(new_board), new_idx, move))
    return neighbors


# ---------------------------------------------------
# 4. Рекурсивна функция за IDA* търсене
# ---------------------------------------------------
def dfs(path, g, bound, zero_idx, n, goal, goal_positions):
    board = path[-1]
    h = manhattan(board, goal_positions, n)
    f = g + h

    if f > bound:
        return f, None

    if board == goal:
        return True, []

    min_over = float('inf')

    for new_board, new_zero, move in get_neighbors(board, zero_idx, n):
        # не връщай състояние към предишното (за да не циклим)
        if len(path) > 1 and new_board == path[-2]:
            continue

        path.append(new_board)
        result, moves = dfs(path, g + 1, bound, new_zero, n, goal, goal_positions)
        path.pop()

        if result is True:
            return True, [move] + moves

        if result < min_over:
            min_over = result

    return min_over, None


# ---------------------------------------------------
# 5. Главната функция - IDA* търсене
# ---------------------------------------------------
def ida_star(start, goal, n, goal_positions, zero_pos):
    bound = manhattan(start, goal_positions, n)
    path = [start]

    while True:
        result, moves = dfs(path, 0, bound, zero_pos, n, goal, goal_positions)
        if result is True:
            return moves
        if result == float('inf'):
            return -1
        bound = result


# ---------------------------------------------------
# 6. Главна програма
# ---------------------------------------------------
def main():
    # прочитане на вход
    N = int(input().strip()) # брой плочки с номера
    I = int(input().strip()) # индекс на празната плочка в целта
    n = int(math.sqrt(N + 1))

    # четем редове за дъската
    tiles = []
    for _ in range(n):
        row = list(map(int, input().split()))
        tiles.extend(row)

    start = tuple(tiles)

    # генерираме целевото състояние
    goal = list(range(1, N + 1)) + [0]
    if I != -1:
        goal[I], goal[-1] = goal[-1], goal[I]
    goal = tuple(goal)

    goal_positions = {tile: i for i, tile in enumerate(goal)}
    zero_pos = start.index(0)

    # проверка за решимост
    if not is_solvable(start, n, zero_pos):
        print(-1)
        return

    # намираме решение
    result = ida_star(start, goal, n, goal_positions, zero_pos)

    if result == -1:
        print(-1)
    else:
        print(len(result))
        for move in result:
            print(move)


# ---------------------------------------------------
# 7. Стартиране
# ---------------------------------------------------
if __name__ == "__main__":
    main()
