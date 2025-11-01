import math

# ---------------------------------------------------
# 1. Евристика - Манхатън разстояние (начално изчисление)
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


# ---------------------------------------------------
# 3. Генериране на съседите
# ---------------------------------------------------
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


# ---------------------------------------------------
# 4. Рекурсивна функция за IDA*
# ---------------------------------------------------
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

        # разменяме плочките
        new_board[zero_idx], new_board[new_zero] = new_board[new_zero], new_board[zero_idx]
        new_board = tuple(new_board)

        # избягваме връщане към предишното състояние
        if len(path) > 1 and new_board == path[-2]:
            continue

        # изчисляваме новата евристика инкрементално
        goal_i = goal_positions[moved_tile]
        gx, gy = divmod(goal_i, n)

        # предишна позиция на плочката
        old_x, old_y = divmod(new_zero, n)
        new_x, new_y = divmod(zero_idx, n)

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


# ---------------------------------------------------
# 5. IDA* търсене
# ---------------------------------------------------
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


# ---------------------------------------------------
# 6. Главна програма
# ---------------------------------------------------
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


# ---------------------------------------------------
# 7. Стартиране
# ---------------------------------------------------
if __name__ == "__main__":
    main()
