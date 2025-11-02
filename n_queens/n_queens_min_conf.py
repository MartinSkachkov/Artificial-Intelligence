import random
import time

def print_board_state(queenPosition):
    n = len(queenPosition)
    for r in range(n):
        row_str = ["*" if queenPosition[c] == r else "_" for c in range(n)]
        print(" ".join(row_str))


def count_conflicts(row, col, n, conflictRows, conflictPosDiag, conflictNegDiag):
    """O(1) изчисляване на броя конфликти за дадена позиция."""
    return (
        conflictRows[row]
        + conflictPosDiag[row + col]
        + conflictNegDiag[n - 1 - row + col]
        - 3  # изваждаме собствената царица
    )


def solve_n_queens(n, print_board=False, measure_time=True, max_steps=1000000):
    if n in (2, 3):
        return -1

    start_time = time.time()

    # --- Инициализация ---
    queenPosition = [random.randint(0, n - 1) for _ in range(n)]
    conflictRows = [0] * n
    conflictPosDiag = [0] * (2 * n - 1)
    conflictNegDiag = [0] * (2 * n - 1)

    # Попълваме конфликтните броячи
    for col in range(n):
        row = queenPosition[col]
        conflictRows[row] += 1
        conflictPosDiag[row + col] += 1
        conflictNegDiag[n - 1 - row + col] += 1

    # --- Главен цикъл ---
    for _ in range(max_steps):
        conflicted_cols = []
        for col in range(n):
            row = queenPosition[col]
            if (
                conflictRows[row]
                + conflictPosDiag[row + col]
                + conflictNegDiag[n - 1 - row + col]
                > 3
            ):
                conflicted_cols.append(col)

        if not conflicted_cols:
            # Намерено решение
            elapsed = time.time() - start_time
            if measure_time:
                print(f"⏱ Време за решаване: {elapsed:.5f} секунди")

            if print_board and n <= 30:
                print_board_state(queenPosition)

            return queenPosition

        # Избираме случайна конфликтна колона
        col = random.choice(conflicted_cols)
        current_row = queenPosition[col]

        # Премахваме временно текущата царица
        conflictRows[current_row] -= 1
        conflictPosDiag[current_row + col] -= 1
        conflictNegDiag[n - 1 - current_row + col] -= 1

        # Намираме ред с минимален конфликт
        min_conf = float("inf")
        best_rows = []
        for row in range(n):
            conf = (
                conflictRows[row]
                + conflictPosDiag[row + col]
                + conflictNegDiag[n - 1 - row + col]
            )
            if conf < min_conf:
                min_conf = conf
                best_rows = [row]
            elif conf == min_conf:
                best_rows.append(row)

        new_row = random.choice(best_rows)
        queenPosition[col] = new_row

        # Актуализираме конфликтните масиви
        conflictRows[new_row] += 1
        conflictPosDiag[new_row + col] += 1
        conflictNegDiag[n - 1 - new_row + col] += 1

    # Ако не намери решение за max_steps
    return -1


# ==== Примерна употреба ====
if __name__ == "__main__":
    n = int(input().strip())
    result = solve_n_queens(n, print_board=True, measure_time=True)
    print(result)
