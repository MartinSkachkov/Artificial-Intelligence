import random
import sys
import time

DEBUG = False  # <- Включи само за ръчно тестване

def initialise_data(n):
    queens = [0] * n
    queens_per_row = [0] * n
    diag1 = [0] * (2 * n - 1)
    diag2 = [0] * (2 * n - 1)
    return queens, queens_per_row, diag1, diag2


def arrange_queens_on_board(n, queens, queens_per_row, diag1, diag2):
    col = 1
    for row in range(n):
        queens[col] = row
        queens_per_row[row] += 1
        diag1[col - row + n - 1] += 1
        diag2[col + row] += 1
        col += 2
        if col >= n:
            col = 0


def retrieve_conflicts(row, col, queens_per_row, diag1, diag2, n):
    return queens_per_row[row] + diag1[col - row + n - 1] + diag2[col + row]


def get_column_with_max_conflict(n, queens, queens_per_row, diag1, diag2):
    max_conf = -1
    cols = []
    for col in range(n):
        row = queens[col]
        current_conf = retrieve_conflicts(row, col, queens_per_row, diag1, diag2, n) - 3
        if current_conf == max_conf:
            cols.append(col)
        elif current_conf > max_conf:
            max_conf = current_conf
            cols = [col]
    if max_conf == 0:
        return -1
    return random.choice(cols)


def get_row_with_min_conflicts(n, col, queens, queens_per_row, diag1, diag2):
    min_conf = float("inf")
    best_rows = []
    for row in range(n):
        conf = retrieve_conflicts(row, col, queens_per_row, diag1, diag2, n)
        if queens[col] == row:
            conf -= 3
        if conf == min_conf:
            best_rows.append(row)
        elif conf < min_conf:
            min_conf = conf
            best_rows = [row]
    return random.choice(best_rows)


def update_arrays(new_row, col, queens, queens_per_row, diag1, diag2, n):
    old_row = queens[col]
    queens_per_row[old_row] -= 1
    diag1[col - old_row + n - 1] -= 1
    diag2[col + old_row] -= 1
    queens[col] = new_row
    queens_per_row[new_row] += 1
    diag1[col - new_row + n - 1] += 1
    diag2[col + new_row] += 1


def solve_queens_board(n, queens, queens_per_row, diag1, diag2, max_steps=10_000):
    iteration = 0
    while iteration <= max_steps:
        iteration += 1
        col = get_column_with_max_conflict(n, queens, queens_per_row, diag1, diag2)
        if col == -1:
            return True
        new_row = get_row_with_min_conflicts(n, col, queens, queens_per_row, diag1, diag2)
        update_arrays(new_row, col, queens, queens_per_row, diag1, diag2, n)
    return False


def main():
    n = int(sys.stdin.readline().strip())
    if n == 1:
        print("[0]")
        return
    if n in (2, 3):
        print(-1)
        return

    start = time.time()

    success = False
    for attempt in range(50):  # до 50 рестарта
        queens, queens_per_row, diag1, diag2 = initialise_data(n)
        if attempt % 2 == 0:
            arrange_queens_on_board(n, queens, queens_per_row, diag1, diag2)
        else:
            for col in range(n):
                row = random.randint(0, n - 1)
                queens[col] = row
                queens_per_row[row] += 1
                diag1[col - row + n - 1] += 1
                diag2[col + row] += 1
        if solve_queens_board(n, queens, queens_per_row, diag1, diag2):
            success = True
            break

    if DEBUG and n <= 20:
        for r in range(n):
            print(" ".join("*" if queens[c] == r else "_" for c in range(n)))
        print(f"Time: {time.time() - start:.4f} s")

    # Изходът, който очаква тестовата система:
    if success:
        print("[" + ", ".join(str(q) for q in queens) + "]")
    else:
        print(-1)


if __name__ == "__main__":
    main()
