import random
import time

def print_board_state(queenPosition):
    n = len(queenPosition)
    for r in range(n):
        row_str = ["*" if queenPosition[c] == r else "_" for c in range(n)]
        print(" ".join(row_str))

def initialize_data(n):
    queenPosition = [0] * n
    conflictRows = [0] * n
    conflictPosDiag = [0] * (2 * n - 1)
    conflictNegDiag = [0] * (2 * n - 1)
    return queenPosition, conflictRows, conflictPosDiag, conflictNegDiag

def arrange_queens_on_board_with_pattern(n, queenPosition, conflictRows, conflictPosDiag, conflictNegDiag):
    """
    Подреждаме цариците по шаблон (по диагонал/през колона):
    колони 1, 3, 5, ... след това 0, 2, 4, ...
    """
    col = 1
    for row in range(n):
        queenPosition[col] = row
        conflictRows[row] += 1
        conflictPosDiag[row + col] += 1
        conflictNegDiag[n - 1 - row + col] += 1
        col += 2
        if col >= n:
            col = 0

def arrange_queens_on_board_random(n, queenPosition, conflictRows, conflictPosDiag, conflictNegDiag):
    """Подреждаме цариците по случаен начин на дъската"""
    for col in range(n):
        row = random.randint(0, n - 1)
        queenPosition[col] = row
        conflictRows[row] += 1
        conflictPosDiag[row + col] += 1
        conflictNegDiag[n - 1 - row + col] += 1

def count_conflicts(row, col, n, conflictRows, conflictPosDiag, conflictNegDiag):
    """O(1) изчисляване на броя конфликти за дадена позиция."""
    return (
        conflictRows[row]
        + conflictPosDiag[row + col]
        + conflictNegDiag[n - 1 - row + col]
    )

def get_column_with_max_conflicts(n, queenPosition, conflictRows, conflictPosDiag, conflictNegDiag):
    """Намира колоната(царицата) с най-много конфликти."""
    max_conflicts = -1 
    cols = []

    for col in range(n):
        row = queenPosition[col]
        current_conflicts = count_conflicts(row, col, n, conflictRows, conflictPosDiag, conflictNegDiag) - 3

        if current_conflicts == max_conflicts:
           cols.append(col)
        elif current_conflicts > max_conflicts:
           max_conflicts = current_conflicts
           cols = [col]

    if max_conflicts == 0:
        return -1 # няма конфликти -> решение
    
    return random.choice(cols)

def get_row_with_min_conflicts(n, col, queenPosition, conflictRows, conflictPosDiag, conflictNegDiag):
    """Намира редът с най-малко конфликти за дадена колона(царица)."""
    min_conflicts = float("inf")
    rows = []
   
    for row in range(n):
        current_conflicts = count_conflicts(row, col, n, conflictRows, conflictPosDiag, conflictNegDiag)
        # ако това е текущият ред на царицата, трябва да премахнем приноса ѝ към конфликтите
        if queenPosition[col] == row:
            current_conflicts -= 3

        if current_conflicts == min_conflicts:
            rows.append(row)
        elif current_conflicts < min_conflicts:
            min_conflicts = current_conflicts
            rows = [row]

    return random.choice(rows)

def update_conflict_arrays(n, new_row, col, queenPosition, conflictRows, conflictPosDiag, conflictNegDiag):
    old_row = queenPosition[col]

    conflictRows[old_row] -= 1
    conflictPosDiag[old_row + col] -= 1
    conflictNegDiag[n - 1 - old_row + col] -= 1

    queenPosition[col] = new_row

    conflictRows[new_row] += 1
    conflictPosDiag[new_row + col] += 1
    conflictNegDiag[n - 1 - new_row + col] += 1

def solve_n_queens(n, max_steps=10_000, restarts=50):
    if n == 1:
        return [0]

    if n in (2, 3):
        return -1

    for attempt in range(restarts):
        # Инициализация
        queenPosition, conflictRows, conflictPosDiag, conflictNegDiag = initialize_data(n)

        if attempt % 2 == 0:
            arrange_queens_on_board_with_pattern(n, queenPosition, conflictRows, conflictPosDiag, conflictNegDiag)
        else:
            arrange_queens_on_board_random(n, queenPosition, conflictRows, conflictPosDiag, conflictNegDiag)

        # Главен цикъл
        for _ in range(max_steps):
            current_col = get_column_with_max_conflicts(n, queenPosition, conflictRows, conflictPosDiag, conflictNegDiag)

            if current_col == -1: # няма повече конфликтни царици -> решение
                return queenPosition

            new_row_with_min_conflicts = get_row_with_min_conflicts(n, current_col, queenPosition, conflictRows, conflictPosDiag, conflictNegDiag)
            update_conflict_arrays(n, new_row_with_min_conflicts, current_col, queenPosition, conflictRows, conflictPosDiag, conflictNegDiag)

    return -1

if __name__ == "__main__":
    n = int(input().strip())
    result = solve_n_queens(n)
    print(result)
