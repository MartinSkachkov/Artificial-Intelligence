import math

X = 'X'
O = 'O'
EMPTY = '_'

def terminal(state):
    """Проверява дали състоянието е терминално (победа или равенство)."""
    # Проверка за победа
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
    # Проверка за равенство (няма празни клетки)
    for row in state:
        if EMPTY in row:
            return False
    return True

def winner(state):
    """Връща победителя (X, O) или None ако няма."""
    lines = [
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        [state[0][0], state[1][0], state[2][0]],
        [state[0][1], state[1][1], state[2][1]],
        [state[0][2], state[1][2], state[2][2]],
        [state[0][0], state[1][1], state[2][2]],
        [state[0][2], state[1][1], state[2][0]],
    ]
    for line in lines:
        if line[0] != EMPTY and line[0] == line[1] == line[2]:
            return line[0]
    return None

def value(state, depth=0):
    """Връща стойността на терминално състояние с отчитане на дълбочината."""
    w = winner(state)
    if w == X:
        return 10 - depth  # По-бърза победа = по-висока стойност
    elif w == O:
        return depth - 10  # По-бърза победа за O = по-ниска стойност
    return 0  # Равенство

def player(state):
    """Определя кой е на ход - X винаги започва първи."""
    x_count = sum(row.count(X) for row in state)
    o_count = sum(row.count(O) for row in state)
    return X if x_count == o_count else O

def actions(state):
    """Връща списък от възможни ходове (row, col) - 0-базирана индексация."""
    moves = []
    for i in range(3):
        for j in range(3):
            if state[i][j] == EMPTY:
                moves.append((i, j))
    return moves

def result(state, action):
    """Прилага ход и връща ново състояние."""
    i, j = action
    new_state = [row[:] for row in state]  # Копие на дъската
    new_state[i][j] = player(state)
    return new_state

def minimax(state, depth, alpha, beta, maximizing):
    """Minimax с alpha-beta отсичане."""
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
                break  # Beta отсичане
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
                break  # Alpha отсичане
        return min_eval, best_action

def best_move(state):
    """Намира най-добрия ход за текущия играч."""
    current = player(state)
    maximizing = (current == X)
    _, action = minimax(state, 0, -math.inf, math.inf, maximizing)
    return action

def parse_board(lines):
    """Парсва дъската от рамкиран формат."""
    state = []
    for line in lines:
        if '|' in line:
            cells = line.split('|')[1:-1]  # Премахваме първия и последния празен елемент
            row = [cell.strip() for cell in cells]
            state.append(row)
    return state

def print_board(state):
    """Отпечатва дъската в рамкиран формат."""
    print("+---+---+---+")
    for row in state:
        print(f"| {row[0]} | {row[1]} | {row[2]} |")
        print("+---+---+---+")

def judge_mode():
    """Режим JUDGE - връща оптимален ход или -1 за терминална позиция."""
    turn_line = input().strip()
    turn = turn_line.split()[1]  # X или O
    
    # Четем 7 реда за дъската
    board_lines = [input() for _ in range(7)]
    state = parse_board(board_lines)
    
    if terminal(state):
        print(-1)
        return
    
    action = best_move(state)
    if action:
        print(f"{action[0] + 1} {action[1] + 1}")  # 1-базирана индексация
    else:
        print(-1)

def game_mode():
    """Режим GAME - интерактивна игра човек-компютър."""
    first_line = input().strip()
    first_player = first_line.split()[1]  # Кой започва
    
    human_line = input().strip()
    human = human_line.split()[1]  # Коя страна е човекът
    
    agent = O if human == X else X
    
    # Четем 7 реда за началната дъска
    board_lines = [input() for _ in range(7)]
    state = parse_board(board_lines)
    
    # Определяме кой е на ход в началото
    current = player(state)
    
    while not terminal(state):
        if current == human:
            # Ход на човека
            move = input().strip().split()
            row, col = int(move[0]) - 1, int(move[1]) - 1  # 0-базирана индексация
            
            if state[row][col] == EMPTY:
                state[row][col] = human
                print_board(state)
            else:
                continue  # Невалиден ход, чакаме нов
        else:
            # Ход на агента
            action = best_move(state)
            state[action[0]][action[1]] = agent
            print_board(state)
        
        current = player(state)
    
    # Отпечатваме резултата
    w = winner(state)
    if w:
        print(f"WINNER: {w}")
    else:
        print("DRAW")

def main():
    mode = input().strip()
    if mode == "JUDGE":
        judge_mode()
    elif mode == "GAME":
        game_mode()

if __name__ == "__main__":
    main()