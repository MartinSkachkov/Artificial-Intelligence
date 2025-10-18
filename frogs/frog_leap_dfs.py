def dfs(state, goal, path):
    if state == goal:
        return True
    
    empty_pos = state.index('_')
    moves = []
    
    # 1. Скок на жаба отляво (>) през жаба (<)
    if empty_pos >= 2 and state[empty_pos - 2] == '>' and state[empty_pos - 1] == '<':
        moves.append(empty_pos - 2)
    
    # 2. Скок на жаба отдясно (<) през жаба (>)
    if empty_pos + 2 < len(state) and state[empty_pos + 2] == '<' and state[empty_pos + 1] == '>':
        moves.append(empty_pos + 2)
    
    # 3. Просто преместване на жаба (>)
    if empty_pos >= 1 and state[empty_pos - 1] == '>':
        moves.append(empty_pos - 1)
    
    # 4. Просто преместване на жаба (<)
    if empty_pos + 1 < len(state) and state[empty_pos + 1] == '<':
        moves.append(empty_pos + 1)
    
    for from_pos in moves:
        new_state = state[:]
        new_state[from_pos], new_state[empty_pos] = new_state[empty_pos], new_state[from_pos]
        path.append(''.join(new_state))
        
        if dfs(new_state, goal, path):
            return True
        
        path.pop()
    
    return False

def solve_frog_puzzle_dfs(n):
    start = ['>'] * n + ['_'] + ['<'] * n
    goal = ['<'] * n + ['_'] + ['>'] * n
    path = [''.join(start)]
    
    dfs(start, goal, path)
    return path

import time

# Пример
N = 2
steps = solve_frog_puzzle_dfs(N)
for s in steps:
    print(s)

# Тест DFS
start_time = time.time()
steps_dfs = solve_frog_puzzle_dfs(N)
end_time = time.time()
print(f"DFS за N={N}: {len(steps_dfs)} стъпки за {end_time - start_time:.4f} сек")
