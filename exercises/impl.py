
from collections import deque
import heapq
from multiprocessing import parent_process


def read_input():
    N, M = map(int, input().split())
    
    edges = {}
    for _ in range(M):
        u, v, c = input().split()
        c = float(c)

        if u not in edges:
            edges[u] = []

        edges[u].append((v,c))

    nodes = {}
    for _ in range(N):
        node, h = input().split()
        nodes[node] = h

    start, goal = input().split()
    l = int(input().strip())

    return edges, nodes, start, goal, l


def beam_search(edges, nodes, start, goal, l):
    frontier = [(nodes[start], start)]
    parents = {start: None}
    visited = set()

    while frontier:
        frontier.sort()
        current_heuristic, current = frontier.pop(0)

        if current == goal:
                path = []
                cur = goal

                while cur is not None:
                    path.append(cur)
                    cur = parents[cur] 

                path.reverse()

                print("Path:", " -> ".join(path))
                print("Cost:", h)
                print("Depth:", len(path) - 1)
                print("Closed:", visited)
                return

        if current in visited:
            continue
        visited.add(current)

        for nbr, cost in edges.get(current, []):
            if nbr not in visited:
                heuristic = nodes[nbr]
                frontier.append((heuristic, nbr))
        
        if len(frontier) > l:
            frontier.sort()
            frontier = frontier[:l]


def hill_climb(edges, nodes, start, goal):
    current = start
    parents = {start: None}
    visited = set()

    while True:
        if current == goal:
            path = []
            cur = goal

            while cur is not None:
                path.append(cur)
                cur = parents[cur]
            path.reverse()

            print("Hill Climb Path:", " -> ".join(path))
            print("H(goal):", nodes[goal])
            print("Depth:", len(path) - 1)
            print("Closed:", visited)
            return
            
        neighbours = edges.get(current, [])
        if not neighbours:
            print("No path found — local minimum reached.")
            return
        
        best = min(key=lambda x: nodes[x[0]])
        best_node = best[0]

        if nodes[best_node] >= nodes[current]:
            print("Stuck in local minimum at:", current)
            return
        
        parents[best_node] = current
        current = best_node
        visited.add(current)


def astar(edges, nodes, start, goal):
    pq = []
    heapq.heappush(pq, (nodes[start], start))

    parents = {start:None}
    g = {start:0}
    visited = set()

    while pq:
        f, node = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parents[cur]
            path.reverse()

            print("A* Path:", " -> ".join(path))
            print("Cost:", g[goal])
            print("Depth:", len(path) - 1)
            print("Closed:", visited)
            return

        for nbr, cost in edges.get(node, []):
            new_g = g[node] + cost

            if nbr not in g or new_g < g[nbr]:
                g[nbr] = new_g
                f_nbr = new_g + nodes[nbr]
                parents[nbr] = node
                heapq.heappush(pq, (f_nbr, nbr)) 

def bfs(edges, start, goal):
    queue = deque()
    queue.append(start)

    parents = {start: None}

    visited = set()
    visited.add(start)

    cost = {start: 0}

    while queue:
        node = queue.popleft()

        if node == goal:
            path = []
            cur = goal

            while cur is not None:
                path.append(cur)
                cur = parents[cur]

            path.reverse()

            print("BFS Path:", " -> ".join(path))
            print("Depth:", len(path) - 1)
            print("Visited:", visited)
            return
        
        for nbr, _ in edges.get(node, []):
            if nbr not in visited:
                visited.add(nbr)
                parents[nbr] = node
                cost[nbr] = cost[node] + 1
                queue.append(nbr)

def dfs(edges, start, goal):
    stack = [start]
    parents = {start: None}
    visited = set()

    while stack:
        node = stack.pop()

        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            path = []
            cur = goal

            while cur is not None:
                path.append(cur)
                cur = parents[cur]

            path.reverse()

            print("DFS Path:", " -> ".join(path))
            print("Depth:", len(path) - 1)
            print("Visited:", visited)
            return
        
        for nbr, _ in reversed(edges.get(node, [])):
            if nbr not in parents:
                parents[nbr] = node
            stack.append(nbr)
            
def dls(edges, start, goal, l):
    stack = [(start, 0)]
    parents = {start: None}
    visited = set()

    while stack:
        node, depth = stack.pop()

        if node in visited:
            continue
        visited.add(node)

        if node == goal: # return True, parents, visited
            path = []
            cur = goal

            while cur is not None:
                path.append(cur)
                cur = parents[cur]
            path.reverse()

            print("DLS Path:", " -> ".join(path))
            print("Depth:", depth)
            print("Closed:", visited)
            return

        if depth < l:
            for nbr, _ in reversed(edges.get(node, [])):
                if nbr not in parents:
                    parents[nbr] = node
                stack.append((nbr, depth + 1))

    #return False, parents, visited

def ids(edges, start, goal, max_depth):
    all_visited = set()

    for limit in range(max_depth):
        found, parents, visited = dls(edges, start, goal, limit)

        for v in visited:
            all_visited.add(v)

        if found:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parents[cur]
            path.reverse()

            print("IDS Path:", " -> ".join(path))
            print("Depth:", len(path) - 1)
            print("Closed:", all_visited)
            return

def ucs(edges, start, goal):
    pq = []
    heapq.heappush(pq, (0, start))

    parents = {start: None}
    visited = set()
    cost_to = {start: 0}

    while pq:
        cost, node = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parents[cur]
            path.reverse()

            print("UCS Path:", " -> ".join(path))
            print("Cost:", cost)
            print("Closed:", visited)
            return
        
        for nbr, edge_cost in edges.get(node, []):
            new_cost = cost + edge_cost

            if nbr not in cost_to or new_cost < cost_to[nbr]:
                cost_to[nbr] = new_cost
                parents[nbr] = node
                heapq.heappush(pq, (new_cost, nbr))

def ida_search(node, goal, edges, h, g, threshold, parents, visited):
    f = g + h[node]
    if f > threshold:
        return f, False

    if node == goal:
        return f, True

    visited.add(node)
    minimum = float("inf")

    for nbr, cost in edges.get(node, []):
        if nbr not in parents:          # избягваме цикли
            parents[nbr] = node
            new_f, found = ida_search(nbr, goal, edges, h, g + cost, threshold, parents, visited)
            
            if found:
                return new_f, True

            if new_f < minimum:
                minimum = new_f

    return minimum, False

def ida_star(start, goal, edges, h):
    threshold = h[start]

    while True:
        parents = {start: None}
        visited = set()

        new_threshold, found = ida_search(start, goal, edges, h, 0, threshold, parents, visited)

        if found:
            # възстановяване на пътя
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parents[cur]
            path.reverse()

            print("IDA* Path:", " -> ".join(path))
            print("Cost:", new_threshold)
            print("Depth:", len(path) - 1)
            print("Closed:", visited)
            return

        if new_threshold == float("inf"):
            print("No path found.")
            return

        threshold = new_threshold