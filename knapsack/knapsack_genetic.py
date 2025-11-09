import enum
import random


POPULATION_SIZE = 5
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.02

def read_input():
    # M = total weight
    # N = # of items
    M, N = map(int, input().split())
    items = []

    for _ in range(N):
        w, v = map(int, input().split())
        items.append((w, v))
    return M, N, items


def generate_chromosome(n, items, max_weight):
    #[101001...]
    chromosome = [random.randint(0, 1) for _ in range(n)]
    #repair(chromosome, items, max_weight)
    return chromosome

def generate_population(size, n, items, max_weight):
    return [generate_chromosome(n, items, max_weight) for _ in range(size)]

def fitness(chromosome, items, max_weight):
    total_weight = 0
    total_value = 0

    for i, bit in enumerate(chromosome):
        if bit:
            total_weight += items[i][0]
            total_value += items[i][1]

            if total_weight > max_weight:
                total_value = 0
                break

    return total_value

def tournament_selection(population, fitnesses):
    scored_population = list(zip(population, fitnesses)) # [([1,0,1], 100), ([0,1,1], 200), ([1,1,0], 150)]
    selected = random.sample(scored_population, TOURNAMENT_SIZE)
    selected.sort(key=lambda x: x[1], reverse=True) # Сортира избраните двойки по фитнес (втората стойност в двойката).
    return selected[0][0]

def one_point_crossover(parent1, parent2, n):
    point = random.randint(1, n - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]

if __name__ == "__main__":
    M, N, items = read_input()
    population = (generate_population(POPULATION_SIZE, N, items, M))
    print(population)
    print(fitness([0, 0, 0, 1, 1], items, M))
    print(tournament_selection([[1,0,1], [0,1,1], [1,1,0]], [100, 200, 150]))
    #genetic_algorithm(M, N, items)