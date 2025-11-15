import time
import random

POPULATION_SIZE = 200
GENERATIONS = 100
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.02
ELITE_COUNT = 5

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
    repair(chromosome, items, max_weight)
    return chromosome

def generate_population(size, n, items, max_weight):
    return [generate_chromosome(n, items, max_weight) for _ in range(size)]

def repair(chromosome, items, max_weight):
    # Поправя хромозомата, като маха най-неефективните предмети
    total_weight = sum(items[i][0] for i, bit in enumerate(chromosome) if bit)

    # Ако вече е валидна — нищо не правим
    if total_weight <= max_weight:
        return

    # Взимаме всички индекси на включени предмети
    indexed = [(i, items[i][1] / items[i][0]) for i, bit in enumerate(chromosome) if bit]
    indexed.sort(key=lambda x: x[1])  # най-нисък ratio първо
    
    for i, _ in indexed:
        chromosome[i] = 0
        total_weight -= items[i][0]
        if total_weight <= max_weight:
            break

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
    # Взимаме TOURNAMENT_SIZE произволни индивиди (с техните фитнеси) и избираме най-добрия
    scored_population = list(zip(population, fitnesses)) # [([1,0,1], 100), ([0,1,1], 200), ([1,1,0], 150)]
    selected = random.sample(scored_population, TOURNAMENT_SIZE)
    selected.sort(key=lambda x: x[1], reverse=True)
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

def genetic_algorithm(M, N, items):
    start_time = time.time()
    population = generate_population(POPULATION_SIZE, N, items, M) 
    best_values = []

    for gen in range(GENERATIONS):
        fitnesses = [fitness(chromosome, items, M) for chromosome in population]
        new_population = []

        # вземи първите k най-добри индивида от популацията
        elite = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:ELITE_COUNT]
        elite_chromosomes = [e[0] for e in elite]
        new_population.extend(elite_chromosomes)

        # допълни остатъка от популация с останали индивиди чрез tournament_selection
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = one_point_crossover(parent1, parent2, N)
            mutate(child1)
            mutate(child2)
            repair(child1, items, M)
            repair(child2, items, M)

            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population
        fitnesses = [fitness(chromosome, items, M) for chromosome in population]
        best_chromosome_value = max(fitnesses)
            
        # Оценка и логване на напредъка
        if gen == 0 or gen == GENERATIONS - 1 or gen % (GENERATIONS // 9) == 0:
            print(best_chromosome_value)
            best_values.append(best_chromosome_value)
   
    print()
    print(best_values[-1])
    print("Time:", round(time.time() - start_time, 3), "seconds")

if __name__ == "__main__":
    M, N, items = read_input()
    genetic_algorithm(M, N, items)