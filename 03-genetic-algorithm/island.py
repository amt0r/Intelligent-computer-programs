import random
from individual import Individual
from config import target_function


def evolve_island_wrapper(args):
    island_obj, island_idx, population, generations, mode = args
    return island_obj.evolve_island(island_idx, population, generations, mode)


class Island:
    def __init__(self, population_size, mutation_rate, elitism_rate=0.1, tournament_size=3):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism_count = max(1, int(population_size * elitism_rate))
        self.tournament_size = tournament_size

    def initialize_population(self):
        return [Individual() for _ in range(self.population_size)]

    def evaluate_population(self, population, mode="max"):
        for ind in population:
            x = ind.decode()
            y = target_function(x)
            if mode == "max":
                ind.fitness = y
            else:
                ind.fitness = -y

    # потрібна для того, щоб підтримувати в популяції «середнячків» із нестандартними генами.
    def tournament_selection(self, population):
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda ind: ind.fitness)

    def select_parents(self, population):
        parents = []
        for _ in range(self.population_size):
            parent = self.tournament_selection(population)
            parents.append(parent)
        return parents

    def create_offspring(self, parents):
        offspring = []
        random.shuffle(parents)
        for i in range(0, len(parents) - 1, 2):
            child1 = parents[i].crossover(parents[i + 1])
            child2 = parents[i + 1].crossover(parents[i])
            child1.mutate(self.mutation_rate)
            child2.mutate(self.mutation_rate)
            offspring.append(child1)
            offspring.append(child2)
        return offspring

    def select_next_generation(self, population, offspring):
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        elites = population[:self.elitism_count]
        
        offspring.sort(key=lambda ind: ind.fitness, reverse=True)
        
        rest_count = self.population_size - self.elitism_count
        rest = offspring[:rest_count]
        
        return elites + rest
        
    def evolve_island(self, island_idx, population, max_generations, mode="max"):
        self.evaluate_population(population, mode)

        best_individual = None
        best_fitness = float('-inf')

        for generation in range(max_generations):
            parents = self.select_parents(population)
            offspring = self.create_offspring(parents)
            self.evaluate_population(offspring, mode)
            population = self.select_next_generation(population, offspring)

            current_best = max(population, key=lambda ind: ind.fitness)

            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                best_individual = current_best.copy()

            if generation % 20 == 0 or generation == max_generations - 1:
                x = current_best.decode()
                y = target_function(x)
                print(f"  Island {island_idx} | Gen {generation:>4d} | x = {x:.6f} | Y(x) = {y:.6f}")

        return best_individual

    def run(self, islands_count, max_generations, mode="max"):
        from concurrent.futures import ProcessPoolExecutor

        island_populations = [self.initialize_population() for _ in range(islands_count)]
        args = [(self, i, island_populations[i], max_generations, mode) for i in range(islands_count)]

        with ProcessPoolExecutor(max_workers=islands_count) as executor:
            results = list(executor.map(evolve_island_wrapper, args))

        best = max(results, key=lambda ind: ind.fitness)
        return best
