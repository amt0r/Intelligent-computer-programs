import time
from config import target_function, X_MIN, X_MAX, NB, get_display_formula
from island import Island
from visualizer import Visualizer


def main():
    print("=" * 60)
    print("  Genetic Algorithm (Island Model)")
    print(f"  Y(x) = {get_display_formula()}, x in [{X_MIN}, {X_MAX}]")
    print(f"  Bits for encoding: NB = {NB}")
    print("=" * 60)

    population_size = int(input("Population size (default 3): ").strip() or "3")
    mutation_rate = float(input("Mutation rate (default 0.15): ").strip() or "0.15")
    elitism_rate = float(input("Elitism rate (default 0.1): ").strip() or "0.1")
    tournament_size = int(input("Tournament size (default 3): ").strip() or "3")
    islands_count = int(input("Number of islands (default 12): ").strip() or "12")
    max_generations = int(input("Number of generations (default 60): ").strip() or "60")

    print(f"\nParams: pop={population_size}, mut={mutation_rate}, "
          f"elit={elitism_rate}, tourn={tournament_size}, "
          f"islands={islands_count}, gens={max_generations}\n")

    island_model = Island(population_size, mutation_rate, elitism_rate, tournament_size)

    print("=" * 60)
    print("  SEARCH FOR MAXIMUM (fitness = Y(x))")
    print("=" * 60)
    start = time.time()
    best_max = island_model.run(islands_count, max_generations, mode="max")
    time_max = time.time() - start

    x_max = best_max.decode()
    y_max = target_function(x_max)
    print(f"\nMaximum: x = {x_max:.6f}, Y(x) = {y_max:.6f}")
    print(f"Time: {time_max:.2f} s\n")

    print("=" * 60)
    print("  SEARCH FOR MINIMUM (fitness = -Y(x))")
    print("=" * 60)
    start = time.time()
    best_min = island_model.run(islands_count, max_generations, mode="min")
    time_min = time.time() - start

    x_min = best_min.decode()
    y_min = target_function(x_min)
    print(f"\nMinimum: x = {x_min:.6f}, Y(x) = {y_min:.6f}")
    print(f"Time: {time_min:.2f} s\n")

    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Maximum: x = {x_max:.6f},  Y(x) = {y_max:.6f}")
    print(f"  Minimum: x = {x_min:.6f},  Y(x) = {y_min:.6f}")
    print(f"  Total time: {time_max + time_min:.2f} s")
    print("=" * 60)

    vis = Visualizer()
    vis.plot_results(best_max, best_min)


if __name__ == "__main__":
    main()
