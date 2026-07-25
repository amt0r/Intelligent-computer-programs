import random
from config import NB, X_MIN, X_MAX


class Individual:
    def __init__(self, genotype=None):
        if genotype is not None:
            self.genotype = genotype[:]
        else:
            self.genotype = [random.randint(0, 1) for _ in range(NB)]
        self.fitness = None

    def decode(self):
        decimal_value = 0
        for bit in self.genotype:
            decimal_value = decimal_value * 2 + bit
        x = X_MIN + decimal_value * (X_MAX - X_MIN) / (2 ** NB - 1) # Інтерполяція значення x в діапазон [X_MIN, X_MAX]
        return x

    def crossover(self, other): # одноточковий кросовер
        point = random.randint(1, NB - 1)
        child_genotype = self.genotype[:point] + other.genotype[point:]
        return Individual(child_genotype)

    def mutate(self, mutation_rate=0.1):
        for i in range(NB):
            if random.random() < mutation_rate:
                self.genotype[i] = 1 - self.genotype[i]

    def copy(self):
        clone = Individual(self.genotype[:])
        clone.fitness = self.fitness
        return clone

    def __repr__(self):
        x = self.decode()
        return f"Individual(x={x:.6f}, fitness={self.fitness})"
