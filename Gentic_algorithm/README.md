
import random
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# PROBLEM PARAMETERS
# --------------------------------------------------

A = 150
alpha = 0.7
beta = 0.3

labour_cost = 180
capital_cost = 220
budget = 150000

# --------------------------------------------------
# GENETIC ALGORITHM PARAMETERS
# --------------------------------------------------

population_size = 100
generations = 1000

crossover_rate = 0.8
mutation_rate = 0.1

random.seed(42)
np.random.seed(42)


# --------------------------------------------------
# FITNESS FUNCTION
# --------------------------------------------------

def fitness(individual):

    x, y = individual

    # Cobb-Douglas production function
    output = A * (x ** alpha) * (y ** beta)

    # Budget constraint
    total_cost = labour_cost * x + capital_cost * y

    # Penalize solutions exceeding budget
    if total_cost > budget:
        return 0

    return output


# --------------------------------------------------
# INITIAL POPULATION
# --------------------------------------------------

def create_population():

    population = []

    for i in range(population_size):

        # Random allocation of budget
        labour_share = random.random()

        x = (budget * labour_share) / labour_cost

        y = (budget * (1 - labour_share)) / capital_cost

        population.append([x, y])

    return population


# --------------------------------------------------
# SELECTION
# --------------------------------------------------

def selection(population):

    candidates = random.sample(population, 4)

    best = max(candidates, key=fitness)

    return best


# --------------------------------------------------
# CROSSOVER
# --------------------------------------------------

def crossover(parent1, parent2):

    if random.random() < crossover_rate:

        alpha_value = random.random()

        child1 = [
            alpha_value * parent1[0]
            + (1 - alpha_value) * parent2[0],

            alpha_value * parent1[1]
            + (1 - alpha_value) * parent2[1]
        ]

        child2 = [
            alpha_value * parent2[0]
            + (1 - alpha_value) * parent1[0],

            alpha_value * parent2[1]
            + (1 - alpha_value) * parent1[1]
        ]

        return child1, child2

    return parent1[:], parent2[:]


# --------------------------------------------------
# MUTATION
# --------------------------------------------------

def mutation(individual):

    child = individual[:]

    if random.random() < mutation_rate:

        child[0] += random.uniform(-50, 50)

        child[1] += random.uniform(-20, 20)

    # Ensure non-negative values
    child[0] = max(0, child[0])
    child[1] = max(0, child[1])

    # Repair budget constraint
    total_cost = (
        labour_cost * child[0]
        + capital_cost * child[1]
    )

    if total_cost > budget:

        scale = budget / total_cost

        child[0] *= scale
        child[1] *= scale

    return child


# --------------------------------------------------
# GENETIC ALGORITHM
# --------------------------------------------------

def genetic_algorithm():

    population = create_population()

    best_history = []

    for generation in range(generations):

        population.sort(
            key=fitness,
            reverse=True
        )

        # Preserve best individual (elitism)
        new_population = [population[0][:]]

        while len(new_population) < population_size:

            parent1 = selection(population)
            parent2 = selection(population)

            child1, child2 = crossover(
                parent1,
                parent2
            )

            child1 = mutation(child1)
            child2 = mutation(child2)

            new_population.append(child1)

            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

        best = max(population, key=fitness)

        best_history.append(fitness(best))

        if generation % 100 == 0:

            print(
                "Generation:",
                generation,
                "Best Output:",
                round(fitness(best), 4)
            )

    best_solution = max(population, key=fitness)

    return best_solution, best_history


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

best_solution, history = genetic_algorithm()

x = best_solution[0]
y = best_solution[1]

z = fitness(best_solution)

total_budget = labour_cost * x + capital_cost * y

remaining_budget = budget - total_budget


# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

print("\nFINAL OPTIMIZATION RESULTS")
print("--------------------------------")

print("Optimal Labour (x):", round(x, 4))

print("Optimal Capital (y):", round(y, 4))

print("Maximum Output (Z):", round(z, 4))

print("Total Budget:", budget)

print("Budget Used:", round(total_budget, 2))

print("Budget Remaining:", round(remaining_budget, 2))


# --------------------------------------------------
# CONVERGENCE GRAPH
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, generations + 1),
    history,
    label="Best Output"
)

plt.title("Genetic Algorithm Convergence")

plt.xlabel("Generation")

plt.ylabel("Best Production Output")

plt.grid(True)

plt.legend()

plt.show()
<img width="1352" height="753" alt="Screenshot 2026-11-27 171718" src="https://github.com/user-attachments/assets/6ca214cd-bdd9-4e7e-ae27-668a44683cc0" />

