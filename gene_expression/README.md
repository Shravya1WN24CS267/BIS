
import random
import math

# ------------------------------------------------
# Gene Expression Programming (GEP)
# Symbolic Regression: y = x^2 + x + 1
# ------------------------------------------------

random.seed(42)

# Training data from your notebook
X = [1, 2, 3, 4, 5]
Y = [3, 7, 13, 21, 31]

# GEP parameters
POPULATION_SIZE = 100
GENERATIONS = 100
HEAD_LENGTH = 5

# Binary functions
FUNCTIONS = ["+", "-", "*", "/"]

# Terminals
TERMINALS = ["x", "1", "2", "3"]

# GEP head can contain functions and terminals
HEAD = FUNCTIONS + TERMINALS

# GEP tail contains terminals only
TAIL_LENGTH = HEAD_LENGTH + 1


# ------------------------------------------------
# Generate chromosome
# ------------------------------------------------

def create_chromosome():
    head = [
        random.choice(HEAD)
        for _ in range(HEAD_LENGTH)
    ]

    tail = [
        random.choice(TERMINALS)
        for _ in range(TAIL_LENGTH)
    ]

    return head + tail


# ------------------------------------------------
# Decode chromosome into expression
# ------------------------------------------------

def decode(chromosome):
    expression = []
    pending = 1

    for gene in chromosome:
        if pending == 0:
            break

        expression.append(gene)

        if gene in FUNCTIONS:
            pending += 1
        else:
            pending -= 1

    return expression


# ------------------------------------------------
# Evaluate expression
# ------------------------------------------------

def evaluate_prefix(expression, x):
    position = [0]

    def calculate():
        if position[0] >= len(expression):
            return 0

        symbol = expression[position[0]]
        position[0] += 1

        if symbol == "x":
            return x

        if symbol in ["1", "2", "3"]:
            return int(symbol)

        left = calculate()
        right = calculate()

        if symbol == "+":
            return left + right

        if symbol == "-":
            return left - right

        if symbol == "*":
            return left * right

        if symbol == "/":
            if abs(right) < 0.000001:
                return 1e6
            return left / right

        return 0

    try:
        result = calculate()

        if not math.isfinite(result):
            return 1e6

        return result

    except Exception:
        return 1e6


# ------------------------------------------------
# Convert prefix expression to readable expression
# ------------------------------------------------

def expression_to_string(expression):
    position = [0]

    def build():
        if position[0] >= len(expression):
            return "0"

        symbol = expression[position[0]]
        position[0] += 1

        if symbol not in FUNCTIONS:
            return symbol

        left = build()
        right = build()

        return f"({left} {symbol} {right})"

    return build()


# ------------------------------------------------
# Fitness calculation
# ------------------------------------------------

def fitness(chromosome):
    expression = decode(chromosome)

    total_error = 0

    for x, actual in zip(X, Y):
        predicted = evaluate_prefix(expression, x)

        error = abs(actual - predicted)

        total_error += error

    # Higher fitness means a better solution
    return 1 / (1 + total_error)


# ------------------------------------------------
# Selection using tournament method
# ------------------------------------------------

def selection(population):
    candidates = random.sample(population, 3)

    return max(candidates, key=fitness).copy()


# ------------------------------------------------
# Mutation
# ------------------------------------------------

def mutation(chromosome, mutation_rate=0.1):
    chromosome = chromosome.copy()

    for i in range(len(chromosome)):

        if random.random() < mutation_rate:

            if i < HEAD_LENGTH:
                chromosome[i] = random.choice(HEAD)
            else:
                chromosome[i] = random.choice(TERMINALS)

    return chromosome


# ------------------------------------------------
# One-point crossover
# ------------------------------------------------

def crossover(parent1, parent2):
    point = random.randint(
        1, len(parent1) - 1
    )

    child1 = (
        parent1[:point] + parent2[point:]
    )

    child2 = (
        parent2[:point] + parent1[point:]
    )

    return child1, child2


# ------------------------------------------------
# Main GEP algorithm
# ------------------------------------------------

population = [
    create_chromosome()
    for _ in range(POPULATION_SIZE)
]

best_chromosome = None
best_fitness = 0

for generation in range(1, GENERATIONS + 1):

    population.sort(
        key=fitness,
        reverse=True
    )

    current_best = population[0]
    current_fitness = fitness(current_best)

    if current_fitness > best_fitness:
        best_fitness = current_fitness
        best_chromosome = current_best.copy()

    if generation == 1 or generation % 10 == 0:

        expression = expression_to_string(
            decode(best_chromosome)
        )

        print(
            f"Generation {generation}: "
            f"Best expression = {expression}, "
            f"Fitness = {best_fitness:.4f}"
        )

    # Stop if exact solution is found
    if best_fitness == 1.0:
        break

    new_population = []

    # Keep best individual (elitism)
    new_population.append(
        best_chromosome.copy()
    )

    while len(new_population) < POPULATION_SIZE:

        parent1 = selection(population)
        parent2 = selection(population)

        child1, child2 = crossover(
            parent1, parent2
        )

        child1 = mutation(child1)
        child2 = mutation(child2)

        new_population.append(child1)

        if len(new_population) < POPULATION_SIZE:
            new_population.append(child2)

    population = new_population


# ------------------------------------------------
# Final result
# ------------------------------------------------

best_expression = expression_to_string(
    decode(best_chromosome)
)

print("\nOptimization Completed")
print("Best Expression:", best_expression)
print("Best Fitness:", round(best_fitness, 4))

print("\nInput | Actual | Predicted")

for x, actual in zip(X, Y):

    predicted = evaluate_prefix(
        decode(best_chromosome), x
    )

    print(
        f"{x:5} | {actual:6} | "
        f"{predicted:9.2f}"
    )


