
import random
import math

# -----------------------------------------
# Ant Colony Optimization for TSP
# -----------------------------------------

random.seed(42)

# City coordinates
cities = {
    0: (10, 20),
    1: (30, 40),
    2: (50, 10),
    3: (60, 30),
    4: (20, 60),
    5: (70, 70),
    6: (40, 80),
    7: (80, 20)
}

NUM_CITIES = len(cities)
NUM_ANTS = 20
NUM_ITERATIONS = 100

ALPHA = 1.0       # Pheromone importance
BETA = 2.0        # Distance importance
EVAPORATION = 0.5
Q = 100


# Calculate distance between two cities
def distance(city1, city2):
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Create distance matrix
dist_matrix = [
    [
        distance(i, j) if i != j else 0
        for j in range(NUM_CITIES)
    ]
    for i in range(NUM_CITIES)
]


# Initialize pheromone matrix
pheromone = [
    [1.0 for _ in range(NUM_CITIES)]
    for _ in range(NUM_CITIES)
]


# Calculate total route distance
def calculate_route_distance(route):
    total = 0

    for i in range(len(route) - 1):
        total += dist_matrix[route[i]][route[i + 1]]

    # Return to starting city
    total += dist_matrix[route[-1]][route[0]]

    return total


# Select next city using probability
def select_next_city(current_city, unvisited):
    probabilities = []

    for city in unvisited:
        pheromone_value = pheromone[current_city][city]
        heuristic_value = 1 / dist_matrix[current_city][city]

        probability = (
            (pheromone_value ** ALPHA)
            * (heuristic_value ** BETA)
        )

        probabilities.append(probability)

    total = sum(probabilities)

    probabilities = [
        p / total for p in probabilities
    ]

    return random.choices(
        unvisited,
        weights=probabilities,
        k=1
    )[0]


# Construct route for one ant
def construct_route():
    start_city = random.randint(0, NUM_CITIES - 1)

    route = [start_city]

    unvisited = list(range(NUM_CITIES))
    unvisited.remove(start_city)

    current_city = start_city

    while unvisited:
        next_city = select_next_city(
            current_city, unvisited
        )

        route.append(next_city)

        unvisited.remove(next_city)

        current_city = next_city

    return route


# Update pheromones
def update_pheromones(routes):
    global pheromone

    # Evaporation
    for i in range(NUM_CITIES):
        for j in range(NUM_CITIES):
            pheromone[i][j] *= (1 - EVAPORATION)

    # Deposit pheromones
    for route, route_distance in routes:
        deposit = Q / route_distance

        for i in range(len(route)):
            city1 = route[i]
            city2 = route[(i + 1) % NUM_CITIES]

            pheromone[city1][city2] += deposit
            pheromone[city2][city1] += deposit


# -----------------------------------------
# Main ACO Algorithm
# -----------------------------------------

best_route = None
best_distance = float("inf")

print("AHACO / ACO - TSP Optimization")
print("--------------------------------")

for iteration in range(1, NUM_ITERATIONS + 1):

    ant_routes = []

    for ant in range(NUM_ANTS):

        route = construct_route()

        route_distance = calculate_route_distance(
            route
        )

        ant_routes.append(
            (route, route_distance)
        )

        if route_distance < best_distance:
            best_distance = route_distance
            best_route = route.copy()

    update_pheromones(ant_routes)

    if iteration == 1 or iteration % 10 == 0:
        print(
            f"Iteration {iteration}: "
            f"Best Distance = {best_distance:.2f}"
        )


# -----------------------------------------
# Final Output
# -----------------------------------------

print("\nOptimization Completed!")
print("Best Distance:", round(best_distance, 2))

print("Best Route:")

route_names = [
    f"City {city + 1}"
    for city in best_route
]

route_names.append(route_names[0])

print(" -> ".join(route_names))

