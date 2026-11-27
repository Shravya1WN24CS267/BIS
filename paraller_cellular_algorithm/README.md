
import random
import heapq

# ==========================================================
# GRASP-CPA: Cellular Processing Algorithm
# Application: Task Scheduling on Heterogeneous Processors
# Objective: Minimize Makespan
# ==========================================================

random.seed(42)

# -------------------- PROBLEM DATA -------------------------

# Tasks: A, B, C, D, E, F
TASKS = ["A", "B", "C", "D", "E", "F"]

# Precedence constraints (Directed Acyclic Graph)
# A must finish before B and C, etc.
PRECEDENCE = {
    "A": [],
    "B": ["A"],
    "C": ["A"],
    "D": ["B"],
    "E": ["B", "C"],
    "F": ["D", "E"]
}

# Processing time of each task on each machine
# Rows = tasks, Columns = machines
PROCESSING_TIME = {
    "A": [4, 3, 5],
    "B": [6, 4, 3],
    "C": [5, 7, 4],
    "D": [3, 6, 5],
    "E": [4, 3, 6],
    "F": [7, 5, 4]
}

# Communication cost between dependent tasks
COMMUNICATION_COST = 2

NUM_MACHINES = 3
NUM_PCELLS = 3
CPA_ITERATIONS = 30
ILS_ITERATIONS = 5


# -------------------- GRASP -------------------------------

def grasp_construct():

    order = []
    remaining = set(TASKS)

    alpha = random.uniform(0.9, 1.0)

    while remaining:

        # Find tasks whose predecessors are completed
        ready = [
            task for task in remaining
            if all(parent in order
                   for parent in PRECEDENCE[task])
        ]

        # Greedy value: average processing time
        values = {
            task: sum(PROCESSING_TIME[task])
                  / NUM_MACHINES
            for task in ready
        }

        minimum = min(values.values())
        maximum = max(values.values())

        threshold = minimum + alpha * (
            maximum - minimum
        )

        # Restricted Candidate List
        rcl = [
            task for task in ready
            if values[task] <= threshold
        ]

        selected = random.choice(rcl)

        order.append(selected)
        remaining.remove(selected)

    return order


# -------------------- SCHEDULE EVALUATION ------------------

def calculate_makespan(order, assignment):

    machine_available = [0] * NUM_MACHINES

    start_time = {}
    finish_time = {}

    for task in order:

        machine = assignment[task]

        # Task cannot start before its predecessors finish
        predecessor_finish = 0

        for parent in PRECEDENCE[task]:

            communication = 0

            if assignment[parent] != machine:
                communication = COMMUNICATION_COST

            arrival = (
                finish_time[parent] + communication
            )

            predecessor_finish = max(
                predecessor_finish,
                arrival
            )

        start = max(
            machine_available[machine],
            predecessor_finish
        )

        duration = PROCESSING_TIME[task][machine]

        finish = start + duration

        start_time[task] = start
        finish_time[task] = finish

        machine_available[machine] = finish

    makespan = max(finish_time.values())

    return makespan, start_time, finish_time


# -------------------- INITIAL SOLUTION --------------------

def create_initial_solution(order):

    assignment = {}

    # Assign each task to the machine giving
    # the earliest finish time
    for task in order:

        best_machine = 0
        best_finish = float("inf")

        for machine in range(NUM_MACHINES):

            temporary = assignment.copy()
            temporary[task] = machine

            # Evaluate partial assignment
            finish = PROCESSING_TIME[task][machine]

            if finish < best_finish:
                best_finish = finish
                best_machine = machine

        assignment[task] = best_machine

    return assignment


# -------------------- LOCAL SEARCH ------------------------

def local_search(order, assignment):

    current = assignment.copy()

    current_cost = calculate_makespan(
        order, current
    )[0]

    improved = True

    while improved:

        improved = False

        for task in order:

            original_machine = current[task]

            for machine in range(NUM_MACHINES):

                if machine == original_machine:
                    continue

                candidate = current.copy()
                candidate[task] = machine

                candidate_cost = calculate_makespan(
                    order, candidate
                )[0]

                if candidate_cost < current_cost:

                    current = candidate
                    current_cost = candidate_cost
                    improved = True
                    break

            if improved:
                break

    return current


# -------------------- ITERATED LOCAL SEARCH ---------------

def ils(order, current_assignment, iterations):

    current = current_assignment.copy()

    best = current.copy()

    best_cost = calculate_makespan(
        order, best
    )[0]

    for _ in range(iterations):

        # Perturbation: change machine assignment
        candidate = current.copy()

        number_of_changes = random.randint(1, 2)

        for _ in range(number_of_changes):

            task = random.choice(order)

            candidate[task] = random.randint(
                0, NUM_MACHINES - 1
            )

        # Improve perturbed solution
        candidate = local_search(
            order, candidate
        )

        candidate_cost = calculate_makespan(
            order, candidate
        )[0]

        if candidate_cost < best_cost:

            best = candidate.copy()
            best_cost = candidate_cost

        # Continue ILS from perturbed local optimum
        current = candidate.copy()

    return best, best_cost


# -------------------- CPA COMMUNICATION -------------------

def communicate(parent1, parent2, order):

    # Single-point crossover of machine assignments
    point = random.randint(1, len(order) - 1)

    child1 = {}
    child2 = {}

    for index, task in enumerate(order):

        if index < point:

            child1[task] = parent1[task]
            child2[task] = parent2[task]

        else:

            child1[task] = parent2[task]
            child2[task] = parent1[task]

    return child1, child2


# -------------------- CELLULAR PROCESSING ALGORITHM ------

def cpa():

    print("Starting GRASP-CPA...")
    print("Application: Heterogeneous Task Scheduling\n")

    # Phase 1: GRASP
    order = grasp_construct()

    print("GRASP Task Order:", order)

    initial_assignment = create_initial_solution(order)

    initial_cost = calculate_makespan(
        order, initial_assignment
    )[0]

    print("Initial Makespan:", initial_cost)

    # Initialize 3 Processing Cells
    pcells = []

    for i in range(NUM_PCELLS):

        pcells.append({
            "current": initial_assignment.copy(),
            "best": initial_assignment.copy(),
            "best_cost": initial_cost
        })

    global_best = initial_assignment.copy()
    global_best_cost = initial_cost

    convergence = []

    # Phase 2: Cellular Processing Algorithm
    for iteration in range(CPA_ITERATIONS):

        print(
            f"\nCPA Iteration {iteration + 1}"
        )

        # Each Processing Cell runs ILS
        for i in range(NUM_PCELLS):

            result, cost = ils(
                order,
                pcells[i]["current"],
                ILS_ITERATIONS
            )

            pcells[i]["current"] = result.copy()

            if cost < pcells[i]["best_cost"]:

                pcells[i]["best"] = result.copy()
                pcells[i]["best_cost"] = cost

            if cost < global_best_cost:

                global_best = result.copy()
                global_best_cost = cost

        # Communication between Processing Cells
        child1, child2 = communicate(
            pcells[0]["best"],
            pcells[1]["best"],
            order
        )

        pcells[0]["current"] = child1
        pcells[1]["current"] = child2

        child3, child4 = communicate(
            pcells[0]["best"],
            pcells[2]["best"],
            order
        )

        pcells[1]["current"] = child3
        pcells[2]["current"] = child4

        convergence.append(global_best_cost)

        print(
            "Best Makespan:",
            global_best_cost
        )

    return (
        order,
        global_best,
        global_best_cost,
        convergence
    )


# -------------------- MAIN PROGRAM ------------------------

if __name__ == "__main__":

    order, assignment, makespan, convergence = cpa()

    print("\n========== FINAL RESULT ==========")

    print("Best Task Order:", order)

    print("\nTask Machine Assignment:")

    for task in order:

        machine = assignment[task]

        print(
            task,
            "-> Machine",
            machine + 1
        )

    print("\nMinimum Makespan:", makespan)

    # Display task start and finish times
    _, starts, finishes = calculate_makespan(
        order, assignment
    )

    print("\nTask Schedule:")

    for task in order:

        print(
            task,
            "Start:",
            starts[task],
            "Finish:",
            finishes[task]
        )

    print("\nCPA completed successfully!")
<img width="1843" height="937" alt="Screenshot 2026-11-27 175331" src="https://github.com/user-attachments/assets/4c872408-0f8a-4ca9-9a3a-37f6460e9b6f" />

