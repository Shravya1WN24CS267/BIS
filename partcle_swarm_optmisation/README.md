
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. PROBLEM PARAMETERS
# =====================================================

NUM_TASKS = 10
NUM_VMS = 3

# Task workload in Million Instructions (MI)
task_sizes = np.array([
    10, 25, 50, 100, 150,
    20, 35, 60, 120, 140
], dtype=float)

# VM processing speed in MIPS
vm_speeds = np.array([
    100, 150, 200
], dtype=float)

# Cost per execution time unit
vm_costs = np.array([
    0.20, 0.40, 0.62
], dtype=float)

DEADLINE = 3000

# =====================================================
# 2. PSO PARAMETERS
# =====================================================

SWARM_SIZE = 50
ITERATIONS = 1000

W = 0.7       # Inertia weight
C1 = 1.5      # Cognitive coefficient
C2 = 1.5      # Social coefficient

random_seed = 42
np.random.seed(random_seed)

# =====================================================
# 3. EXECUTION TIME
# =====================================================

def execution_time(task, vm):

    return task_sizes[task] / vm_speeds[vm]


# =====================================================
# 4. CONVERT PARTICLE POSITION TO VM ALLOCATION
# =====================================================

def decode_particle(position):

    allocation = np.rint(position).astype(int)

    allocation = np.clip(
        allocation, 0, NUM_VMS - 1
    )

    return allocation


# =====================================================
# 5. CALCULATE EXECUTION COST
# =====================================================

def calculate_cost(allocation):

    total_cost = 0

    for task in range(NUM_TASKS):

        vm = allocation[task]

        time = execution_time(task, vm)

        cost = time * vm_costs[vm]

        total_cost += cost

    return total_cost


# =====================================================
# 6. CALCULATE MAKESPAN
# =====================================================

def calculate_makespan(allocation):

    vm_load = np.zeros(NUM_VMS)

    for task in range(NUM_TASKS):

        vm = allocation[task]

        time = execution_time(task, vm)

        vm_load[vm] += time

    makespan = np.max(vm_load)

    return makespan, vm_load


# =====================================================
# 7. FITNESS FUNCTION
# =====================================================

def fitness(position):

    allocation = decode_particle(position)

    cost = calculate_cost(allocation)

    makespan, _ = calculate_makespan(allocation)

    # Deadline penalty
    penalty = max(0, makespan - DEADLINE)

    total_fitness = cost + penalty * 0.01

    return total_fitness


# =====================================================
# 8. INITIALIZE PARTICLES
# =====================================================

positions = np.random.uniform(
    0, NUM_VMS - 1,
    (SWARM_SIZE, NUM_TASKS)
)

velocities = np.random.uniform(
    -1, 1,
    (SWARM_SIZE, NUM_TASKS)
)

# Personal best positions
pbest_positions = positions.copy()

pbest_fitness = np.array([
    fitness(position)
    for position in positions
])

# Global best position
gbest_index = np.argmin(pbest_fitness)

gbest_position = pbest_positions[
    gbest_index
].copy()

gbest_fitness = pbest_fitness[
    gbest_index
]


# =====================================================
# 9. PSO OPTIMIZATION
# =====================================================

fitness_history = []
cost_history = []
makespan_history = []

for iteration in range(ITERATIONS):

    for i in range(SWARM_SIZE):

        current_fitness = fitness(
            positions[i]
        )

        # Update personal best
        if current_fitness < pbest_fitness[i]:

            pbest_fitness[i] = current_fitness

            pbest_positions[i] = positions[i].copy()

        # Update global best
        if current_fitness < gbest_fitness:

            gbest_fitness = current_fitness

            gbest_position = positions[i].copy()

    # Update velocity and position
    for i in range(SWARM_SIZE):

        r1 = np.random.random(NUM_TASKS)
        r2 = np.random.random(NUM_TASKS)

        velocities[i] = (
            W * velocities[i]
            + C1 * r1 * (
                pbest_positions[i] - positions[i]
            )
            + C2 * r2 * (
                gbest_position - positions[i]
            )
        )

        positions[i] = (
            positions[i] + velocities[i]
        )

        # Keep positions within VM range
        positions[i] = np.clip(
            positions[i],
            0,
            NUM_VMS - 1
        )

    # Record best solution
    best_allocation = decode_particle(
        gbest_position
    )

    best_cost = calculate_cost(
        best_allocation
    )

    best_makespan, _ = calculate_makespan(
        best_allocation
    )

    fitness_history.append(gbest_fitness)
    cost_history.append(best_cost)
    makespan_history.append(best_makespan)

    if iteration % 100 == 0:

        print(
            "Iteration:", iteration,
            "| Best Cost:", round(best_cost, 4),
            "| Makespan:", round(best_makespan, 4)
        )


# =====================================================
# 10. FINAL RESULTS
# =====================================================

best_allocation = decode_particle(
    gbest_position
)

final_cost = calculate_cost(
    best_allocation
)

final_makespan, vm_load = calculate_makespan(
    best_allocation
)

print("\nFINAL PSO SCHEDULING RESULTS")
print("--------------------------------")

print("Best Execution Cost:", round(final_cost, 4))

print("Best Makespan:", round(final_makespan, 4))

print("Deadline:", DEADLINE)

print("Deadline Satisfied:",
      final_makespan <= DEADLINE)


# =====================================================
# 11. TASK ALLOCATION
# =====================================================

print("\nBEST TASK ALLOCATION")
print("--------------------------------")

for vm in range(NUM_VMS):

    tasks = []

    for task in range(NUM_TASKS):

        if best_allocation[task] == vm:

            tasks.append(
                "T" + str(task + 1)
            )

    print("VM", vm + 1, ":", tasks)


# =====================================================
# 12. VM EXECUTION LOAD
# =====================================================

print("\nVM EXECUTION LOAD")
print("--------------------------------")

for vm in range(NUM_VMS):

    print(
        "VM", vm + 1,
        "Execution Time:",
        round(vm_load[vm], 4)
    )


# =====================================================
# 13. CONVERGENCE GRAPH
# =====================================================

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, ITERATIONS + 1),
    fitness_history
)

plt.title("PSO Workflow Scheduling Convergence")

plt.xlabel("Iteration")

plt.ylabel("Best Fitness")

plt.grid(True)

plt.tight_layout()

plt.show()


# =====================================================
# 14. EXECUTION COST GRAPH
# =====================================================

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, ITERATIONS + 1),
    cost_history
)

plt.title("PSO Execution Cost Optimization")

plt.xlabel("Iteration")

plt.ylabel("Execution Cost")

plt.grid(True)

plt.tight_layout()

plt.show()
<img width="1271" height="722" alt="Screenshot 2026-11-27 110851" src="https://github.com/user-attachments/assets/57f5427a-c876-4e68-bdfe-7b6a48a8a232" />

