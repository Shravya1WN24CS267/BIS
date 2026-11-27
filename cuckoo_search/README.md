
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import gamma
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==================================================
# CUCKOO SEARCH FOR FEATURE SELECTION
# ==================================================

np.random.seed(42)


# -----------------------------
# 1. Load Dataset
# -----------------------------

data = load_breast_cancer()

X = data.data
y = data.target

feature_names = data.feature_names

num_features = X.shape[1]

print("Dataset: Breast Cancer Classification")
print("Total samples:", X.shape[0])
print("Total features:", num_features)


# -----------------------------
# 2. Split Dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# 3. Classification Model
# -----------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000))
])


# -----------------------------
# 4. Baseline Accuracy
# -----------------------------

model.fit(X_train, y_train)

baseline_predictions = model.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print("\nAccuracy using all features:",
      round(baseline_accuracy * 100, 2), "%")


# ==================================================
# CUCKOO SEARCH PARAMETERS
# ==================================================

NESTS = 15
MAX_ITERATIONS = 25

PA = 0.25
ALPHA = 0.5
LAMBDA = 0.15

# Cross-validation
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# ==================================================
# 5. Fitness Function
# ==================================================

fitness_cache = {}


def fitness(position):

    # Convert continuous position into binary features
    selected = position > 0.5

    # Ensure at least one feature is selected
    if not np.any(selected):
        selected[np.argmax(position)] = True

    selected_key = tuple(selected)

    # Reuse previously calculated fitness
    if selected_key in fitness_cache:
        return fitness_cache[selected_key]

    selected_count = np.sum(selected)

    X_selected = X_train[:, selected]

    # Cross-validation accuracy
    scores = cross_val_score(
        model,
        X_selected,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    mean_accuracy = np.mean(scores)

    # Feature selection objective:
    # minimize classification error + feature penalty
    feature_ratio = selected_count / num_features

    objective = (
        (1 - mean_accuracy)
        + LAMBDA * feature_ratio
    )

    fitness_cache[selected_key] = objective

    return objective


# ==================================================
# 6. Levy Flight
# ==================================================

def levy_flight():

    beta = 1.5

    sigma = (
        gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (
            gamma((1 + beta) / 2)
            * beta
            * 2 ** ((beta - 1) / 2)
        )
    ) ** (1 / beta)

    u = np.random.normal(
        0, sigma, num_features
    )

    v = np.random.normal(
        0, 1, num_features
    )

    step = u / (
        np.abs(v) ** (1 / beta)
    )

    return step


# ==================================================
# 7. Initialize Nests
# ==================================================

nests = np.random.rand(
    NESTS, num_features
)

fitness_values = np.array([
    fitness(nest)
    for nest in nests
])


best_index = np.argmin(fitness_values)

best_nest = nests[best_index].copy()

best_fitness = fitness_values[best_index]

convergence = []


# ==================================================
# 8. Main Cuckoo Search Algorithm
# ==================================================

print("\nStarting Cuckoo Search Optimization")
print("----------------------------------")


for iteration in range(1, MAX_ITERATIONS + 1):

    # Generate new solutions using Levy flight
    for i in range(NESTS):

        step = levy_flight()

        new_nest = (
            nests[i]
            + ALPHA * step
        )

        # Keep values within [0, 1]
        new_nest = np.clip(
            new_nest, 0, 1
        )

        new_fitness = fitness(new_nest)

        # Select a random host nest
        j = np.random.randint(NESTS)

        # Replace if the new solution is better
        if new_fitness < fitness_values[j]:

            nests[j] = new_nest.copy()

            fitness_values[j] = new_fitness


    # Discover and abandon some nests
    number_to_abandon = max(
        1, int(PA * NESTS)
    )

    worst_indices = np.argsort(
        fitness_values
    )[-number_to_abandon:]


    for index in worst_indices:

        nests[index] = np.random.rand(
            num_features
        )

        fitness_values[index] = fitness(
            nests[index]
        )


    # Update global best
    current_best_index = np.argmin(
        fitness_values
    )

    current_best_fitness = fitness_values[
        current_best_index
    ]

    if current_best_fitness < best_fitness:

        best_fitness = current_best_fitness

        best_nest = nests[
            current_best_index
        ].copy()


    convergence.append(best_fitness)


    print(
        f"Iteration {iteration:2d} | "
        f"Best Fitness: {best_fitness:.5f}"
    )


# ==================================================
# 9. Final Selected Features
# ==================================================

selected_features = best_nest > 0.5

# Ensure at least one feature is selected
if not np.any(selected_features):

    selected_features[
        np.argmax(best_nest)
    ] = True


selected_names = feature_names[
    selected_features
]

print("\nOptimization Completed")
print("---------------------")

print("Total features:", num_features)

print(
    "Selected features:",
    len(selected_names)
)

print("\nSelected Feature Names:")

for name in selected_names:
    print("-", name)


# ==================================================
# 10. Evaluate Selected Features
# ==================================================

X_train_selected = X_train[
    :, selected_features
]

X_test_selected = X_test[
    :, selected_features
]


final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000))
])


final_model.fit(
    X_train_selected,
    y_train
)


predictions = final_model.predict(
    X_test_selected
)


selected_accuracy = accuracy_score(
    y_test,
    predictions
)


print("\nFinal Results")
print("------------")

print(
    "Accuracy using all features:",
    round(baseline_accuracy * 100, 2),
    "%"
)

print(
    "Accuracy using selected features:",
    round(selected_accuracy * 100, 2),
    "%"
)

print(
    "Feature reduction:",
    round(
        (1 - len(selected_names) / num_features)
        * 100,
        2
    ),
    "%"
)


# ==================================================
# 11. Convergence Graph
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, MAX_ITERATIONS + 1),
    convergence,
    marker="o"
)

plt.title(
    "Cuckoo Search Convergence"
)

plt.xlabel("Iteration")

plt.ylabel("Best Fitness")

plt.grid(True)

plt.tight_layout()

plt.show()


# ==================================================
# 12. Feature Selection Graph
# ==================================================

plt.figure(figsize=(8, 5))

categories = [
    "Original Features",
    "Selected Features"
]

values = [
    num_features,
    len(selected_names)
]

plt.bar(
    categories,
    values
)

plt.title(
    "Feature Reduction Using Cuckoo Search"
)

plt.ylabel("Number of Features")

plt.tight_layout()

plt.show()
<img width="1006" height="727" alt="Screenshot 2026-11-27 130830" src="https://github.com/user-attachments/assets/1a3db82b-a41d-41be-99b9-15b404b82122" />

