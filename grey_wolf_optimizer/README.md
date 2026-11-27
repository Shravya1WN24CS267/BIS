
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Download Chronic Kidney Disease dataset
print("Loading CKD dataset...")

dataset = fetch_ucirepo(id=336)

X = dataset.data.features
y = dataset.data.targets.iloc[:, 0]

# Clean data
X = X.replace("?", np.nan)
X = X.replace(r"^\s*$", np.nan, regex=True)

for column in X.select_dtypes(include="object").columns:
    X[column] = X[column].str.strip()

# Convert categorical features
X = pd.get_dummies(X, dummy_na=True)

# Convert target labels
y = y.astype(str).str.strip().str.lower()
y = y.replace({"ckd": 1, "notckd": 0, "not ckd": 0})

valid = y.isin([0, 1])
X = X.loc[valid]
y = y.loc[valid].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Preprocessing and SVM pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

# Fitness function for GWO
def fitness(position):
    C = 10 ** position[0]
    gamma = 10 ** position[1]

    pipeline.set_params(svm__C=C, svm__gamma=gamma)

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=3,
        scoring="accuracy"
    )

    return 1 - scores.mean()


# Grey Wolf Optimizer
def gwo(population=8, iterations=15, dimensions=2):

    wolves = np.random.uniform(-3, 2, (population, dimensions))

    alpha = np.zeros(dimensions)
    beta = np.zeros(dimensions)
    delta = np.zeros(dimensions)

    alpha_score = beta_score = delta_score = float("inf")
    convergence = []

    for iteration in range(iterations):

        for wolf in wolves:
            score = fitness(wolf)

            if score < alpha_score:
                delta_score, delta = beta_score, beta.copy()
                beta_score, beta = alpha_score, alpha.copy()
                alpha_score, alpha = score, wolf.copy()

            elif score < beta_score:
                delta_score, delta = beta_score, beta.copy()
                beta_score, beta = score, wolf.copy()

            elif score < delta_score:
                delta_score, delta = score, wolf.copy()

        a = 2 - 2 * iteration / iterations

        for i in range(population):
            A1 = 2 * a * np.random.rand(dimensions) - a
            C1 = 2 * np.random.rand(dimensions)
            D1 = abs(C1 * alpha - wolves[i])
            X1 = alpha - A1 * D1

            A2 = 2 * a * np.random.rand(dimensions) - a
            C2 = 2 * np.random.rand(dimensions)
            D2 = abs(C2 * beta - wolves[i])
            X2 = beta - A2 * D2

            A3 = 2 * a * np.random.rand(dimensions) - a
            C3 = 2 * np.random.rand(dimensions)
            D3 = abs(C3 * delta - wolves[i])
            X3 = delta - A3 * D3

            wolves[i] = np.clip(
                (X1 + X2 + X3) / 3,
                -3, 2
            )

        convergence.append(alpha_score)

        print(
            f"Iteration {iteration + 1}/{iterations} "
            f"Fitness: {alpha_score:.4f}"
        )

    return alpha, convergence


# Run GWO
print("\nStarting Grey Wolf Optimization...")

best_position, convergence = gwo()

best_C = 10 ** best_position[0]
best_gamma = 10 ** best_position[1]

print("\nBest C:", best_C)
print("Best Gamma:", best_gamma)

# Train final SVM
pipeline.set_params(
    svm__C=best_C,
    svm__gamma=best_gamma
)

pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluation
print("\n========== GWO-SVM RESULTS ==========")
print("Accuracy :", accuracy_score(y_test, y_pred) * 100, "%")
print("Precision:", precision_score(y_test, y_pred) * 100, "%")
print("Recall   :", recall_score(y_test, y_pred) * 100, "%")
print("F1 Score :", f1_score(y_test, y_pred) * 100, "%")

# Confusion matrix
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred),
    display_labels=["Not CKD", "CKD"]
).plot()

plt.title("GWO-SVM Confusion Matrix")
plt.show()

# Convergence graph
plt.plot(convergence, marker="o")
plt.title("Grey Wolf Optimizer Convergence")
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.grid()
plt.show()
<img width="860" height="797" alt="Screenshot 2026-11-27 170525" src="https://github.com/user-attachments/assets/95d5d326-e9b2-4375-9b85-0b0158532709" />

