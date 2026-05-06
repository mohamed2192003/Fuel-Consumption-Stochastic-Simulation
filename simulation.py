"""
simulation.py
─────────────────────────────────────────────────────────────
Fuel Consumption Simulation using Stochastic Processes
Course: Stochastic Processes
Dataset: auto-mpg.csv
─────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─── 1. LOAD & CLEAN DATASET ──────────────────────────────────────────────────

def load_data(path="auto-mpg.csv"):
    df = pd.read_csv(path, na_values="?")
    df = df[["mpg", "cylinders", "horsepower", "weight"]].dropna()
    df = df.astype({"horsepower": float})
    return df

def show_stats(df):
    print("=== Dataset Statistics ===")
    print(df.describe().round(2))
    print(f"\nRows used: {len(df)}")

def plot_mpg(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram of MPG
    axes[0].hist(df["mpg"], bins=20, color="#4C72B0", edgecolor="white")
    axes[0].set_title("MPG Distribution")
    axes[0].set_xlabel("Miles Per Gallon")
    axes[0].set_ylabel("Count")

    # Scatter: Weight vs MPG
    axes[1].scatter(df["weight"], df["mpg"], alpha=0.5, color="#DD8452", s=20)
    axes[1].set_title("Weight vs MPG")
    axes[1].set_xlabel("Weight (lbs)")
    axes[1].set_ylabel("MPG")

    plt.tight_layout()
    plt.savefig("mpg_plots.png", dpi=120)
    plt.show()
    print("Plot saved -> mpg_plots.png")


# ─── 2. POISSON PROCESS ───────────────────────────────────────────────────────
# We model the number of car trips per day as a Poisson Process.
# If on average λ trips happen per day, the number of trips in one day
# follows:  P(X = k) = (λ^k * e^(-λ)) / k!
# We simulate N days to see how trips are distributed.

def poisson_simulation(lam=3.0, days=30, seed=42):
    """
    Simulate daily car trips using a Poisson process.

    Parameters
    ----------
    lam  : average number of trips per day (λ)
    days : number of days to simulate
    seed : random seed for reproducibility

    Returns
    -------
    trips : numpy array of trip counts per day
    """
    np.random.seed(seed)
    trips = np.random.poisson(lam=lam, size=days)
    return trips


# ─── 3. MARKOV CHAIN ──────────────────────────────────────────────────────────
# Three fuel states:
#   0 = Full Tank   1 = Medium Fuel   2 = Low Fuel
#
# Transition matrix P (row i → col j):
#          Full  Med  Low
# Full   [  0.1  0.7  0.2 ]
# Medium [  0.3  0.4  0.3 ]
# Low    [  0.8  0.2  0.0 ]   ← driver refills when low

STATES = {0: "Full Tank", 1: "Medium Fuel", 2: "Low Fuel"}

TRANSITION_MATRIX = np.array([
    [0.1, 0.7, 0.2],   # From Full
    [0.3, 0.4, 0.3],   # From Medium
    [0.8, 0.2, 0.0],   # From Low → high chance of refill (→ Full)
])

def markov_simulation(steps=30, start_state=0, seed=42):
    """
    Simulate fuel-state transitions using a Markov Chain.

    Parameters
    ----------
    steps       : number of simulation steps (days)
    start_state : initial fuel state (0=Full, 1=Medium, 2=Low)
    seed        : random seed

    Returns
    -------
    states : list of state indices visited
    """
    np.random.seed(seed)
    current = start_state
    states = [current]

    for _ in range(steps - 1):
        probs = TRANSITION_MATRIX[current]
        current = np.random.choice([0, 1, 2], p=probs)
        states.append(current)

    return states


# ─── 4. QUICK CONSOLE RUN ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Dataset ---
    df = load_data()
    show_stats(df)
    plot_mpg(df)

    # --- Poisson ---
    lam = 3
    days = 30
    trips = poisson_simulation(lam=lam, days=days)
    print(f"\n=== Poisson Process (lam={lam}, {days} days) ===")
    print(f"Daily trips: {trips}")
    print(f"Total trips : {trips.sum()} | Average/day: {trips.mean():.2f}")

    # --- Markov Chain ---
    states = markov_simulation(steps=30, start_state=0)
    state_names = [STATES[s] for s in states]
    print("\n=== Markov Chain (30 steps) ===")
    print(" -> ".join(state_names[:8]), "-> ...")
    print(f"Final state : {STATES[states[-1]]}")

