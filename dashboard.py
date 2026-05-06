"""
dashboard.py
─────────────────────────────────────────────────────────────
Streamlit Dashboard — Fuel Consumption Simulation
Run: streamlit run dashboard.py
─────────────────────────────────────────────────────────────
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation import load_data, poisson_simulation, markov_simulation, STATES, TRANSITION_MATRIX

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fuel Consumption Simulator",
    page_icon="⛽",
    layout="centered",
)

st.title("⛽ Fuel Consumption Simulator")
st.caption("Stochastic Processes Project · Auto-MPG Dataset")
st.markdown("---")

# ─── SECTION 1: DATASET PREVIEW ───────────────────────────────────────────────
st.header("📂 Dataset Preview")

df = load_data()
st.dataframe(df.head(10), use_container_width=True)
st.write(f"**Total rows (after cleaning):** {len(df)}")

col1, col2, col3 = st.columns(3)
col1.metric("Avg MPG",        f"{df['mpg'].mean():.1f}")
col2.metric("Avg Horsepower", f"{df['horsepower'].mean():.1f}")
col3.metric("Avg Weight",     f"{df['weight'].mean():.0f} lbs")

st.markdown("---")

# ─── SECTION 2: MPG PLOT ──────────────────────────────────────────────────────
st.header("📊 MPG Distribution")

fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(df["mpg"], bins=20, color="#4C72B0", edgecolor="white")
ax.set_xlabel("Miles Per Gallon")
ax.set_ylabel("Number of Cars")
ax.set_title("Distribution of MPG in the Dataset")
st.pyplot(fig)

st.markdown("---")

# ─── SECTION 3: TRANSITION MATRIX ───────────────────────────────────────────
st.header("🔄 Markov Chain — Transition Matrix")
st.write(
    "The table below shows the probability of moving from one fuel state to another "
    "in a single step. Each **row** represents the *current* state and each **column** "
    "represents the *next* state — row probabilities sum to 1."
)

state_labels = ["Full Tank", "Medium Fuel", "Low Fuel"]



# ── Heatmap ──
fig_tm, ax_tm = plt.subplots(figsize=(5, 3))
im = ax_tm.imshow(TRANSITION_MATRIX, cmap="YlOrRd", vmin=0, vmax=1)
plt.colorbar(im, ax=ax_tm, label="Probability")
ax_tm.set_xticks(range(3))
ax_tm.set_yticks(range(3))
ax_tm.set_xticklabels(state_labels, fontsize=9)
ax_tm.set_yticklabels(state_labels, fontsize=9)
ax_tm.set_xlabel("Next State", fontsize=9)
ax_tm.set_ylabel("Current State", fontsize=9)
ax_tm.set_title("Transition Probability Heatmap", fontsize=10)
for i in range(3):
    for j in range(3):
        ax_tm.text(j, i, f"{TRANSITION_MATRIX[i, j]:.1f}",
                   ha="center", va="center",
                   color="black" if TRANSITION_MATRIX[i, j] < 0.6 else "white",
                   fontsize=11, fontweight="bold")
plt.tight_layout()
st.pyplot(fig_tm)

# ── Numeric table ──
tm_df = pd.DataFrame(
    TRANSITION_MATRIX,
    index=[f"From: {s}" for s in state_labels],
    columns=[f"To: {s}" for s in state_labels],
)
st.dataframe(
    tm_df.style.format("{:.1%}").background_gradient(cmap="YlOrRd", vmin=0, vmax=1),
    use_container_width=True,
)

st.markdown("---")

# ─── SECTION 4: SIMULATION CONTROLS ──────────────────────────────────────────
st.header("🎮 Run Simulation")

col_a, col_b = st.columns(2)
lam  = col_a.slider("Avg Trips per Day  (λ)", min_value=1, max_value=10, value=3)
days = col_b.slider("Number of Days",          min_value=7, max_value=60, value=30)

start_label = st.selectbox("Starting Fuel State", ["Full Tank", "Medium Fuel", "Low Fuel"])
start_state = {"Full Tank": 0, "Medium Fuel": 1, "Low Fuel": 2}[start_label]

if st.button("▶  Run Simulation", type="primary"):

    # --- Poisson ---
    trips = poisson_simulation(lam=lam, days=days)

    st.subheader("🚗 Poisson Process — Daily Trips")
    st.write(
        f"With **λ = {lam}**, the expected trips per day follow a Poisson distribution. "
        f"Over **{days} days** we simulated **{trips.sum()} total trips** "
        f"(avg **{trips.mean():.2f}** / day)."
    )

    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.bar(range(1, days + 1), trips, color="#DD8452", width=0.8)
    ax2.axhline(lam, color="red", linestyle="--", label=f"λ = {lam}")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Trips")
    ax2.set_title("Simulated Trips per Day (Poisson)")
    ax2.legend()
    st.pyplot(fig2)

    col_p1, col_p2 = st.columns(2)
    col_p1.metric("Total Trips",    trips.sum())
    col_p2.metric("Average / Day",  f"{trips.mean():.2f}")

    st.markdown("---")

    # --- Markov Chain ---
    states = markov_simulation(steps=days, start_state=start_state)
    state_names = [STATES[s] for s in states]
    final_state = STATES[states[-1]]

    st.subheader("⛽ Markov Chain — Fuel States")
    st.write(
        f"Starting from **{start_label}**, the car moved through "
        f"fuel states over **{days} steps**."
    )

    # Colour-code states
    color_map = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
    colors = [color_map[s] for s in states]

    fig3, ax3 = plt.subplots(figsize=(7, 3))
    ax3.bar(range(1, days + 1), [s + 1 for s in states],
            color=colors, width=0.8)
    ax3.set_yticks([1, 2, 3])
    ax3.set_yticklabels(["Full", "Medium", "Low"])
    ax3.set_xlabel("Step (Day)")
    ax3.set_title("Fuel State over Time (Markov Chain)")
    st.pyplot(fig3)

    # State visit counts
    from collections import Counter
    counts = Counter(state_names)
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Full Tank days",    counts.get("Full Tank",    0))
    c2.metric("🟡 Medium Fuel days",  counts.get("Medium Fuel",  0))
    c3.metric("🔴 Low Fuel days",     counts.get("Low Fuel",     0))

    st.success(f"**Final Fuel State after {days} days → {final_state}**")

st.markdown("---")
st.caption("Built with Streamlit · NumPy · Pandas · Matplotlib")
