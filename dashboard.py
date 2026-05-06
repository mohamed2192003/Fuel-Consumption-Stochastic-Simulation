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

# ── State Transition Diagram ──
st.subheader("🔵 State Transition Diagram")
st.write("Nodes are fuel states; arrows show possible transitions. "
         "Thicker / darker arrows indicate higher probabilities.")

import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle

# Node positions — triangle layout, inset so self-loops don't clip
node_pos = {
    0: np.array([0.50, 0.80]),   # Full Tank   – top centre
    1: np.array([0.18, 0.22]),   # Medium Fuel – bottom left
    2: np.array([0.82, 0.22]),   # Low Fuel    – bottom right
}
node_colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
node_labels = {0: "Full Tank", 1: "Medium\nFuel", 2: "Low\nFuel"}
NODE_R = 0.09   # node radius in axes units

fig_diag, ax_diag = plt.subplots(figsize=(6, 5.5))
# Expanded limits so all nodes and self-loop arcs fit without clipping
ax_diag.set_xlim(-0.10, 1.10)
ax_diag.set_ylim(-0.12, 1.12)
ax_diag.axis("off")
ax_diag.set_facecolor("#f8f9fa")
fig_diag.patch.set_facecolor("#f8f9fa")

def _angle(p1, p2):
    d = p2 - p1
    return np.degrees(np.arctan2(d[1], d[0]))

CENTRE = np.array([0.50, 0.42])   # visual centroid of the triangle

for i in range(3):
    for j in range(3):
        prob = TRANSITION_MATRIX[i, j]
        if prob == 0:
            continue

        pi, pj = node_pos[i], node_pos[j]

        if i == j:
            # Self-loop: arc positioned outward from the triangle centroid
            direction = pi - CENTRE
            direction = direction / np.linalg.norm(direction)
            loop_r = 0.085
            lx = pi[0] + direction[0] * (NODE_R + loop_r * 0.85)
            ly = pi[1] + direction[1] * (NODE_R + loop_r * 0.85)
            loop = mpatches.Arc(
                (lx, ly), loop_r * 2, loop_r * 2,
                angle=_angle(pi, np.array([lx, ly])) + 90,
                theta1=40, theta2=320,
                color="#888888", lw=max(1.2, prob * 4),
                zorder=2,
            )
            ax_diag.add_patch(loop)
            lbl_pos = np.array([lx, ly]) + direction * (loop_r * 1.35)
            ax_diag.text(
                lbl_pos[0], lbl_pos[1], f"{prob:.1f}",
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="#555555", zorder=5,
            )
            continue

        # Inter-node arrow
        theta_ij = np.radians(_angle(pi, pj))
        theta_ji = np.radians(_angle(pj, pi))
        perp = np.array([-np.sin(theta_ij), np.cos(theta_ij)]) * 0.055

        src = pi + np.array([np.cos(theta_ij), np.sin(theta_ij)]) * NODE_R + perp
        dst = pj + np.array([np.cos(theta_ji), np.sin(theta_ji)]) * NODE_R + perp

        lw    = 1.2 + prob * 5
        alpha = 0.40 + prob * 0.60
        color = node_colors[i]

        arrow = FancyArrowPatch(
            posA=src, posB=dst,
            arrowstyle=mpatches.ArrowStyle.CurveB(head_length=0.016, head_width=0.009),
            connectionstyle="arc3,rad=0.20",
            color=color, lw=lw, alpha=alpha, zorder=3,
        )
        ax_diag.add_patch(arrow)

        mid = (src + dst) / 2 + perp * 1.5
        ax_diag.text(
            mid[0], mid[1], f"{prob:.1f}",
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=color, alpha=min(alpha + 0.2, 1.0),
            zorder=6,
        )

# Draw nodes last (on top)
for pos, col, lbl in zip(node_pos.values(), node_colors.values(), node_labels.values()):
    circ = Circle(pos, NODE_R, color=col, zorder=4, ec="white", lw=2.5)
    ax_diag.add_patch(circ)
    ax_diag.text(
        pos[0], pos[1], lbl,
        ha="center", va="center", fontsize=9, fontweight="bold",
        color="white", zorder=7, linespacing=1.3,
    )

ax_diag.set_title("Markov Chain — State Transition Diagram", fontsize=11, pad=10)
fig_diag.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)
st.pyplot(fig_diag)

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
