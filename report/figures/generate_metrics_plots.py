"""Generate publication-quality SVG, PDF, and PNG metric plots for MATP."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("matp-monorepo/report/figures", exist_ok=True)
out_dir = "matp-monorepo/report/figures"

# Clean IEEE publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# -----------------------------------------------------------------------------
# Plot 1: Long Horizon Retrieval Accuracy & Stale-State Leakage
# -----------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(6.2, 3.4), dpi=300)

methods = ["FIFO Window\n(k=10 steps)", "Naive RAG\n(Cosine Only)", "Lineage Filter\n(Same-Resource)", "Unified MATP\n(DAG + Decay)"]
active_prec = [25.0, 60.0, 75.0, 100.0]  # % active valid state returned
stale_leakage = [0.0, 40.0, 25.0, 0.0]   # % obsolete or dependent-invalid state leaked

x = np.arange(len(methods))
width = 0.35

rects1 = ax1.bar(x - width/2, active_prec, width, label="Active Precision (%)", color="#1f77b4", edgecolor="#0e446c", zorder=3)
rects2 = ax1.bar(x + width/2, stale_leakage, width, label="Stale/Invalid Leakage (%)", color="#d62728", edgecolor="#8c1b1c", zorder=3)

ax1.set_ylabel("Percentage (%)")
ax1.set_title("Long-Horizon Retrieval: Active State Precision vs. Obsolete Leakage")
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.set_ylim(0, 115)
ax1.legend(loc="upper left", framealpha=0.9)

for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#8c1b1c")

plt.tight_layout()
plt.savefig(f"{out_dir}/metric_retrieval_precision.svg", format="svg")
plt.savefig(f"{out_dir}/metric_retrieval_precision.pdf", format="pdf")
plt.savefig(f"{out_dir}/metric_retrieval_precision.png", format="png", dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# Plot 2: Distillation Compression Ratio vs Key Fact Retention
# -----------------------------------------------------------------------------
fig, (ax_tokens, ax_facts) = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)

categories = ["Raw Output\n(Window Truncated)", "MATP Distilled\n(Tuple + Triples)"]
tokens = [61122, 4524]
facts_retained = [37.3, 88.0]

# Left: Tokens
bars1 = ax_tokens.bar(categories, tokens, color=["#7f7f7f", "#2ca02c"], edgecolor="#333333", width=0.5, zorder=3)
ax_tokens.set_ylabel("Total Tokens Consumed")
ax_tokens.set_title("Token Footprint (13.5× Compression)")
ax_tokens.set_ylim(0, 70000)
for bar in bars1:
    h = bar.get_height()
    ax_tokens.annotate(f"{h:,}", xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
ax_tokens.text(0.5, 0.7, "92.6%\nToken Savings", transform=ax_tokens.transAxes, ha="center",
               bbox=dict(boxstyle="round,pad=0.4", fc="#d4edda", ec="#28a745", lw=1.2))

# Right: Facts
bars2 = ax_facts.bar(categories, facts_retained, color=["#e377c2", "#17becf"], edgecolor="#333333", width=0.5, zorder=3)
ax_facts.set_ylabel("Fact Triple Retention (%)")
ax_facts.set_title("Key Fact Retention Under Budget")
ax_facts.set_ylim(0, 105)
for bar in bars2:
    h = bar.get_height()
    ax_facts.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{out_dir}/metric_compression_fact_retention.svg", format="svg")
plt.savefig(f"{out_dir}/metric_compression_fact_retention.pdf", format="pdf")
plt.savefig(f"{out_dir}/metric_compression_fact_retention.png", format="png", dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# Plot 3: Temporal Decay Curves & Adaptive Lambda Adaptation
# -----------------------------------------------------------------------------
fig, (ax_curve, ax_adapt) = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)

age_steps = np.linspace(0, 50, 100)
lambdas = [0.01, 0.05, 0.15, 0.35]
colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

for lam, c in zip(lambdas, colors):
    decay = np.exp(-lam * age_steps)
    ax_curve.plot(age_steps, decay, label=f"$\lambda={lam}$", color=c)

ax_curve.set_xlabel("Age $\Delta t = t_{curr} - t_i$ (Logical Steps)")
ax_curve.set_ylabel("Temporal Weight $W_T = e^{-\lambda \Delta t}$")
ax_curve.set_title("Continuous Exponential Decay")
ax_curve.legend(loc="upper right", framealpha=0.9)
ax_curve.set_ylim(0, 1.05)

# Adaptive adaptation steps
feedback_steps = np.arange(1, 11)
# Simulated convergence of lambda toward ln(2)/median_age = 0.3466 with alpha=0.35
target_lam = 0.3466
sim_lam = []
cur = 0.05
alpha = 0.35
for _ in feedback_steps:
    cur = (1 - alpha) * cur + alpha * target_lam
    sim_lam.append(cur)

ax_adapt.plot(feedback_steps, sim_lam, marker="o", color="#9467bd", label="Learned $\lambda_{task}$")
ax_adapt.axhline(target_lam, color="#d62728", linestyle="--", label=f"Target $\lambda$ ({target_lam:.3f})")
ax_adapt.axhline(0.05, color="#7f7f7f", linestyle=":", label="Initial Default (0.05)")
ax_adapt.set_xlabel("Feedback Iteration")
ax_adapt.set_ylabel("Estimated $\lambda_{task}$")
ax_adapt.set_title("Adaptive Decay Rate Tuning")
ax_adapt.legend(loc="lower right", framealpha=0.9)
ax_adapt.set_ylim(0, 0.40)

plt.tight_layout()
plt.savefig(f"{out_dir}/metric_temporal_decay_curves.svg", format="svg")
plt.savefig(f"{out_dir}/metric_temporal_decay_curves.pdf", format="pdf")
plt.savefig(f"{out_dir}/metric_temporal_decay_curves.png", format="png", dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# Plot 4: Latency Breakdown Across Monorepo Operations
# -----------------------------------------------------------------------------
fig, ax_lat = plt.subplots(figsize=(6.2, 3.2), dpi=300)

operations = [
    "Causal Invalidation\nUpdate",
    "Parameter Hint\nCosine Match",
    "Vector Search\n(Subword n-gram)",
    "Macro Mining\n& Matching",
    "Trajectory Distillation\n& Ingest"
]
latencies_ms = [0.34, 0.52, 0.85, 1.12, 4.68]

y_pos = np.arange(len(operations))
bars = ax_lat.barh(y_pos, latencies_ms, color="#3498db", edgecolor="#21618c", height=0.55, zorder=3)
ax_lat.set_yticks(y_pos)
ax_lat.set_yticklabels(operations)
ax_lat.set_xlabel("Operation Latency (Milliseconds, log scale)")
ax_lat.set_xscale("log")
ax_lat.set_xlim(0.1, 10.0)
ax_lat.set_title("Unified Monorepo Operational Latency Profile (CPU Local)")

for bar in bars:
    w = bar.get_width()
    ax_lat.annotate(f"{w:.2f} ms", xy=(w, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{out_dir}/metric_latency_breakdown.svg", format="svg")
plt.savefig(f"{out_dir}/metric_latency_breakdown.pdf", format="pdf")
plt.savefig(f"{out_dir}/metric_latency_breakdown.png", format="png", dpi=300)
plt.close()

print("Successfully generated all metric plots in SVG, PDF, and PNG formats!")
