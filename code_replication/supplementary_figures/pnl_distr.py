import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. Configuration
# ==========================================
MODEL_NAME = "GBM_new_kappa3_19000"
FILE_PATH = f'results/testing/{MODEL_NAME}.csv'
OUTPUT_FILE = f'results/{MODEL_NAME}_pnl_dist.pdf'
RISK_AVERSION = 1

# Professional Style settings
sns.set_context("paper", font_scale=1.6) # Increased readability
sns.set_style("whitegrid", {'grid.linestyle': ':', 'grid.alpha': 0.5})
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False
})

# ==========================================
# 2. Data Processing
# ==========================================
if not os.path.exists(FILE_PATH):
    print(f"Error: File {FILE_PATH} not found.")
    exit()

df = pd.read_csv(FILE_PATH)
episode_stats = df.groupby('episode')[['A PnL', 'B PnL']].sum()

agent_pnl = 100 * episode_stats['A PnL']
bs_pnl = 100 * episode_stats['B PnL']

# Smart Zoom (1st to 99th percentile)
combined = pd.concat([agent_pnl, bs_pnl])
x_min, x_max = combined.quantile(0.005), combined.quantile(0.995)
padding = (x_max - x_min) * 0.15
x_min -= padding
x_max += padding

# Stats Text
stats_text = (
    r"$\bf{Performance}$" + "\n"
    f"Agent Mean: {agent_pnl.mean():.4f}%\n"
    f"BS Mean:    {bs_pnl.mean():.4f}%\n"
    f"Agent Std:  {agent_pnl.std():.4f}%\n"
    f"BS Std:     {bs_pnl.std():.4f}%" 
)

# COLOR PALETTE (High Contrast, No Mud)
# Agent: Navy Blue
COLOR_AGENT = "#003f5c"  
# Benchmark: Strong Burnt Orange
COLOR_BM = "#d35400"     

# ==========================================
# 3. Plotting
# ==========================================
fig, ax = plt.subplots()

# --- BENCHMARK (Orange Dashed Line, NO FILL) ---
# Removing the fill solves the color mixing catastrophe.
sns.kdeplot(
    bs_pnl, 
    fill=False,        # No fill to avoid mud
    color=COLOR_BM,
    linewidth=2.5,
    linestyle="--",    # Dashed to distinguish
    label='Black-Scholes (Benchmark)',
    bw_adjust=0.7,
    clip=(x_min, x_max),
    ax=ax
)

# --- AGENT (Blue Solid Line, Blue FILL) ---
# The Agent gets the visual weight (Fill)
sns.kdeplot(
    agent_pnl, 
    fill=True, 
    color=COLOR_AGENT, 
    alpha=0.25,        # Clean, light transparency
    linewidth=3,
    linestyle="-",
    label='Deep Hedging Agent',
    bw_adjust=0.7,
    clip=(x_min, x_max),
    ax=ax
)

# Mean Lines
ax.axvline(agent_pnl.mean(), color=COLOR_AGENT, linestyle='-', linewidth=2, alpha=0.9)
ax.axvline(bs_pnl.mean(), color=COLOR_BM, linestyle='--', linewidth=2, alpha=0.9)
ax.axvline(0, color='black', linewidth=1, alpha=0.3) 

# Labels and Title
ax.set_xlim(x_min, x_max)
ax.set_title(rf'PnL Distribution: Stochastic Volatility (Risk Aversion $\xi={RISK_AVERSION}$)', fontsize=18, pad=15)
ax.set_xlabel('Total PnL per Episode in %', labelpad=10)
ax.set_ylabel('Probability Density', labelpad=10)

# Legend
legend = ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=12)
legend.get_frame().set_linewidth(0.5)

# Stats Box
props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, edgecolor='#dddddd')
ax.text(0.96, 0.96, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right', bbox=props)

# ==========================================
# 4. Save
# ==========================================
plt.tight_layout()
plt.savefig(OUTPUT_FILE, format='pdf', dpi=300, bbox_inches='tight')
print(f"Saved Refined Diagram (No Color Mixing) to {OUTPUT_FILE}")
plt.show()