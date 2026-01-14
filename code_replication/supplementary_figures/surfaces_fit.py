import numpy as np
import torch
import joblib
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import erf

from include.network import Actor
from include.settings import getSettings

# ==========================================
# 0. PUBLICATION STYLE
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# ==========================================
# 1. SETUP
# ==========================================
FULL_MODEL_NAME = "Heston_new_kappa1_12000"
BASE_MODEL_NAME = FULL_MODEL_NAME.rsplit('_', 1)[0]
SCALER_PATH = f"model/{BASE_MODEL_NAME}_scaler"
ACTOR_PATH = f"model/{FULL_MODEL_NAME}_actor"
DAYS_IN_YEAR = 365.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
s = getSettings()
actor = Actor(4, s).to(device)

if os.path.exists(ACTOR_PATH):
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device, weights_only=True))
    actor.eval()
else:
    sys.exit("Actor not found")
    
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
else:
    sys.exit("Scaler not found")

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def symbolic_regression_predict(x0, x1, x2):
    return 0.50660795 * erf(0.6403294 * (x0 - 1.013572) / (np.sqrt(x1) * x2)) + 0.48919007

def query_agent(moneyness, days, volatility):
    T_30 = days / 30.0
    state = np.array([moneyness, T_30, 0.0, volatility])
    state_scaled = scaler.transform(state.reshape(1, -1))
    
    state_tensor = torch.FloatTensor(state_scaled).to(device)
    with torch.no_grad():
        raw = actor(state_tensor).cpu().numpy().flatten()[0]
    return np.clip(0.5 * (raw + 1), 0, 1)

# ==========================================
# 3. METHOD A: CONTOUR OVERLAYS
# ==========================================
def plot_contour(vol, filename):
    print(f"Generating Contour for Vol={vol}...")
    
    # High resolution for smooth lines
    m_vals = np.linspace(0.85, 1.15, 100)
    d_vals = np.linspace(40, 90, 100)
    M, D = np.meshgrid(m_vals, d_vals)
    
    Z_Ag = np.zeros_like(M)
    Z_SR = np.zeros_like(M)
    
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            t_yr = D[r,c] / DAYS_IN_YEAR
            Z_Ag[r,c] = query_agent(M[r,c], D[r,c], vol)
            Z_SR[r,c] = symbolic_regression_predict(M[r,c], t_yr, vol)

    fig, ax = plt.subplots(figsize=(7, 6))
    
    # 1. Agent = Filled Contours (Background)
    # levels=20 ensures smooth color gradients
    cntr = ax.contourf(M, D, Z_Ag, levels=20, cmap='viridis', alpha=0.9)
    
    # 2. Symbolic Regression = Dashed Lines (Overlay)
    # We use fewer levels (10) for clarity
    # colors='white' provides high contrast on viridis
    lines = ax.contour(M, D, Z_SR, levels=10, colors='white', linestyles='dashed', linewidths=1.5)
    
    ax.clabel(lines, inline=True, fontsize=10, fmt='%.2f')
    
    ax.set_title(f'Policy Topology ($\sigma={vol}$)')
    ax.set_xlabel('Moneyness ($S/K$)')
    ax.set_ylabel('Days to Maturity')
    
    # Add a dummy legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='teal', lw=4, label='Deep Agent (Fill)'),
        Line2D([0], [0], color='white', lw=2, linestyle='--', label='Symbolic Eq (Lines)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.2)
    
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

# ==========================================
# 4. METHOD B: PARITY PLOTS (SCATTER)
# ==========================================
def plot_parity(vol, filename):
    print(f"Generating Parity Plot for Vol={vol}...")
    
    # Generate Random Points
    N_POINTS = 1000
    m_rand = np.random.uniform(0.85, 1.15, N_POINTS)
    d_rand = np.random.uniform(40, 90, N_POINTS)
    
    y_true = []
    y_pred = []
    
    for i in range(N_POINTS):
        t_yr = d_rand[i] / DAYS_IN_YEAR
        y_true.append(query_agent(m_rand[i], d_rand[i], vol))
        y_pred.append(symbolic_regression_predict(m_rand[i], t_yr, vol))
        
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # 1. Identity Line (Perfect Fit)
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1, alpha=0.5, label='Ideal Fit')
    
    # 2. Scatter Points
    # Color by Moneyness to show structure
    sc = ax.scatter(y_true, y_pred, c=m_rand, cmap='coolwarm', s=10, alpha=0.7)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    ax.set_xlabel(r'Deep Agent Output ($\delta_{\text{Stochastic}}$)')
    ax.set_ylabel(r'Symbolic Formula ($\delta_{\text{Stochsatic}}^{\text{SR}}$)')
    ax.set_title(f'Goodness of Fit ($\sigma={vol}$)', weight='bold', fontsize=20)
    
    # Add R^2 Score
    corr_matrix = np.corrcoef(y_true, y_pred)
    r2 = corr_matrix[0,1]**2
    ax.text(0.05, 0.9, f'$R^2 = {r2:.4f}$', transform=ax.transAxes, fontsize=14, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Moneyness', rotation=270, labelpad=15)
    
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

# ==========================================
# 5. EXECUTION
# ==========================================
vols = [0.15, 0.25, 0.35]

for v in vols:
    # plot_contour(v, f'contour_v{v}.pdf')
    plot_parity(v, f'parity_v{v}.pdf')

print("Done.")