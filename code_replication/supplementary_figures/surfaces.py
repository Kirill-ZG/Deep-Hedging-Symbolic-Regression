import numpy as np
import torch
import joblib
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm

from include.network import Actor
from include.settings import getSettings

# ==========================================
# 1. CONFIGURATION
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 16,              # Big fonts for separate plots
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "mathtext.fontset": "cm",
})

FULL_MODEL_NAME = "GBM_new_kappa1_17000"
BASE_MODEL_NAME = FULL_MODEL_NAME.rsplit('_', 1)[0]
SCALER_PATH = f"model/{BASE_MODEL_NAME}_scaler"
ACTOR_PATH = f"model/{FULL_MODEL_NAME}_actor"
SIGMA_TRUE = 0.25
R = 0.0
Q = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
s = getSettings()
actor = Actor(4, s).to(device)
if os.path.exists(ACTOR_PATH):
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device, weights_only=True))
    actor.eval()
scaler = joblib.load(SCALER_PATH)
DETECTED_VOL = scaler.mean_[3]

# ==========================================
# 2. DATA GENERATION
# ==========================================
def get_bs_delta(moneyness, T_trading_years):
    v = SIGMA_TRUE
    T = np.maximum(T_trading_years, 1e-5)
    d1 = (np.log(moneyness) + (R - Q + 0.5 * v**2) * T) / (v * np.sqrt(T))
    return norm.cdf(d1)

def get_agent_action(moneyness, T_trading_years, perfect_pos):
    cal_days = T_trading_years * 365.0
    T_30 = cal_days / 30.0
    state = np.array([moneyness, T_30, perfect_pos, DETECTED_VOL])
    state_scaled = scaler.transform(state.reshape(1, -1))
    state_tensor = torch.FloatTensor(state_scaled).to(device)
    with torch.no_grad():
        raw = actor(state_tensor).cpu().numpy().flatten()[0]
    return np.clip(0.5 * (raw + 1), 0, 1)

def get_symbolic_formula(x0, x1):
    from scipy.special import erf
    x1_safe = np.maximum(x1, 1e-6)
    x0_safe = np.maximum(x0, 1e-6)
    term1 = 0.099981345 * x1
    term2 = 0.5068208 * erf( (2.81399308714458 * np.log(x0_safe)) / np.sqrt(x1_safe) )
    term3 = 0.4943565
    return term1 + term2 + term3

RES = 60
m_range = np.linspace(0.85, 1.15, RES)
t_range = np.linspace(20/252, 90/252, RES) 
M_grid, T_grid = np.meshgrid(m_range, t_range)
T_days_plot = T_grid * 252 

Z_BS = np.zeros_like(M_grid)
Z_Agent = np.zeros_like(M_grid)
Z_SR = np.zeros_like(M_grid)

for i in range(RES):
    for j in range(RES):
        m = M_grid[i, j]
        t = T_grid[i, j]
        bs = get_bs_delta(m, t)
        Z_BS[i, j] = bs
        Z_Agent[i, j] = get_agent_action(m, t, -bs)
        Z_SR[i, j] = get_symbolic_formula(m, t)

# ==========================================
# 3. SAVE INDIVIDUAL PLOTS
# ==========================================
def save_plot(filename, Z, title, error_plot=False):
    fig = plt.figure(figsize=(9, 7), layout="constrained")
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Surface
    cmap = cm.coolwarm if error_plot else cm.viridis
    surf = ax.plot_surface(M_grid, T_days_plot, Z, cmap=cmap, 
                           linewidth=0, antialiased=True, alpha=0.95)
    
    # Format Axis
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    
    ax.set_xlabel(r'Moneyness ($S/K$)', labelpad=12)
    ax.set_ylabel(r'Time (Days)', labelpad=12)
    
    z_label = "Error" if error_plot else "Delta"
    ax.set_zlabel(z_label, labelpad=15, rotation=90, fontsize=16)
    
    ax.set_xticks([0.85, 0.95, 1.05, 1.15])
    ax.set_yticks([20, 45, 70, 95])
    
    # Limits
    if error_plot:
        ax.set_zlim(-0.05, 0.05)
    else:
        ax.set_zlim(0, 1)

    ax.set_title(title, pad=10, weight='bold')
    ax.view_init(elev=20, azim=-110)
    
    # Save
    plt.savefig(filename, format="pdf", bbox_inches='tight', pad_inches=0.1)
    print(f"Saved {filename}")
    plt.close()

print("Saving individual figures...")
save_plot("fig_surface_bs.pdf", Z_BS, "Black-Scholes Formula")
save_plot("fig_surface_agent.pdf", Z_Agent, "Deep RL Agent")
save_plot("fig_surface_sr.pdf", Z_SR, "Distilled Formula")
save_plot("fig_error_agent.pdf", Z_Agent - Z_BS, "Agent Error", error_plot=True)
save_plot("fig_error_sr.pdf", Z_SR - Z_BS, "Formula Error", error_plot=True)