import numpy as np
import torch
import joblib
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import norm

# Import project files
from include.network import Actor
from include.settings import getSettings

# ==========================================
# 0. PUBLICATION STYLE CONFIGURATION
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ==========================================
# 1. CONFIG & LOAD
# ==========================================
FULL_MODEL_NAME = "Heston_new_kappa1_12000"
BASE_MODEL_NAME = FULL_MODEL_NAME.rsplit('_', 1)[0]
SCALER_PATH = f"model/{BASE_MODEL_NAME}_scaler"
ACTOR_PATH = f"model/{FULL_MODEL_NAME}_actor"

# Heston Constants
DAYS_IN_YEAR = 365.0
R = 0.0
Q = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
s = getSettings()
actor = Actor(4, s).to(device)

if os.path.exists(ACTOR_PATH):
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device, weights_only=True))
    actor.eval()
else:
    sys.exit(f"Error: Actor not found at {ACTOR_PATH}")
    
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
else:
    sys.exit(f"Error: Scaler not found at {SCALER_PATH}")

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_bs_delta(moneyness, T_years, v):
    T_years = np.maximum(T_years, 1e-5)
    d1 = (np.log(moneyness) + (R - Q + 0.5 * v**2) * T_years) / (v * np.sqrt(T_years))
    return norm.cdf(d1)

def query_agent(moneyness, days, volatility, perfect_position):
    T_30 = days / 30.0
    state = np.array([moneyness, T_30, perfect_position, volatility])
    state_scaled = scaler.transform(state.reshape(1, -1))
    
    state_tensor = torch.FloatTensor(state_scaled).to(device)
    with torch.no_grad():
        raw = actor(state_tensor).cpu().numpy().flatten()[0]
        
    return np.clip(0.5 * (raw + 1), 0, 1)

def generate_surface_data(vol_target):
    plot_m = np.linspace(0.85, 1.15, 100)
    plot_t_days = np.linspace(40, 90, 100)
    
    M_grid, T_grid = np.meshgrid(plot_m, plot_t_days)
    Z_Diff = np.zeros_like(M_grid)
    
    for r in range(M_grid.shape[0]):
        for c in range(M_grid.shape[1]):
            m = M_grid[r, c]
            d = T_grid[r, c]
            t_yr = d / DAYS_IN_YEAR
            
            bs = get_bs_delta(m, t_yr, vol_target)
            ag = query_agent(m, d, vol_target, -bs)
            Z_Diff[r, c] = ag - bs
            
    return M_grid, T_grid, Z_Diff

# ==========================================
# 3. IMPROVED PLOTTING FUNCTION
# ==========================================
def plot_and_save(vol, filename):
    print(f"Generating polished surface for Volatility = {vol}...")
    X, Y, Z = generate_surface_data(vol)
    
    # 1. Setup Figure
    fig = plt.figure(figsize=(9, 7)) # Slightly wider
    ax = fig.add_subplot(111, projection='3d')
    
    # 2. Refined Grid Style (Subtle)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # Make grid lines thin, dashed, and grey
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
    
    # 3. Surface Plot
    surf = ax.plot_surface(X, Y, Z, cmap=cm.seismic, 
                           vmin=-0.15, vmax=0.05, 
                           linewidth=0, antialiased=False, alpha=0.95)
    
    # 4. Axes Labels
    ax.set_xlabel('Moneyness ($S/K$)', labelpad=12)
    ax.set_ylabel('Time to Maturity (Days)', labelpad=12)
    
    # FIX: Remove Z-label from the axis itself to prevent overlap
    # ax.set_zlabel(...) -> REMOVED
    
    # 5. Tick Formatting (Clean numbers)
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(MaxNLocator(5)) # Limit number of ticks
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis='z', pad=8)
    
    # 6. View Angle
    ax.view_init(elev=32, azim=130)
    
    # 7. Title (Use Figure Suptitle for perfect centering)
    fig.suptitle(f'Hedging Deviation ($\sigma={vol}$)', y=0.92, fontsize=20, weight='bold')
    
    # 8. Colorbar (The Fix for Overlap)
    # Move colorbar further right (pad=0.1)
    # Add the Z-label HERE instead of on the axis
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.10)
    cbar.set_label(r'$\Delta_{Agent} - \Delta_{BS}$', rotation=90, labelpad=15)
    cbar.outline.set_linewidth(0.5)
    
    # Save
    print(f"Saving {filename}...")
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

# ==========================================
# 4. EXECUTION
# ==========================================
vols = [0.15, 0.25, 0.35]
names = ['diff_low_vol.pdf', 'diff_med_vol.pdf', 'diff_high_vol.pdf']

for v, n in zip(vols, names):
    plot_and_save(v, n)

print("Done.")