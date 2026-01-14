import numpy as np
import torch
import joblib
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
from pysr import PySRRegressor

from include.network import Actor
from include.settings import getSettings

# ==========================================
# 1. CONFIG & LOAD
# ==========================================
FULL_MODEL_NAME = "Heston_new_kappa1_12000"
BASE_MODEL_NAME = FULL_MODEL_NAME.rsplit('_', 1)[0]
SCALER_PATH = f"model/{BASE_MODEL_NAME}_scaler"
ACTOR_PATH = f"model/{FULL_MODEL_NAME}_actor"

# Simulation Constants for Heston
DAYS_IN_YEAR = 365.0
R = 0.0
Q = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
s = getSettings()
actor = Actor(4, s).to(device)

print(f"--- Loading Model: {FULL_MODEL_NAME} ---")

if os.path.exists(ACTOR_PATH):
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device, weights_only=True))
    actor.eval()
else:
    sys.exit(f"Error: Actor not found at {ACTOR_PATH}")
    
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")
else:
    sys.exit(f"Error: Scaler not found at {SCALER_PATH}")

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_bs_delta(moneyness, T_years, v):
    """ Standard Black-Scholes Delta - Our benchmark """
    T_years = np.maximum(T_years, 1e-5)
    d1 = (np.log(moneyness) + (R - Q + 0.5 * v**2) * T_years) / (v * np.sqrt(T_years))
    return norm.cdf(d1)

def query_agent(moneyness, days, volatility, perfect_position):
    """
    Query the agent. Time is in calendar days.
    """
    # Agent expects T in days / 30
    T_30 = days / 30.0
    
    # State: [S/K, T/30, StockOwned, v]
    state = np.array([moneyness, T_30, perfect_position, volatility])
    state_scaled = scaler.transform(state.reshape(1, -1))
    
    state_tensor = torch.FloatTensor(state_scaled).to(device)
    with torch.no_grad():
        raw = actor(state_tensor).cpu().numpy().flatten()[0]
        
    return np.clip(0.5 * (raw + 1), 0, 1)

# ==========================================
# 3. GENERATE DATASET FOR PySR
# ==========================================
N_SAMPLES = 4000  # More samples for the 3D input space
print(f"\nGenerating {N_SAMPLES} samples for Symbolic Regression...")

# Bounds
min_t_yr = 40 / DAYS_IN_YEAR
max_t_yr = 90 / DAYS_IN_YEAR
min_v, max_v = 0.10, 0.40 # A reasonable range for Heston vol

X_moneyness = np.random.uniform(0.85, 1.15, N_SAMPLES)
X_time_yr = np.random.uniform(min_t_yr, max_t_yr, N_SAMPLES)
X_vol = np.random.uniform(min_v, max_v, N_SAMPLES)

y_agent = []

for i in range(N_SAMPLES):
    m, t, v = X_moneyness[i], X_time_yr[i], X_vol[i]
    
    # Use BS as the 'perfect hedge' reference to remove transaction cost friction
    bs_hedge = get_bs_delta(m, t, v)
    
    # Query Agent
    agent_val = query_agent(m, t * DAYS_IN_YEAR, v, -bs_hedge)
    y_agent.append(agent_val)

y_agent = np.array(y_agent)
# PySR input is now 3D: [Moneyness, Time_years, Volatility]
X_sr = np.stack([X_moneyness, X_time_yr, X_vol], axis=1)

# ==========================================
# 4. RUN SYMBOLIC REGRESSION
# ==========================================
print("\n--- Running PySR on 3D Heston Data ---")
model = PySRRegressor(
    niterations=200, 
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["log", "sqrt", "erf"], 
    # Let PySR know there are 3 variables: x0, x1, x2
    model_selection="best",
    elementwise_loss="loss(x, y) = (x - y)^2",
    verbosity=0
)

model.fit(X_sr, y_agent)

print("\n" + "="*50)
print("RECOVERED FORMULA:")
print(model.sympy())
print("="*50 + "\n")

# ==========================================
# 5. COMPARISON TABLE (50 Examples)
# ==========================================
print(f"{'Moneyness':<10} | {'Days':<6} | {'Vol':<6} | {'BS Delta':<10} | {'Agent':<10} | {'SR Form':<10} | {'Ag-BS':<8}")
print("-" * 85)

# Generate 50 random test points
test_m = np.random.uniform(0.85, 1.15, 50)
test_t = np.random.uniform(min_t_yr, max_t_yr, 50)
test_v = np.random.uniform(min_v, max_v, 50)

for i in range(50):
    m, t, v = test_m[i], test_t[i], test_v[i]
    days = t * DAYS_IN_YEAR
    
    bs_val = get_bs_delta(m, t, v)
    agent_val = query_agent(m, days, v, -bs_val)
    sr_val = model.predict(np.array([[m, t, v]]))[0]
    diff = agent_val - bs_val
    
    print(f"{m:10.4f} | {days:6.1f} | {v:6.3f} | {bs_val:10.4f} | {agent_val:10.4f} | {sr_val:10.4f} | {diff:8.4f}")

# ==========================================
# 6. VISUALIZATION (Research Paper Style)
# ==========================================
print("\nGenerating 3D Surface Plots at Different Volatility Slices...")

v_slices = [0.15, 0.25, 0.35]  # Low, Medium, High Volatility
fig, axes = plt.subplots(len(v_slices), 3, figsize=(18, 15), subplot_kw={'projection': '3d'})
fig.suptitle('Agent vs. Benchmarks Across Volatility Regimes', fontsize=16)

# Create Grid
plot_m = np.linspace(0.85, 1.15, 30)
plot_t_days = np.linspace(40, 90, 30)
plot_t_yrs = plot_t_days / DAYS_IN_YEAR
M_grid, T_grid_yrs = np.meshgrid(plot_m, plot_t_yrs)
_, T_grid_days = np.meshgrid(plot_m, plot_t_days) # For agent query

# Set titles for columns
axes[0, 0].set_title('Black-Scholes Benchmark')
axes[0, 1].set_title('Deep RL Agent')
axes[0, 2].set_title('Symbolic Formula')

for i, v_slice in enumerate(v_slices):
    Z_BS = np.zeros_like(M_grid)
    Z_Agent = np.zeros_like(M_grid)
    Z_SR = np.zeros_like(M_grid)
    
    for r in range(M_grid.shape[0]):
        for c in range(M_grid.shape[1]):
            m = M_grid[r, c]
            t_yr = T_grid_yrs[r, c]
            t_day = T_grid_days[r, c]
            
            bs = get_bs_delta(m, t_yr, v_slice)
            Z_BS[r, c] = bs
            Z_Agent[r, c] = query_agent(m, t_day, v_slice, -bs)
            Z_SR[r, c] = model.predict(np.array([[m, t_yr, v_slice]]))[0]

    # Plot BS Surface
    axes[i, 0].plot_surface(M_grid, T_grid_days, Z_BS, cmap=cm.coolwarm, alpha=0.9)
    axes[i, 0].set_zlim(0, 1)
    axes[i, 0].set_ylabel(f'v={v_slice:.2f}\nDays to Mat.')

    # Plot Agent Surface
    axes[i, 1].plot_surface(M_grid, T_grid_days, Z_Agent, cmap=cm.viridis, alpha=0.9)
    axes[i, 1].set_zlim(0, 1)

    # Plot SR Surface
    axes[i, 2].plot_surface(M_grid, T_grid_days, Z_SR, cmap=cm.plasma, alpha=0.9)
    axes[i, 2].set_zlim(0, 1)
    
# Set labels for the bottom row only to avoid clutter
for j in range(3):
    axes[-1, j].set_xlabel('Moneyness (S/K)')

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
plt.show()