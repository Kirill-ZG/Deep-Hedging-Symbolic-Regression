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
FULL_MODEL_NAME = "GBM_new_kappa1_17000"
BASE_MODEL_NAME = FULL_MODEL_NAME.rsplit('_', 1)[0]
SCALER_PATH = f"model/{BASE_MODEL_NAME}_scaler"
ACTOR_PATH = f"model/{FULL_MODEL_NAME}_actor"
OUTPUT_FILE = f"surfaces_gbm.pdf"

# Simulation Constants
SIGMA_TRUE = 0.25
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
    # The volatile feature index 3 corresponds to what the agent 'sees'
    DETECTED_VOL = scaler.mean_[3]
    print(f"Scaler loaded. Agent Internal Vol: {DETECTED_VOL:.4f}")
else:
    sys.exit(f"Error: Scaler not found at {SCALER_PATH}")

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_bs_delta(moneyness, T_years, v):
    """ Standard Black-Scholes Delta """
    T_years = np.maximum(T_years, 1e-5)
    d1 = (np.log(moneyness) + (R - Q + 0.5 * v**2) * T_years) / (v * np.sqrt(T_years))
    return norm.cdf(d1)

def query_agent(moneyness, trading_time_years, perfect_position):
    """
    Query the agent converting Physical Trading Time -> Agent's Calendar Time
    """
    # 1. Convert Trading Years -> Calendar Days
    # Logic: T_years * 365 gives the calendar days the Agent expects
    # (Since we proved Agent(Cal) ~= BS(Trading))
    cal_days = trading_time_years * 365.0
    
    # 2. Agent expects days/30
    T_30 = cal_days / 30.0
    
    state = np.array([moneyness, T_30, perfect_position, DETECTED_VOL])
    state_scaled = scaler.transform(state.reshape(1, -1))
    
    state_tensor = torch.FloatTensor(state_scaled).to(device)
    with torch.no_grad():
        raw = actor(state_tensor).cpu().numpy().flatten()[0]
        
    return np.clip(0.5 * (raw + 1), 0, 1)

# ==========================================
# 3. GENERATE DATASET
# ==========================================
N_SAMPLES = 5000
print(f"\nGenerating {N_SAMPLES} samples for Symbolic Regression...")

# Bounds (Trading Days: 30 to 90 approx)
min_t_yr = 30 / 252.0
max_t_yr = 90 / 252.0

X_moneyness = np.random.uniform(0.85, 1.15, N_SAMPLES)
X_time_yr = np.random.uniform(min_t_yr, max_t_yr, N_SAMPLES)

y_agent = []

for i in range(N_SAMPLES):
    m = X_moneyness[i]
    t = X_time_yr[i]
    
    # Calculate Perfect Hedge (Benchmark) to remove friction
    bs_hedge = get_bs_delta(m, t, SIGMA_TRUE)
    
    # Query Agent with Perfect Hedge
    ag_val = query_agent(m, t, -bs_hedge)
    y_agent.append(ag_val)

y_agent = np.array(y_agent)
X_sr = np.stack([X_moneyness, X_time_yr], axis=1)

# ==========================================
# 4. RUN SYMBOLIC REGRESSION
# ==========================================
print("\n--- Running PySR ---")
model = PySRRegressor(
    niterations=200, 
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["log", "sqrt", "erf"], 
    constraints={"/": (-1, 9), "erf": 9, "log": 9, "sqrt": 9},
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
print(f"{'Moneyness':<10} | {'Days':<6} | {'BS Delta':<10} | {'Agent':<10} | {'SR Form':<10} | {'Ag-BS':<8}")
print("-" * 75)

# Generate 50 points
test_m = np.linspace(0.85, 1.15, 50)
test_t = np.linspace(min_t_yr, max_t_yr, 50) 
np.random.shuffle(test_t) # Shuffle time to mix things up

for i in range(50):
    m = test_m[i]
    t = test_t[i]
    d_trading = t * 252.0
    
    # 1. BS
    bs_val = get_bs_delta(m, t, SIGMA_TRUE)
    
    # 2. Agent
    agent_val = query_agent(m, t, -bs_val)
    
    # 3. Symbolic Regression Prediction
    sr_val = model.predict(np.array([[m, t]]))[0]
    
    diff = agent_val - bs_val
    
    print(f"{m:10.4f} | {d_trading:6.1f} | {bs_val:10.4f} | {agent_val:10.4f} | {sr_val:10.4f} | {diff:8.4f}")

# ==========================================
# 6. VISUALIZATION
# ==========================================
print("\nGenerating 3D Comparison...")

# Create Grid
plot_m = np.linspace(0.85, 1.15, 50)
plot_t = np.linspace(min_t_yr, max_t_yr, 50)
M_grid, T_grid = np.meshgrid(plot_m, plot_t)

Z_BS = np.zeros_like(M_grid)
Z_Agent = np.zeros_like(M_grid)
Z_SR = np.zeros_like(M_grid)

for i in range(M_grid.shape[0]):
    for j in range(M_grid.shape[1]):
        m = M_grid[i, j]
        t = T_grid[i, j]
        
        # 1. BS Benchmark
        bs = get_bs_delta(m, t, SIGMA_TRUE)
        Z_BS[i, j] = bs
        
        # 2. Agent
        Z_Agent[i, j] = query_agent(m, t, -bs)
        
        # 3. SR Formula
        Z_SR[i, j] = model.predict(np.array([[m, t]]))[0]

# Plotting
fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(M_grid, T_grid * 252, Z_BS, cmap=cm.plasma, alpha=0.8)
ax1.set_title('Black-Scholes Benchmark')
ax1.set_xlabel('Moneyness')
ax1.set_ylabel('Trading Days')
ax1.set_zlabel('Delta')

ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(M_grid, T_grid * 252, Z_Agent, cmap=cm.plasma, alpha=0.8)
ax2.set_title(f'Agent')
ax2.set_xlabel('Moneyness')
ax2.set_ylabel('Trading Days')
ax2.set_zlabel('Delta')

ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(M_grid, T_grid * 252, Z_SR, cmap=cm.plasma, alpha=0.8)
ax3.set_title('Symbolic Regression')
ax3.set_xlabel('Moneyness')
ax3.set_ylabel('Trading Days')
ax3.set_zlabel('Delta')

# plt.tight_layout()
plt.savefig(OUTPUT_FILE, format='pdf', dpi=300, bbox_inches='tight')
plt.show()