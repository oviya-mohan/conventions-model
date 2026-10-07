import numpy as np
import pandas as pd
from scipy.optimize import minimize
import random
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --- 1. VECTORIZED RSA LOGIC (3 Parameters) ---
def get_rsa_probs(c1, c2, V, alpha, lapse, level):
    u_base = np.array([V[c1], V[c2]])
    
    if level == 0:
        u_final = u_base
    else:
        m0 = np.max(u_base)
        p0 = np.exp(alpha * (u_base - m0)) / np.sum(np.exp(alpha * (u_base - m0)))
        if level == 1:
            u_final = u_base + np.log(np.maximum(p0, 1e-10))
        else: 
            u1 = u_base + np.log(np.maximum(p0, 1e-10))
            m1 = np.max(u1)
            p1 = np.exp(alpha * (u1 - m1)) / np.sum(np.exp(alpha * (u1 - m1)))
            u_final = u_base + np.log(np.maximum(p1, 1e-10))
            
    m_fin = np.max(u_final)
    p_fin = np.exp(alpha * (u_final - m_fin)) / np.sum(np.exp(alpha * (u_final - m_fin)))
    
    p_final = (1.0 - lapse) * p_fin + (lapse / 2.0)
    return p_final[0]

def simulate_rsa_dyad(params1, level1, params2, level2, n_trials=294):
    V1 = {i: 0.0 for i in range(1, 8)}
    V2 = {i: 0.0 for i in range(1, 8)}
    trials1, trials2 = [], []
    
    for _ in range(n_trials):
        c1, c2 = random.sample(range(1, 8), 2)
        
        p_c1_1 = get_rsa_probs(c1, c2, V1, params1[0], params1[1], level1)
        choice1 = c1 if np.random.rand() < p_c1_1 else c2
        
        p_c1_2 = get_rsa_probs(c1, c2, V2, params2[0], params2[1], level2)
        choice2 = c1 if np.random.rand() < p_c1_2 else c2
        
        reward = 1.0 if choice1 == choice2 else 0.0
        
        trials1.append({'options': (c1, c2), 'choice': choice1, 'partner_choice': choice2})
        trials2.append({'options': (c1, c2), 'choice': choice2, 'partner_choice': choice1})
        
        for k in V1: V1[k] *= params1[2]
        if reward == 1.0: V1[choice1] += 1.0 
        
        for k in V2: V2[k] *= params2[2]
        if reward == 1.0: V2[choice2] += 1.0 
        
    return trials1, trials2

# --- 2. ESTIMATOR & MODEL SELECTION ---
def rsa_nll(params, data, level):
    alpha, lapse, gamma = params
    if not (0.1 <= alpha <= 15) or not (0 <= lapse <= 0.5) or not (0 <= gamma <= 1): 
        return 1e9
        
    V = {i: 0.0 for i in range(1, 8)}
    nll = 0
    
    for row in data:
        c1, c2, choice, p_choice = row['options'][0], row['options'][1], row['choice'], row['partner_choice']
        p_c1 = get_rsa_probs(c1, c2, V, alpha, lapse, level)
        p_choice_val = p_c1 if choice == c1 else (1.0 - p_c1)
        
        nll -= np.log(max(p_choice_val, 1e-10))
        
        reward = 1.0 if choice == p_choice else 0.0
        for k in V: V[k] *= gamma
        if reward == 1.0: V[choice] += 1.0
        
    return nll

def get_winning_rsa_model(trials):
    """Fits L0, L1, and L2 models and returns the level with the lowest AIC."""
    bounds = [(0.1, 15), (0, 0.5), (0, 1)]
    best_nlls = {0: np.inf, 1: np.inf, 2: np.inf}
    
    for level in [0, 1, 2]:
        # Using 3 random starts per model to balance speed and accuracy in a large loop
        for _ in range(3):
            x0 = [np.random.uniform(1.0, 8.0), np.random.uniform(0.01, 0.2), np.random.uniform(0.5, 0.99)]
            res = minimize(rsa_nll, x0, args=(trials, level), bounds=bounds, method='L-BFGS-B')
            if res.fun < best_nlls[level]:
                best_nlls[level] = res.fun
                
    # All models have 3 parameters, so AIC penalty is (2 * 3)
    aics = {lvl: 6 + (2 * nll) for lvl, nll in best_nlls.items()}
    winner = min(aics, key=aics.get)
    return f"Fitted_L{winner}"

# --- 3. EXHAUSTIVE 3x3 DYADIC MATRIX ---
n_dyads = 15 # Simulates 30 subjects per condition (90 subjects total)
matrix = {
    'True_L0': {'Fitted_L0': 0, 'Fitted_L1': 0, 'Fitted_L2': 0},
    'True_L1': {'Fitted_L0': 0, 'Fitted_L1': 0, 'Fitted_L2': 0},
    'True_L2': {'Fitted_L0': 0, 'Fitted_L1': 0, 'Fitted_L2': 0}
}

print(f"Simulating and fitting {n_dyads * 3} dyads ({n_dyads * 6} subjects)... This will take a few minutes.")

def draw_params(): 
    return (np.random.uniform(3.0, 8.0), np.random.uniform(0.01, 0.15), np.random.uniform(0.75, 0.95))

# Condition 1: True L0 (Simulate L0 vs L0)
for _ in range(n_dyads):
    t1, t2 = simulate_rsa_dyad(draw_params(), 0, draw_params(), 0)
    matrix['True_L0'][get_winning_rsa_model(t1)] += 1
    matrix['True_L0'][get_winning_rsa_model(t2)] += 1

# Condition 2: True L1 (Simulate L1 vs L1)
for _ in range(n_dyads):
    t1, t2 = simulate_rsa_dyad(draw_params(), 1, draw_params(), 1)
    matrix['True_L1'][get_winning_rsa_model(t1)] += 1
    matrix['True_L1'][get_winning_rsa_model(t2)] += 1

# Condition 3: True L2 (Simulate L2 vs L2)
for _ in range(n_dyads):
    t1, t2 = simulate_rsa_dyad(draw_params(), 2, draw_params(), 2)
    matrix['True_L2'][get_winning_rsa_model(t1)] += 1
    matrix['True_L2'][get_winning_rsa_model(t2)] += 1

# --- 4. PLOT 3x3 CONFUSION MATRIX ---
# Convert raw counts to percentages based on total subjects per True category (n_dyads * 2)
n_subjects = n_dyads * 2
df_cm = pd.DataFrame([
    [matrix['True_L0']['Fitted_L0'] / n_subjects, matrix['True_L0']['Fitted_L1'] / n_subjects, matrix['True_L0']['Fitted_L2'] / n_subjects],
    [matrix['True_L1']['Fitted_L0'] / n_subjects, matrix['True_L1']['Fitted_L1'] / n_subjects, matrix['True_L1']['Fitted_L2'] / n_subjects],
    [matrix['True_L2']['Fitted_L0'] / n_subjects, matrix['True_L2']['Fitted_L1'] / n_subjects, matrix['True_L2']['Fitted_L2'] / n_subjects]
], index=['True: L0', 'True: L1', 'True: L2'], columns=['Fitted: L0', 'Fitted: L1', 'Fitted: L2'])

plt.figure(figsize=(9, 7))
sns.heatmap(df_cm, annot=True, fmt=".0%", cmap="Blues", cbar=False, annot_kws={"size": 16})
plt.title("RSA 3x3 Model Recovery Matrix", fontsize=18, pad=20)
plt.ylabel("Data Generating Model", fontsize=14, fontweight='bold')
plt.xlabel("Winning Model (Lowest AIC)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("rsa_confusion_matrix_3x3.png", dpi=300)
print("\nMatrix complete! Saved as rsa_confusion_matrix_3x3.png")