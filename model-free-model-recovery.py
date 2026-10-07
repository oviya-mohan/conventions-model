import numpy as np
import pandas as pd
from scipy.optimize import minimize
import random
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. SIMULATORS ---
def simulate_ind_rl(alpha, beta, gamma, n_trials=294):
    V = {i: 0.0 for i in range(1, 8)}
    trials = []
    for _ in range(n_trials):
        c1, c2 = random.sample(range(1, 8), 2)
        m = max(V[c1], V[c2])
        exp_v1, exp_v2 = np.exp(beta * (V[c1] - m)), np.exp(beta * (V[c2] - m))
        p_c1 = exp_v1 / (exp_v1 + exp_v2)
        choice = c1 if np.random.rand() < p_c1 else c2
        partner_choice = np.random.choice([c1, c2])
        reward = 1.0 if choice == partner_choice else 0.0
        trials.append({'options': (c1, c2), 'choice': choice, 'partner_choice': partner_choice})
        
        for k in V: V[k] *= gamma
        V[choice] += alpha * (reward - V[choice])
    return trials

def simulate_joint_rl_tied(alpha, beta, gamma, n_trials=294):
    """Joint RL using a single unified learning rate for self and partner"""
    V = {i: 0.0 for i in range(1, 8)}
    trials = []
    for _ in range(n_trials):
        c1, c2 = random.sample(range(1, 8), 2)
        m = max(V[c1], V[c2])
        exp_v1, exp_v2 = np.exp(beta * (V[c1] - m)), np.exp(beta * (V[c2] - m))
        p_c1 = exp_v1 / (exp_v1 + exp_v2)
        choice = c1 if np.random.rand() < p_c1 else c2
        partner_choice = np.random.choice([c1, c2])
        reward = 1.0 if choice == partner_choice else 0.0
        trials.append({'options': (c1, c2), 'choice': choice, 'partner_choice': partner_choice})
        
        for k in V: V[k] *= gamma
        V[choice] += alpha * (reward - V[choice])
        if reward == 0.0:
            V[partner_choice] += alpha * (1.0 - V[partner_choice])
    return trials

# --- 2. ESTIMATORS ---
def ind_rl_nll(params, data):
    alpha, beta, gamma = params
    if not (0 <= alpha <= 1) or not (0.1 <= beta <= 15) or not (0 <= gamma <= 1): return 1e9
    V = {i: 0.0 for i in range(1, 8)}
    nll = 0
    for row in data:
        c1, c2, choice, p_choice = row['options'][0], row['options'][1], row['choice'], row['partner_choice']
        m = max(V[c1], V[c2])
        exp_v1, exp_v2 = np.exp(beta * (V[c1] - m)), np.exp(beta * (V[c2] - m))
        p_c1 = exp_v1 / (exp_v1 + exp_v2)
        p_choice_val = p_c1 if choice == c1 else (1 - p_c1)
        nll -= np.log(max(p_choice_val, 1e-10))
        
        for k in V: V[k] *= gamma
        V[choice] += alpha * ((1.0 if choice == p_choice else 0.0) - V[choice])
    return nll

def joint_rl_tied_nll(params, data):
    alpha, beta, gamma = params
    if not (0 <= alpha <= 1) or not (0.1 <= beta <= 15) or not (0 <= gamma <= 1): return 1e9
    V = {i: 0.0 for i in range(1, 8)}
    nll = 0
    for row in data:
        c1, c2, choice, p_choice = row['options'][0], row['options'][1], row['choice'], row['partner_choice']
        m = max(V[c1], V[c2])
        exp_v1, exp_v2 = np.exp(beta * (V[c1] - m)), np.exp(beta * (V[c2] - m))
        p_c1 = exp_v1 / (exp_v1 + exp_v2)
        p_choice_val = p_c1 if choice == c1 else (1 - p_c1)
        nll -= np.log(max(p_choice_val, 1e-10))
        
        reward = 1.0 if choice == p_choice else 0.0
        for k in V: V[k] *= gamma
        V[choice] += alpha * (reward - V[choice])
        if reward == 0.0:
            V[p_choice] += alpha * (1.0 - V[p_choice])
    return nll

def get_winning_model(trials):
    bounds = [(0, 1), (0.1, 15), (0, 1)]
    
    # Ind RL (3 parameters)
    best_ind = minimize(ind_rl_nll, [0.1, 5.0, 0.8], args=(trials,), bounds=bounds).fun
    best_ind = min(best_ind, minimize(ind_rl_nll, [0.5, 1.0, 0.5], args=(trials,), bounds=bounds).fun)
    aic_ind = (2 * 3) + 2 * best_ind 
    
    # Tied Joint RL (3 parameters)
    best_joint = minimize(joint_rl_tied_nll, [0.1, 5.0, 0.8], args=(trials,), bounds=bounds).fun
    best_joint = min(best_joint, minimize(joint_rl_tied_nll, [0.5, 1.0, 0.5], args=(trials,), bounds=bounds).fun)
    aic_joint = (2 * 3) + 2 * best_joint 
    
    return 'Ind RL' if aic_ind < aic_joint else 'Joint RL'

# --- 3. RUN MODEL RECOVERY MATRIX ---
n_subjects_per_model = 50
matrix = {
    'True_Ind_RL': {'Fitted_Ind_RL': 0, 'Fitted_Joint_RL': 0},
    'True_Joint_RL': {'Fitted_Ind_RL': 0, 'Fitted_Joint_RL': 0}
}

print(f"Simulating and fitting {n_subjects_per_model * 2} subjects... (This will take a minute)")

for _ in range(n_subjects_per_model):
    trials = simulate_ind_rl(alpha=np.random.uniform(0.1, 0.9), beta=np.random.uniform(2.0, 10.0), gamma=np.random.uniform(0.5, 0.99))
    if get_winning_model(trials) == 'Ind RL':
        matrix['True_Ind_RL']['Fitted_Ind_RL'] += 1
    else:
        matrix['True_Ind_RL']['Fitted_Joint_RL'] += 1

for _ in range(n_subjects_per_model):
    trials = simulate_joint_rl_tied(alpha=np.random.uniform(0.1, 0.9), beta=np.random.uniform(2.0, 10.0), gamma=np.random.uniform(0.5, 0.99))
    if get_winning_model(trials) == 'Ind RL':
        matrix['True_Joint_RL']['Fitted_Ind_RL'] += 1
    else:
        matrix['True_Joint_RL']['Fitted_Joint_RL'] += 1

# --- 4. PLOT CONFUSION MATRIX ---
df_cm = pd.DataFrame([
    [matrix['True_Ind_RL']['Fitted_Ind_RL'] / n_subjects_per_model, matrix['True_Ind_RL']['Fitted_Joint_RL'] / n_subjects_per_model],
    [matrix['True_Joint_RL']['Fitted_Ind_RL'] / n_subjects_per_model, matrix['True_Joint_RL']['Fitted_Joint_RL'] / n_subjects_per_model]
], index=['True: Ind RL', 'True: Joint RL'], columns=['Fitted: Ind RL', 'Fitted: Joint RL'])

plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt=".0%", cmap="Blues", cbar=False, annot_kws={"size": 16})
plt.title("Model Recovery Confusion Matrix (Tied Learning Rates)", fontsize=16, pad=20)
plt.ylabel("Data Generating Model", fontsize=14, fontweight='bold')
plt.xlabel("Winning Model (Lowest AIC)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("confusion_matrix_tied.png", dpi=300)
print("\nMatrix complete! Saved as confusion_matrix_tied.png")