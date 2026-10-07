import numpy as np
import pandas as pd
from scipy.optimize import minimize
import random
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. DYADIC SIMULATOR ---
def simulate_dyad(agent1_type, params1, agent2_type, params2, n_trials=294):
    V1 = {i: 0.0 for i in range(1, 8)}
    V2 = {i: 0.0 for i in range(1, 8)}
    trials1, trials2 = [], []
    
    for _ in range(n_trials):
        c1, c2 = random.sample(range(1, 8), 2)
        
        m1 = max(V1[c1], V1[c2])
        p_c1_1 = np.exp(params1[1] * (V1[c1] - m1)) / (np.exp(params1[1] * (V1[c1] - m1)) + np.exp(params1[1] * (V1[c2] - m1)))
        choice1 = c1 if np.random.rand() < p_c1_1 else c2
            
        m2 = max(V2[c1], V2[c2])
        p_c1_2 = np.exp(params2[1] * (V2[c1] - m2)) / (np.exp(params2[1] * (V2[c1] - m2)) + np.exp(params2[1] * (V2[c2] - m2)))
        choice2 = c1 if np.random.rand() < p_c1_2 else c2

        reward = 1.0 if choice1 == choice2 else 0.0
        
        trials1.append({'options': (c1, c2), 'choice': choice1, 'partner_choice': choice2})
        trials2.append({'options': (c1, c2), 'choice': choice2, 'partner_choice': choice1})
        
        a1, _, g1 = params1
        for k in V1: V1[k] *= g1
        V1[choice1] += a1 * (reward - V1[choice1])
        if agent1_type == 'Joint' and reward == 0.0:
            V1[choice2] += a1 * (1.0 - V1[choice2])
                
        a2, _, g2 = params2
        for k in V2: V2[k] *= g2
        V2[choice2] += a2 * (reward - V2[choice2])
        if agent2_type == 'Joint' and reward == 0.0:
            V2[choice1] += a2 * (1.0 - V2[choice1])

    return trials1, trials2

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
    # Use 3 random starts to balance speed and accuracy in a large loop
    best_ind_nll, best_joint_nll = np.inf, np.inf
    
    for _ in range(3):
        x0 = [np.random.uniform(0.1, 0.9), np.random.uniform(1.0, 10.0), np.random.uniform(0.5, 0.99)]
        res_ind = minimize(ind_rl_nll, x0, args=(trials,), bounds=bounds, method='L-BFGS-B')
        if res_ind.fun < best_ind_nll: best_ind_nll = res_ind.fun
            
        res_joint = minimize(joint_rl_tied_nll, x0, args=(trials,), bounds=bounds, method='L-BFGS-B')
        if res_joint.fun < best_joint_nll: best_joint_nll = res_joint.fun

    aic_ind = (2 * 3) + 2 * best_ind_nll
    aic_joint = (2 * 3) + 2 * best_joint_nll
    return 'Ind' if aic_ind < aic_joint else 'Joint'

# --- 3. EXHAUSTIVE DYADIC MATRIX ---
n_dyads_per_condition = 20 # Total 120 subjects simulated and fit
matrix = {
    'True_Ind': {'Fitted_Ind': 0, 'Fitted_Joint': 0},
    'True_Joint': {'Fitted_Ind': 0, 'Fitted_Joint': 0}
}

print(f"Simulating and fitting {n_dyads_per_condition * 3} dyads ({n_dyads_per_condition * 6} subjects)...")

def draw_params(): return (np.random.uniform(0.2, 0.8), np.random.uniform(2.0, 8.0), np.random.uniform(0.7, 0.99))

# 1. Ind vs Ind
for _ in range(n_dyads_per_condition):
    t1, t2 = simulate_dyad('Ind', draw_params(), 'Ind', draw_params())
    matrix['True_Ind'][f"Fitted_{get_winning_model(t1)}"] += 1
    matrix['True_Ind'][f"Fitted_{get_winning_model(t2)}"] += 1

# 2. Joint vs Joint
for _ in range(n_dyads_per_condition):
    t1, t2 = simulate_dyad('Joint', draw_params(), 'Joint', draw_params())
    matrix['True_Joint'][f"Fitted_{get_winning_model(t1)}"] += 1
    matrix['True_Joint'][f"Fitted_{get_winning_model(t2)}"] += 1

# 3. Ind vs Joint
for _ in range(n_dyads_per_condition):
    t1, t2 = simulate_dyad('Ind', draw_params(), 'Joint', draw_params())
    matrix['True_Ind'][f"Fitted_{get_winning_model(t1)}"] += 1
    matrix['True_Joint'][f"Fitted_{get_winning_model(t2)}"] += 1

# --- 4. PLOT CONFUSION MATRIX ---
total_true_ind = matrix['True_Ind']['Fitted_Ind'] + matrix['True_Ind']['Fitted_Joint']
total_true_joint = matrix['True_Joint']['Fitted_Ind'] + matrix['True_Joint']['Fitted_Joint']

df_cm = pd.DataFrame([
    [matrix['True_Ind']['Fitted_Ind'] / total_true_ind, matrix['True_Ind']['Fitted_Joint'] / total_true_ind],
    [matrix['True_Joint']['Fitted_Ind'] / total_true_joint, matrix['True_Joint']['Fitted_Joint'] / total_true_joint]
], index=['True: Ind RL', 'True: Joint RL'], columns=['Fitted: Ind RL', 'Fitted: Joint RL'])

plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt=".0%", cmap="Blues", cbar=False, annot_kws={"size": 16})
plt.title("Dyadic Model Recovery Matrix", fontsize=16, pad=20)
plt.ylabel("Data Generating Model", fontsize=14, fontweight='bold')
plt.xlabel("Winning Model (Lowest AIC)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("dyadic_confusion_matrix.png", dpi=300)
print("\nMatrix complete! Saved as dyadic_confusion_matrix.png")