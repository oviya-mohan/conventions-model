import numpy as np
from scipy.optimize import minimize
import random

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
        
        # 1. Decay all values
        for k in V: V[k] *= gamma
        # 2. Update chosen value
        V[choice] += alpha * (reward - V[choice])
    return trials

def simulate_joint_rl_tied(alpha, beta, gamma, n_trials=294):
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
        
        # 1. Decay all values
        for k in V: V[k] *= gamma
        # 2. Update chosen value
        V[choice] += alpha * (reward - V[choice])
        # 3. Update partner's choice using the SAME alpha
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

def recover_params(trials, model_type, n_starts=15):
    best_nll = np.inf
    best_params = None
    bounds = [(0, 1), (0.1, 15), (0, 1)] # alpha, beta, gamma
    
    estimator = ind_rl_nll if model_type == 'Ind RL' else joint_rl_tied_nll
    
    for _ in range(n_starts):
        x0 = [np.random.uniform(0.1, 0.9), np.random.uniform(1.0, 10.0), np.random.uniform(0.5, 1.0)]
        res = minimize(estimator, x0, args=(trials,), bounds=bounds, method='L-BFGS-B')
        if res.fun < best_nll:
            best_nll, best_params = res.fun, res.x
            
    return best_params

# --- 3. EXECUTE TESTS ---
print("Running Individual RL Recovery...")
true_ind_params = (0.60, 5.0, 0.85)
ind_trials = simulate_ind_rl(*true_ind_params)
rec_ind = recover_params(ind_trials, 'Ind RL')

print("Running Tied Joint RL Recovery...")
true_joint_params = (0.50, 4.0, 0.85)
joint_trials = simulate_joint_rl_tied(*true_joint_params)
rec_joint = recover_params(joint_trials, 'Joint RL')

print("\n--- RESULTS ---")
print(f"Ind RL       | True: {true_ind_params} | Recovered: {np.round(rec_ind, 2)}")
print(f"Tied Joint RL| True: {true_joint_params} | Recovered: {np.round(rec_joint, 2)}")