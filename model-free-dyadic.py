import numpy as np
from scipy.optimize import minimize
import random

# --- 1. TRUE DYADIC SIMULATOR ---
def simulate_dyad(agent1_type, params1, agent2_type, params2, n_trials=294):
    """
    Simulates a true interacting dyad.
    agent_type: 'Ind', 'Joint', or 'Random'
    params: tuple of (alpha, beta, gamma). Pass None for 'Random'.
    """
    V1 = {i: 0.0 for i in range(1, 8)}
    V2 = {i: 0.0 for i in range(1, 8)}
    trials1, trials2 = [], []
    
    for _ in range(n_trials):
        c1, c2 = random.sample(range(1, 8), 2)
        
        # --- Agent 1 Choice ---
        if agent1_type == 'Random':
            choice1 = np.random.choice([c1, c2])
        else:
            m1 = max(V1[c1], V1[c2])
            p_c1_1 = np.exp(params1[1] * (V1[c1] - m1)) / (np.exp(params1[1] * (V1[c1] - m1)) + np.exp(params1[1] * (V1[c2] - m1)))
            choice1 = c1 if np.random.rand() < p_c1_1 else c2
            
        # --- Agent 2 Choice ---
        if agent2_type == 'Random':
            choice2 = np.random.choice([c1, c2])
        else:
            m2 = max(V2[c1], V2[c2])
            p_c1_2 = np.exp(params2[1] * (V2[c1] - m2)) / (np.exp(params2[1] * (V2[c1] - m2)) + np.exp(params2[1] * (V2[c2] - m2)))
            choice2 = c1 if np.random.rand() < p_c1_2 else c2

        # --- Evaluate Interaction ---
        reward = 1.0 if choice1 == choice2 else 0.0
        
        trials1.append({'options': (c1, c2), 'choice': choice1, 'partner_choice': choice2})
        trials2.append({'options': (c1, c2), 'choice': choice2, 'partner_choice': choice1})
        
        # --- Update Agent 1 ---
        if agent1_type != 'Random':
            a1, _, g1 = params1
            for k in V1: V1[k] *= g1 # Decay
            V1[choice1] += a1 * (reward - V1[choice1]) # Self Update
            if agent1_type == 'Joint' and reward == 0.0:
                V1[choice2] += a1 * (1.0 - V1[choice2]) # Partner Update
                
        # --- Update Agent 2 ---
        if agent2_type != 'Random':
            a2, _, g2 = params2
            for k in V2: V2[k] *= g2 # Decay
            V2[choice2] += a2 * (reward - V2[choice2]) # Self Update
            if agent2_type == 'Joint' and reward == 0.0:
                V2[choice1] += a2 * (1.0 - V2[choice1]) # Partner Update

    return trials1, trials2

# --- 2. ESTIMATORS (Unchanged) ---
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
    bounds = [(0, 1), (0.1, 15), (0, 1)] 
    estimator = ind_rl_nll if model_type == 'Ind' else joint_rl_tied_nll
    for _ in range(n_starts):
        x0 = [np.random.uniform(0.1, 0.9), np.random.uniform(1.0, 10.0), np.random.uniform(0.5, 1.0)]
        res = minimize(estimator, x0, args=(trials,), bounds=bounds, method='L-BFGS-B')
        if res.fun < best_nll:
            best_nll, best_params = res.fun, res.x
    return best_params

# --- 3. EXECUTE DYADIC TESTS ---
print("--- TEST 1: Joint RL vs Joint RL (Optimal Coordination) ---")
true_p1_params = (0.60, 6.0, 0.85)
true_p2_params = (0.40, 4.0, 0.90)

t1_joint, t2_joint = simulate_dyad('Joint', true_p1_params, 'Joint', true_p2_params)

rec_p1_joint = recover_params(t1_joint, 'Joint')
rec_p2_joint = recover_params(t2_joint, 'Joint')

print(f"P1 (Joint) True: {true_p1_params} | Rec: {np.round(rec_p1_joint, 2)}")
print(f"P2 (Joint) True: {true_p2_params} | Rec: {np.round(rec_p2_joint, 2)}")


print("\n--- TEST 2: Ind RL vs Joint RL (Asymmetric Coordination) ---")
true_p1_ind = (0.70, 5.0, 0.80)
true_p2_joint = (0.50, 4.5, 0.85)

t1_mix, t2_mix = simulate_dyad('Ind', true_p1_ind, 'Joint', true_p2_joint)

rec_p1_mix = recover_params(t1_mix, 'Ind')
rec_p2_mix = recover_params(t2_mix, 'Joint')

print(f"P1 (Ind)   True: {true_p1_ind} | Rec: {np.round(rec_p1_mix, 2)}")
print(f"P2 (Joint) True: {true_p2_joint} | Rec: {np.round(rec_p2_mix, 2)}")