import numpy as np
from scipy.optimize import minimize
import random
import warnings
warnings.filterwarnings('ignore')

# --- 1. VECTORIZED RSA LOGIC (3 Parameters) ---
def get_rsa_probs(c1, c2, V, alpha, lapse, level):
    """Calculates L0, L1, or L2 probabilities instantly using vectorized math."""
    u_base = np.array([V[c1], V[c2]])
    
    # Mental Simulations
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
            
    # Final Softmax & Lapse Rate (Trembling Hand)
    m_fin = np.max(u_final)
    p_fin = np.exp(alpha * (u_final - m_fin)) / np.sum(np.exp(alpha * (u_final - m_fin)))
    
    p_final = (1.0 - lapse) * p_fin + (lapse / 2.0)
    return p_final[0]

# --- 2. TRUE DYADIC SIMULATOR ---
def simulate_rsa_dyad(params1, level1, params2, level2, n_trials=294):
    """Simulates two RSA agents interacting and co-adapting."""
    V1 = {i: 0.0 for i in range(1, 8)}
    V2 = {i: 0.0 for i in range(1, 8)}
    trials1, trials2 = [], []
    
    for _ in range(n_trials):
        c1, c2 = random.sample(range(1, 8), 2)
        
        # Agent 1 Choice 
        p_c1_1 = get_rsa_probs(c1, c2, V1, params1[0], params1[1], level1)
        choice1 = c1 if np.random.rand() < p_c1_1 else c2
        
        # Agent 2 Choice 
        p_c1_2 = get_rsa_probs(c1, c2, V2, params2[0], params2[1], level2)
        choice2 = c1 if np.random.rand() < p_c1_2 else c2
        
        reward = 1.0 if choice1 == choice2 else 0.0
        
        trials1.append({'options': (c1, c2), 'choice': choice1, 'partner_choice': choice2})
        trials2.append({'options': (c1, c2), 'choice': choice2, 'partner_choice': choice1})
        
        # Memory Updates (gamma is index 2)
        for k in V1: V1[k] *= params1[2]
        if reward == 1.0: V1[choice1] += 1.0 
        
        for k in V2: V2[k] *= params2[2]
        if reward == 1.0: V2[choice2] += 1.0 
        
    return trials1, trials2

# --- 3. ESTIMATOR ---
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

def recover_rsa_params(trials, level, n_starts=15):
    best_nll = np.inf
    best_params = None
    bounds = [(0.1, 15), (0, 0.5), (0, 1)] 
    
    for _ in range(n_starts):
        x0 = [np.random.uniform(1.0, 8.0), np.random.uniform(0.01, 0.2), np.random.uniform(0.5, 0.99)]
        res = minimize(rsa_nll, x0, args=(trials, level), bounds=bounds, method='L-BFGS-B')
        if res.fun < best_nll:
            best_nll = res.fun
            best_params = res.x
            
    return best_params

# --- 4. EXECUTE DYADIC RECOVERY TESTS ---
print("--- TEST 1: L0 vs L0 (Naive Coordination) ---")
true_L0_A = (4.0, 0.05, 0.85)
true_L0_B = (5.0, 0.08, 0.90)
t_L0_A, t_L0_B = simulate_rsa_dyad(true_L0_A, 0, true_L0_B, 0)

rec_L0_A = recover_rsa_params(t_L0_A, level=0)
print(f"L0 (Agent A) True: {true_L0_A} | Rec: {np.round(rec_L0_A, 2)}")

print("\n--- TEST 2: L1 vs L0 (Pragmatic Coordination) ---")
true_L1_A = (5.0, 0.10, 0.90)
true_L0_C = (4.0, 0.05, 0.85)
t_L1_A, t_L0_C = simulate_rsa_dyad(true_L1_A, 1, true_L0_C, 0)

rec_L1_A = recover_rsa_params(t_L1_A, level=1)
print(f"L1 (Agent A) True: {true_L1_A} | Rec: {np.round(rec_L1_A, 2)}")

print("\n--- TEST 3: L2 vs L1 (Meta-Pragmatic Coordination) ---")
true_L2_A = (6.0, 0.05, 0.80)
true_L1_B = (5.0, 0.10, 0.90)
t_L2_A, t_L1_B = simulate_rsa_dyad(true_L2_A, 2, true_L1_B, 1)

rec_L2_A = recover_rsa_params(t_L2_A, level=2)
print(f"L2 (Agent A) True: {true_L2_A} | Rec: {np.round(rec_L2_A, 2)}")

print("\n--- TEST 4: L1 vs L1 (Symmetric Pragmatic) ---")
true_L1_C = (4.5, 0.08, 0.85)
true_L1_D = (5.5, 0.12, 0.88)
t_L1_C, t_L1_D = simulate_rsa_dyad(true_L1_C, 1, true_L1_D, 1)

rec_L1_C = recover_rsa_params(t_L1_C, level=1)
print(f"L1 (Agent A) True: {true_L1_C} | Rec: {np.round(rec_L1_C, 2)}")

print("\n--- TEST 5: L2 vs L2 (Symmetric Meta-Pragmatic) ---")
true_L2_B = (6.5, 0.04, 0.82)
true_L2_C = (7.0, 0.06, 0.86)
t_L2_B, t_L2_C = simulate_rsa_dyad(true_L2_B, 2, true_L2_C, 2)

rec_L2_B = recover_rsa_params(t_L2_B, level=2)
print(f"L2 (Agent A) True: {true_L2_B} | Rec: {np.round(rec_L2_B, 2)}")

print("\n--- TEST 6: L2 vs L0 (Meta-Pragmatic??) ---")
true_L2_D = (6.5, 0.04, 0.82)
true_L0_C = (4.5, 0.08, 0.85)
t_L2_D, t_L0_C = simulate_rsa_dyad(true_L2_D, 2, true_L0_C, 1)

rec_L2_D = recover_rsa_params(t_L2_D, level=2)
print(f"L2 (Agent A) True: {true_L2_D} | Rec: {np.round(rec_L2_D, 2)}")