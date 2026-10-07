import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 1. Model fitting functions
def fit_baseline(data):
    return -len(data) * np.log(0.5)

def ind_rl_nll(params, data):
    alpha, beta = params
    if alpha < 0 or alpha > 1 or beta < 0: return 1e9
    V = {i: 0.0 for i in range(1, 8)}
    nll = 0
    for row in data:
        c1, c2, choice, p_choice = row['options'][0], row['options'][1], row['choice'], row['partner_choice']
        v1, v2 = V[c1], V[c2]
        m = max(v1, v2)
        exp_v1, exp_v2 = np.exp(beta * (v1 - m)), np.exp(beta * (v2 - m))
        p_c1 = exp_v1 / (exp_v1 + exp_v2)
        p_choice_val = p_c1 if choice == c1 else (1 - p_c1)
        nll -= np.log(max(p_choice_val, 1e-10))
        V[choice] += alpha * (1.0 if choice == p_choice else 0.0 - V[choice])
    return nll

def joint_rl_nll(params, data):
    alpha_self, alpha_partner, beta = params
    if alpha_self < 0 or alpha_self > 1 or alpha_partner < 0 or alpha_partner > 1 or beta < 0: return 1e9
    V = {i: 0.0 for i in range(1, 8)}
    nll = 0
    for row in data:
        c1, c2, choice, p_choice = row['options'][0], row['options'][1], row['choice'], row['partner_choice']
        v1, v2 = V[c1], V[c2]
        m = max(v1, v2)
        exp_v1, exp_v2 = np.exp(beta * (v1 - m)), np.exp(beta * (v2 - m))
        p_c1 = exp_v1 / (exp_v1 + exp_v2)
        p_choice_val = p_c1 if choice == c1 else (1 - p_c1)
        nll -= np.log(max(p_choice_val, 1e-10))
        reward = 1.0 if choice == p_choice else 0.0
        V[choice] += alpha_self * (reward - V[choice])
        if reward == 0.0:
            V[p_choice] += alpha_partner * (1.0 - V[p_choice])
    return nll

def run_rolling_window(trials, window_size, total_trials):
    weights = {'Baseline': [], 'Ind RL': [], 'Joint RL': []}
    for i in range(total_trials - window_size + 1):
        window = trials[i:i+window_size]
        
        aic_base = 2 * fit_baseline(window)
        aic_ind = 4 + 2 * minimize(ind_rl_nll, [0.1, 1.0], args=(window,), bounds=[(0,1), (0, 20)]).fun
        aic_joint = 6 + 2 * minimize(joint_rl_nll, [0.1, 0.1, 1.0], args=(window,), bounds=[(0,1), (0,1), (0, 20)]).fun
        
        aics = np.array([aic_base, aic_ind, aic_joint])
        delta = aics - np.min(aics)
        w = np.exp(-0.5 * delta) / np.sum(np.exp(-0.5 * delta))
        
        weights['Baseline'].append(w[0])
        weights['Ind RL'].append(w[1])
        weights['Joint RL'].append(w[2])
    return weights

# 2. Iterate through all pairs
pairs = [
    'NI_OT_xq913l4v', 'I_TO_apqr0rpe', 'NI_TO_agx6z9dn', 'NI_TO_7wshhrbw',
    'I_OT_c8j2ro94', 'I_TO_4r55skxj', 'I_OT_h9uh3mul', 'I_OT_g4xpbpj6',
    'NI_OT_sh8cf924', 'NI_TO_wiml9wm0', 'NI_OT_ldhtljr9', 'NI_OT_eo2uzk2s',
    'I_TO_h2uit3zp', 'I_TO_kuae9j3j', 'NI_TO_4bwl0qbj', 'I_OT_3ix0zaem',
    'NI_OT_rrlxmpp5', 'I_TO_t7x3vvi1', 'NI_TO_3ooaxiup', 'I_OT_tvge9zyu',
    'NI_OT_fpghgezy', 'NI_TO_d3nyyc5v', 'I_TO_bv42zb0r', 'NI_OT_ut8qatoj',
    'I_OT_aw71h5pi', 'NI_TO_98fqxtm8', 'I_TO_ohn5xo30', 'I_OT_v395919d',
    'NI_TO_4fei2let', 'NI_OT_scc6ut4m', 'I_TO_w08oe7pq', 'I_OT_fhfu2taw',
    'NI_TO_5393z70p', 'NI_OT_f46z2j09', 'I_OT_14f0zewk', 'NI_OT_3zl2di7o',
    'NI_TO_qvuayn21', 'I_TO_29zq6soa', 'I_OT_axg2ydsa', 'I_TO_lx66ptw4'
]

window_size = 42

for pair in pairs:
    condition = pair[:-9]
    folder = pair[-8:]
    folder_path = f"data/{condition}/{folder}"
    file_name = f"{folder_path}/forelo_1.csv"
    
    if not os.path.exists(file_name):
        print(f"Skipping {pair}: File not found ({file_name})")
        continue
        
    print(f"Processing {pair}...")
    df = pd.read_csv(file_name)
    
    p1_trials, p2_trials = [], []
    for r in df['round_number'].unique():
        subset = df[df['round_number'] == r]
        if len(subset) < 2: continue
        p1 = subset[subset['participant'] == 1].iloc[0]
        p2 = subset[subset['participant'] == 2].iloc[0]
        c1, c2 = int(p1['response']), int(p1['loser'])
        
        p1_trials.append({'options': (c1, c2), 'choice': int(p1['response']), 'partner_choice': int(p2['response'])})
        p2_trials.append({'options': (c1, c2), 'choice': int(p2['response']), 'partner_choice': int(p1['response'])})

    total_trials = len(p1_trials)
    if total_trials < window_size:
        print(f"Skipping {pair}: Not enough trials.")
        continue

    # Run rolling windows
    p1_weights = run_rolling_window(p1_trials, window_size, total_trials)
    p2_weights = run_rolling_window(p2_trials, window_size, total_trials)

    # 3. Plot and save
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(window_size, total_trials + 1)

    # P1 Plot
    ax1.plot(x, p1_weights['Baseline'], label='Baseline', color='#777777', lw=2)
    ax1.plot(x, p1_weights['Ind RL'], label='Ind RL', color='#4DB6AC', lw=2)
    ax1.plot(x, p1_weights['Joint RL'], label='Joint RL', color='#FFB74D', lw=2)
    ax1.set_title(f'{pair} - Participant 1: Strategy Likelihood', fontsize=14, pad=15)
    ax1.set_ylabel('Model Probability', fontsize=11)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(frameon=False, loc='upper left')

    # P2 Plot
    ax2.plot(x, p2_weights['Baseline'], label='Baseline', color='#777777', lw=2)
    ax2.plot(x, p2_weights['Ind RL'], label='Ind RL', color='#4DB6AC', lw=2)
    ax2.plot(x, p2_weights['Joint RL'], label='Joint RL', color='#FFB74D', lw=2)
    ax2.set_title(f'{pair} - Participant 2: Strategy Likelihood', fontsize=14, pad=15)
    ax2.set_xlabel('Trial Number (End of 42-Trial Window)', fontsize=11)
    ax2.set_ylabel('Model Probability', fontsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    
    # Save inside the pair's specific folder
    save_path = os.path.join(folder_path, f'rolling_window_{pair}.png')
    plt.savefig(save_path, transparent=True, dpi=300)
    plt.close()