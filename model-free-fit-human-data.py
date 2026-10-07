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
        
        # Decay and Update
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
        
        # Decay and Tied Update
        for k in V: V[k] *= gamma
        V[choice] += alpha * (reward - V[choice])
        if reward == 0.0:
            V[p_choice] += alpha * (1.0 - V[p_choice])
    return nll

def evaluate_participant_weights(trials):
    aic_base = 2 * fit_baseline(trials) # Baseline has 0 parameters
    bounds = [(0, 1), (0.1, 15), (0, 1)]
    
    # Evaluate Individual RL with 3 random starts to avoid local minima
    best_ind_nll = np.inf
    for _ in range(3):
        x0 = [np.random.uniform(0.1, 0.9), np.random.uniform(1.0, 10.0), np.random.uniform(0.5, 0.99)]
        res = minimize(ind_rl_nll, x0, args=(trials,), bounds=bounds, method='L-BFGS-B')
        if res.fun < best_ind_nll:
            best_ind_nll = res.fun
    aic_ind = (2 * 3) + 2 * best_ind_nll # Penalty of 6 for 3 parameters
    
    # Evaluate Tied Joint RL with 3 random starts
    best_joint_nll = np.inf
    for _ in range(3):
        x0 = [np.random.uniform(0.1, 0.9), np.random.uniform(1.0, 10.0), np.random.uniform(0.5, 0.99)]
        res = minimize(joint_rl_tied_nll, x0, args=(trials,), bounds=bounds, method='L-BFGS-B')
        if res.fun < best_joint_nll:
            best_joint_nll = res.fun
    aic_joint = (2 * 3) + 2 * best_joint_nll # Penalty of 6 for 3 parameters
    
    aics = np.array([aic_base, aic_ind, aic_joint])
    delta = aics - np.min(aics)
    weights = np.exp(-0.5 * delta) / np.sum(np.exp(-0.5 * delta))
    
    return weights

# 2. Extract Data
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

results = []

for pair in pairs:
    condition = pair[:-9]
    folder = pair[-8:]
    file_name = f"data/{condition}/{folder}/forelo_1.csv"
    
    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        continue
        
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

    print(f"Fitting {pair}...")
    p1_w = evaluate_participant_weights(p1_trials)
    p2_w = evaluate_participant_weights(p2_trials)
    
    results.append({'Subject': f"{folder}_1", 'Condition': condition, 'Baseline': p1_w[0], 'Ind RL': p1_w[1], 'Joint RL': p1_w[2]})
    results.append({'Subject': f"{folder}_2", 'Condition': condition, 'Baseline': p2_w[0], 'Ind RL': p2_w[1], 'Joint RL': p2_w[2]})

df_results = pd.DataFrame(results)

# 3. Generating the Plots
plt.style.use('ggplot')
conditions_list = ['I_OT', 'NI_OT', 'I_TO', 'NI_TO']
colors = ['#E0E0E0', '#888888', '#000000'] 
labels = ['Baseline', 'Individual RL', 'Joint RL']

for cond in conditions_list:
    subset = df_results[df_results['Condition'] == cond]
    if subset.empty:
        continue
        
    ax = subset.set_index('Subject')[['Baseline', 'Ind RL', 'Joint RL']].plot(
        kind='bar', stacked=True, color=colors, figsize=(14, 8), width=0.9
    )
    
    plt.title(f'Strategy Probabilities: {cond}', fontsize=18, pad=20)
    plt.ylabel('probability', fontsize=14, color='black', fontweight='bold')
    plt.xlabel('subject', fontsize=14, color='black')
    
    plt.ylim(0, 1.0)
    plt.yticks([0.0, 0.25, 0.50, 0.75, 1.00], ['0.00', '0.25', '0.50', '0.75', '1.00'], color='black')
    
    ax.set_xticklabels(subset['Subject'], rotation=90, fontsize=10, color='black')
    ax.tick_params(axis='x', bottom=True, length=5, color='black')
    
    ax.yaxis.grid(True, color='white', linestyle='-', linewidth=1.5)
    ax.xaxis.grid(False)
    
    ax.set_facecolor('#F0F0F0')
    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(1)
        spine.set_visible(True)
        
    plt.legend(title='type', labels=labels, bbox_to_anchor=(1.02, 0.5), loc='center left', 
               frameon=True, facecolor='white', edgecolor='white')
    
    plt.tight_layout()
    plt.savefig(f'Strategy_Distribution_{cond}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Strategy_Distribution_{cond}.png.")