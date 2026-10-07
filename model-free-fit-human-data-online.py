import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 1. Model fitting functions (Decay-augmented Ind RL & Tied Joint RL)
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

def evaluate_participant_weights(trials):
    aic_base = 2 * fit_baseline(trials) 
    bounds = [(0, 1), (0.1, 15), (0, 1)]
    
    # Evaluate Individual RL (3 parameters, 3 starts)
    best_ind_nll = np.inf
    for _ in range(3):
        x0 = [np.random.uniform(0.1, 0.9), np.random.uniform(1.0, 10.0), np.random.uniform(0.5, 0.99)]
        res = minimize(ind_rl_nll, x0, args=(trials,), bounds=bounds, method='L-BFGS-B')
        if res.fun < best_ind_nll:
            best_ind_nll = res.fun
    aic_ind = (2 * 3) + 2 * best_ind_nll 
    
    # Evaluate Tied Joint RL (3 parameters, 3 starts)
    best_joint_nll = np.inf
    for _ in range(3):
        x0 = [np.random.uniform(0.1, 0.9), np.random.uniform(1.0, 10.0), np.random.uniform(0.5, 0.99)]
        res = minimize(joint_rl_tied_nll, x0, args=(trials,), bounds=bounds, method='L-BFGS-B')
        if res.fun < best_joint_nll:
            best_joint_nll = res.fun
    aic_joint = (2 * 3) + 2 * best_joint_nll 
    
    aics = np.array([aic_base, aic_ind, aic_joint])
    delta = aics - np.min(aics)
    weights = np.exp(-0.5 * delta) / np.sum(np.exp(-0.5 * delta))
    
    return weights

# 2. Dynamic Data Extraction
base_dir = "Data_Online"
conditions_list = ['I', 'NI', 'SS', 'KW', 'KH']
results = []

for cond in conditions_list:
    cond_path = os.path.join(base_dir, cond)
    
    if not os.path.exists(cond_path):
        print(f"Skipping: Condition folder '{cond}' not found at {cond_path}")
        continue
        
    # Get all pair folders inside the condition directory (e.g., '3hqko3xz_1')
    pair_folders = [f for f in os.listdir(cond_path) if os.path.isdir(os.path.join(cond_path, f))]
    
    for pair_folder in pair_folders:
        file_name = os.path.join(cond_path, pair_folder, "forelo.csv")
        
        if not os.path.exists(file_name):
            continue
            
        df = pd.read_csv(file_name)
        p1_trials, p2_trials = [], []
        
        # Loop through the 588 trials (assuming 'round_number' spans the full session)
        for r in df['round_number'].unique():
            subset = df[df['round_number'] == r]
            if len(subset) < 2: continue
            
            p1 = subset[subset['participant'] == 1].iloc[0]
            p2 = subset[subset['participant'] == 2].iloc[0]
            c1, c2 = int(p1['response']), int(p1['loser'])
            
            p1_trials.append({'options': (c1, c2), 'choice': int(p1['response']), 'partner_choice': int(p2['response'])})
            p2_trials.append({'options': (c1, c2), 'choice': int(p2['response']), 'partner_choice': int(p1['response'])})

        print(f"Fitting {cond} / {pair_folder} ({len(p1_trials)} trials)...")
        p1_w = evaluate_participant_weights(p1_trials)
        p2_w = evaluate_participant_weights(p2_trials)
        
        # Storing subject names as pairfolder_P1 and pairfolder_P2 to avoid confusion with existing suffixes
        results.append({'Subject': f"{pair_folder}_P1", 'Condition': cond, 'Baseline': p1_w[0], 'Ind RL': p1_w[1], 'Joint RL': p1_w[2]})
        results.append({'Subject': f"{pair_folder}_P2", 'Condition': cond, 'Baseline': p2_w[0], 'Ind RL': p2_w[1], 'Joint RL': p2_w[2]})

df_results = pd.DataFrame(results)

# 3. Generating the Plots
plt.style.use('ggplot')
colors = ['#E0E0E0', '#888888', '#000000'] 
labels = ['Baseline', 'Individual RL', 'Joint RL']

for cond in conditions_list:
    subset = df_results[df_results['Condition'] == cond]
    if subset.empty:
        continue
        
    ax = subset.set_index('Subject')[['Baseline', 'Ind RL', 'Joint RL']].plot(
        kind='bar', stacked=True, color=colors, figsize=(14, 8), width=0.9
    )
    
    plt.title(f'Strategy Probabilities (588 Trials): Condition {cond}', fontsize=18, pad=20)
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
    plt.savefig(f'Strategy_Distribution_Online_{cond}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Strategy_Distribution_Online_{cond}.png.")