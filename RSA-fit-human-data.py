import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# --- 1. VECTORIZED RSA LOGIC (3 Params) ---
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

def evaluate_rsa_weights(trials):
    bounds = [(0.1, 15), (0, 0.5), (0, 1)]
    best_nlls = []
    
    for level in [0, 1, 2]:
        best_nll = np.inf
        for _ in range(3): 
            x0 = [np.random.uniform(1.0, 8.0), np.random.uniform(0.01, 0.2), np.random.uniform(0.5, 0.99)]
            res = minimize(rsa_nll, x0, args=(trials, level), bounds=bounds, method='L-BFGS-B')
            if res.fun < best_nll:
                best_nll = res.fun
        best_nlls.append(best_nll)
        
    aics = np.array([6 + 2 * nll for nll in best_nlls])
    delta = aics - np.min(aics)
    weights = np.exp(-0.5 * delta) / np.sum(np.exp(-0.5 * delta))
    return weights

# --- 2. DATA EXTRACTION ---
results = []

# A. Original Data Extraction (294 Trials)
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

for pair in pairs:
    condition = pair[:-9]
    folder = pair[-8:]
    file_name = f"Data_OG/{condition}/{folder}/forelo_1.csv"
    
    if not os.path.exists(file_name):
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

    print(f"Fitting Original RSA for {pair}...")
    p1_w = evaluate_rsa_weights(p1_trials)
    p2_w = evaluate_rsa_weights(p2_trials)
    
    results.append({'Subject': f"{folder}_1", 'Condition': condition, 'Dataset': 'Original', 'L0': p1_w[0], 'L1': p1_w[1], 'L2': p1_w[2]})
    results.append({'Subject': f"{folder}_2", 'Condition': condition, 'Dataset': 'Original', 'L0': p2_w[0], 'L1': p2_w[1], 'L2': p2_w[2]})

# B. Online Data Extraction (588 Trials)
base_dir = "Data_Online"
online_conditions = ['I', 'NI', 'SS', 'KW', 'KH']

for cond in online_conditions:
    cond_path = os.path.join(base_dir, cond)
    if not os.path.exists(cond_path):
        continue
        
    pair_folders = [f for f in os.listdir(cond_path) if os.path.isdir(os.path.join(cond_path, f))]
    
    for pair_folder in pair_folders:
        file_name = os.path.join(cond_path, pair_folder, "forelo.csv")
        if not os.path.exists(file_name):
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

        print(f"Fitting Online RSA for {cond} / {pair_folder}...")
        p1_w = evaluate_rsa_weights(p1_trials)
        p2_w = evaluate_rsa_weights(p2_trials)
        
        results.append({'Subject': f"{pair_folder}_P1", 'Condition': cond, 'Dataset': 'Online', 'L0': p1_w[0], 'L1': p1_w[1], 'L2': p1_w[2]})
        results.append({'Subject': f"{pair_folder}_P2", 'Condition': cond, 'Dataset': 'Online', 'L0': p2_w[0], 'L1': p2_w[1], 'L2': p2_w[2]})

df_results = pd.DataFrame(results)

# --- 3. GENERATE PLOTS ---
plt.style.use('ggplot')
colors = ['#E0E0E0', '#888888', '#000000'] 
labels = ['Level 0 (Naive)', 'Level 1 (Pragmatic)', 'Level 2 (Meta)']

# Loop through all unique conditions identified across both datasets
for cond in df_results['Condition'].unique():
    subset = df_results[df_results['Condition'] == cond]
    if subset.empty:
        continue
    
    # Identify whether this condition is from Original or Online for titling
    dataset_source = subset['Dataset'].iloc[0]
    
    ax = subset.set_index('Subject')[['L0', 'L1', 'L2']].plot(
        kind='bar', stacked=True, color=colors, figsize=(14, 8), width=0.9
    )
    
    plt.title(f'RSA Strategy Probabilities: {cond} ({dataset_source})', fontsize=18, pad=20)
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
        
    plt.legend(title='Agent Type', labels=labels, bbox_to_anchor=(1.02, 0.5), loc='center left', 
               frameon=True, facecolor='white', edgecolor='white')
    
    plt.tight_layout()
    plt.savefig(f'RSA_Strategy_{dataset_source}_{cond}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated RSA_Strategy_{dataset_source}_{cond}.png.")