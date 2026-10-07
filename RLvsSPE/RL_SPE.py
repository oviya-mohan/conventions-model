import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os

# 1. Core Model Functions
def softmax(q_values, beta):
    shifted_q = beta * (q_values - np.max(q_values))
    prob = np.exp(shifted_q)
    return prob / (np.sum(prob) + 1e-10)

def nll_rl(params, choices, partners, options):
    alpha, beta = params
    q = np.ones(8) * 0.5  # Assumes colors 1-7
    log_lik = 0
    for i in range(len(choices)):
        c, p = choices[i], partners[i]
        opt = options[i]
        q_options = np.array([q[opt[0]], q[opt[1]]])
        probs = softmax(q_options, beta)
        log_lik += np.log(probs[0] + 1e-10)
        reward = 1 if c == p else 0
        q[c] = q[c] + alpha * (reward - q[c])
    return -log_lik

def nll_social(params, choices, partners, options):
    alpha, beta = params
    p_model = np.ones(8) * 0.5
    log_lik = 0
    for i in range(len(choices)):
        c, p_choice = choices[i], partners[i]
        opt = options[i]
        p_options = np.array([p_model[opt[0]], p_model[opt[1]]])
        probs = softmax(p_options, beta)
        log_lik += np.log(probs[0] + 1e-10)
        p_model[p_choice] = p_model[p_choice] + alpha * (1 - p_model[p_choice])
    return -log_lik

# 2. Batch Processing Function
def analyze_pair(file_path, pair_id):
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    pivoted = df.pivot(index='round_number', columns='participant', values='response')
    loser_pivoted = df.pivot(index='round_number', columns='participant', values='loser')
    
    pair_results = []
    participants = pivoted.columns
    for focal_p in participants:
        partner_p = [p for p in participants if p != focal_p][0]
        choices = pivoted[focal_p].values
        partner_choices = pivoted[partner_p].values
        losers = loser_pivoted[focal_p].values
        options = [[c, l] for c, l in zip(choices, losers)]
        
        # Fit RL
        res_rl = minimize(nll_rl, [0.1, 1.0], args=(choices, partner_choices, options), bounds=[(0, 1), (0, 50)])
        aic_rl = 4 + 2 * res_rl.fun
        
        # Fit Social PE
        res_soc = minimize(nll_social, [0.1, 1.0], args=(choices, partner_choices, options), bounds=[(0, 1), (0, 50)])
        aic_soc = 4 + 2 * res_soc.fun
        
        pair_results.append({
            'Pair_ID': pair_id,
            'Condition': pair_id[:-9],
            'Participant': focal_p,
            'AIC_RL': round(aic_rl, 2),
            'AIC_Social': round(aic_soc, 2),
            'Best_Model': 'Social' if aic_soc < aic_rl else 'RL',
            'Alpha_Soc': round(res_soc.x[0], 4)
        })
    return pair_results

# 3. Main Loop
pairs = [
    'NI_OT_xq913l4v', 'I_TO_apqr0rpe', 'NI_TO_agx6z9dn', 'NI_TO_7wshhrbw', 'I_OT_c8j2ro94',
    'I_TO_4r55skxj', 'I_OT_h9uh3mul', 'I_OT_g4xpbpj6', 'NI_OT_sh8cf924', 'NI_TO_wiml9wm0',
    'NI_OT_ldhtljr9', 'NI_OT_eo2uzk2s', 'I_TO_h2uit3zp', 'I_TO_kuae9j3j', 'NI_TO_4bwl0qbj',
    'I_OT_3ix0zaem', 'NI_OT_rrlxmpp5', 'I_TO_t7x3vvi1', 'NI_TO_3ooaxiup', 'I_OT_tvge9zyu',
    'NI_OT_fpghgezy', 'NI_TO_d3nyyc5v', 'I_TO_bv42zb0r', 'NI_OT_ut8qatoj', 'I_OT_aw71h5pi',
    'NI_TO_98fqxtm8', 'I_TO_ohn5xo30', 'I_OT_v395919d', 'NI_TO_4fei2let', 'NI_OT_scc6ut4m',
    'I_TO_w08oe7pq', 'I_OT_fhfu2taw', 'NI_TO_5393z70p', 'NI_OT_f46z2j09', 'I_OT_14f0zewk',
    'NI_OT_3zl2di7o', 'NI_TO_qvuayn21', 'I_TO_29zq6soa', 'I_OT_axg2ydsa', 'I_TO_lx66ptw4'
]

final_results = []
for pair in pairs:
    condition = pair[:-9]
    folder = pair[-8:]
    path = f"Data/{condition}/{folder}/forelo_1.csv"
    
    result = analyze_pair(path, pair)
    if result:
        final_results.extend(result)
        print(f"Processed: {pair}")
    else:
        print(f"Skipped (File Not Found): {path}")

# 4. Save and Display
results_df = pd.DataFrame(final_results)
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n--- Summary Table ---")
print(results_df.to_string())