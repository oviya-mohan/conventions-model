import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# --- Model Definitions ---
def softmax(q_values, beta):
    shifted_q = beta * (q_values - np.max(q_values))
    prob = np.exp(shifted_q)
    return prob / (np.sum(prob) + 1e-10)

def get_trial_log_liks_rl(params, choices, partners, options):
    alpha, beta = params
    q = np.ones(8) * 0.5
    log_liks = []
    for i in range(len(choices)):
        c, p = choices[i], partners[i]
        opt = options[i]
        q_options = np.array([q[opt[0]], q[opt[1]]])
        probs = softmax(q_options, beta)
        log_liks.append(np.log(probs[0] + 1e-10))
        reward = 1 if c == p else 0
        q[c] = q[c] + alpha * (reward - q[c])
    return np.array(log_liks)

def get_trial_log_liks_social(params, choices, partners, options):
    alpha, beta = params
    p_model = np.ones(8) * 0.5
    log_liks = []
    for i in range(len(choices)):
        c, p_choice = choices[i], partners[i]
        opt = options[i]
        p_options = np.array([p_model[opt[0]], p_model[opt[1]]])
        probs = softmax(p_options, beta)
        log_liks.append(np.log(probs[0] + 1e-10))
        p_model[p_choice] = p_model[p_choice] + alpha * (1 - p_model[p_choice])
    return np.array(log_liks)

def nll_rl(params, choices, partners, options):
    return -np.sum(get_trial_log_liks_rl(params, choices, partners, options))

def nll_social(params, choices, partners, options):
    return -np.sum(get_trial_log_liks_social(params, choices, partners, options))

# --- Plotting Function ---
def plot_rolling_aic(pair_id, participant_data, window=25):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.4)
    
    for i, (p_id, data) in enumerate(participant_data.items()):
        ax = axes[i]
        ll_rl_series = pd.Series(data['ll_rl'])
        ll_soc_series = pd.Series(data['ll_soc'])
        
        # Calculate Rolling AIC: 2*k - 2*Sum(LL_window)
        rolling_ll_rl = ll_rl_series.rolling(window=window, center=True).sum()
        rolling_ll_soc = ll_soc_series.rolling(window=window, center=True).sum()
        
        k = 2 # alpha and beta
        rolling_aic_rl = 2*k - 2*rolling_ll_rl
        rolling_aic_soc = 2*k - 2*rolling_ll_soc
        
        ax.plot(rolling_aic_rl, label='RL Model', color='#d62728', lw=2)
        ax.plot(rolling_aic_soc, label='Social PE Model', color='#1f77b4', lw=2)
        
        ax.set_title(f"Participant {p_id} - {pair_id}")
        ax.set_ylabel(f"Rolling AIC (W={window})")
        if i == 1: ax.set_xlabel("Trial Number")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.savefig(f"{pair_id}_AIC.jpg")
    plt.close()

# --- Batch Processing ---
pairs = ['NI_OT_xq913l4v', 'I_TO_apqr0rpe', 'NI_TO_agx6z9dn', 'NI_TO_7wshhrbw', 'I_OT_c8j2ro94',
         'I_TO_4r55skxj', 'I_OT_h9uh3mul', 'I_OT_g4xpbpj6', 'NI_OT_sh8cf924', 'NI_TO_wiml9wm0',
         'NI_OT_ldhtljr9', 'NI_OT_eo2uzk2s', 'I_TO_h2uit3zp', 'I_TO_kuae9j3j', 'NI_TO_4bwl0qbj',
         'I_OT_3ix0zaem', 'NI_OT_rrlxmpp5', 'I_TO_t7x3vvi1', 'NI_TO_3ooaxiup', 'I_OT_tvge9zyu',
         'NI_OT_fpghgezy', 'NI_TO_d3nyyc5v', 'I_TO_bv42zb0r', 'NI_OT_ut8qatoj', 'I_OT_aw71h5pi',
         'NI_TO_98fqxtm8', 'I_TO_ohn5xo30', 'I_OT_v395919d', 'NI_TO_4fei2let', 'NI_OT_scc6ut4m',
         'I_TO_w08oe7pq', 'I_OT_fhfu2taw', 'NI_TO_5393z70p', 'NI_OT_f46z2j09', 'I_OT_14f0zewk',
         'NI_OT_3zl2di7o', 'NI_TO_qvuayn21', 'I_TO_29zq6soa', 'I_OT_axg2ydsa', 'I_TO_lx66ptw4']

for pair in pairs:
    condition = pair[:-9]
    folder = pair[-8:]
    file_path = f"Data/{condition}/{folder}/forelo_1.csv"
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        pivoted = df.pivot(index='round_number', columns='participant', values='response')
        loser_pivoted = df.pivot(index='round_number', columns='participant', values='loser')
        
        participant_data = {}
        for focal_p in pivoted.columns:
            partner_p = [p for p in pivoted.columns if p != focal_p][0]
            choices = pivoted[focal_p].values
            partner_choices = pivoted[partner_p].values
            losers = loser_pivoted[focal_p].values
            options = [[c, l] for c, l in zip(choices, losers)]
            
            # Global Fit
            res_rl = minimize(nll_rl, [0.1, 1.0], args=(choices, partner_choices, options), bounds=[(0,1), (0,50)])
            res_soc = minimize(nll_social, [0.1, 1.0], args=(choices, partner_choices, options), bounds=[(0,1), (0,50)])
            
            participant_data[focal_p] = {
                'll_rl': get_trial_log_liks_rl(res_rl.x, choices, partner_choices, options),
                'll_soc': get_trial_log_liks_social(res_soc.x, choices, partner_choices, options)
            }
        plot_rolling_aic(pair, participant_data)
        print(f"Processed {pair}")