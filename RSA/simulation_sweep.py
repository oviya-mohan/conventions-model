import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def softmax(x, alpha=5.0):
    e_x = np.exp(alpha * (x - np.max(x)))
    return e_x / e_x.sum()

class RSADyad:
    def __init__(self, n_colors=7, depth1=0, depth2=0, alpha=5.0, decay=0.80, lapse=0.10, stickiness=2.0):
        self.n_colors = n_colors
        self.depths = [depth1, depth2]
        self.alpha = alpha
        self.decay = decay
        self.lapse = lapse
        self.stickiness = stickiness
        self.scores = [np.ones(n_colors) * 1000, np.ones(n_colors) * 1000]
        self.last_choices = [None, None]

    def get_probs_iterative(self, agent_idx, indices):
        l0_probs = []
        for i in range(2):
            local_scores = self.scores[i][indices].copy()
            if self.last_choices[i] in indices:
                local_scores[0 if indices[0] == self.last_choices[i] else 1] += self.stickiness
            l0_probs.append(softmax(local_scores, self.alpha))
            
        probs_at_level = {0: l0_probs}
        max_d = max(self.depths)
        for d in range(1, max_d + 1):
            p1_at_d = softmax(np.log(probs_at_level[d-1][1] + 1e-9), self.alpha)
            p2_at_d = softmax(np.log(probs_at_level[d-1][0] + 1e-9), self.alpha)
            probs_at_level[d] = [p1_at_d, p2_at_d]
            
        return probs_at_level[self.depths[agent_idx]][agent_idx]

    def run_trial(self, c1, c2):
        p1 = self.get_probs_iterative(0, [c1, c2])
        p2 = self.get_probs_iterative(1, [c1, c2])
        
        choices = []
        for p in [p1, p2]:
            if np.random.rand() < self.lapse:
                choices.append(np.random.choice([c1, c2]))
            else:
                choices.append(np.random.choice([c1, c2], p=p))
        
        success = (choices[0] == choices[1])
        
        for i in range(2):
            penalty = 30 if self.depths[i] >= 2 else 10
            self.scores[i][choices[i]] += 10 if success else -penalty
            self.scores[i] *= self.decay
            self.last_choices[i] = choices[i]
            
        return success

def run_experiment_with_sd(depth1, depth2, n_sims=30, **params):
    emergence_times = []
    for _ in range(n_sims):
        dyad = RSADyad(depth1=depth1, depth2=depth2, **params)
        history = []
        emergence = 294
        pairings = [(i, j) for i in range(7) for j in range(i+1, 7)]
        for t in range(294):
            if t % 21 == 0: np.random.shuffle(pairings)
            history.append(dyad.run_trial(*pairings[t % 21]))
            if t >= 42 and emergence == 294:
                if all(np.mean(history[i:i+21]) >= 0.8 for i in range(t-42+1, t-21+1)):
                    emergence = t - 21
        emergence_times.append(emergence)
    return np.mean(emergence_times), np.std(emergence_times)

sweeps = {
    'alpha': [2, 5, 10, 20],
    'lapse': [0.0, 0.05, 0.1, 0.2],
    'stickiness': [0.0, 1.0, 2.0, 5.0],
    'decay': [0.7, 0.8, 0.9, 0.98]
}

default_params = {'alpha': 5.0, 'lapse': 0.10, 'stickiness': 2.0, 'decay': 0.80}
conditions = [(0, 0, "NI"), (0, 2, "Asymmetric"), (2, 2, "Informed")]
all_data = []

for param_name, param_values in sweeps.items():
    for val in param_values:
        current_params = default_params.copy()
        current_params[param_name] = val
        for d1, d2, label in conditions:
            mean_e, sd_e = run_experiment_with_sd(d1, d2, n_sims=30, **current_params)
            all_data.append({
                'parameter': param_name,
                'value': val,
                'condition': label,
                'mean': mean_e,
                'sd': sd_e
            })

df_results = pd.DataFrame(all_data)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, param_name in enumerate(sweeps.keys()):
    ax = axes[i]
    param_df = df_results[df_results['parameter'] == param_name]
    for label in ["NI", "Asymmetric", "Informed"]:
        cond_df = param_df[param_df['condition'] == label]
        ax.errorbar(cond_df['value'], cond_df['mean'], yerr=cond_df['sd'], 
                    marker='o', capsize=5, label=label)
    ax.set_title(f"Effect of {param_name.capitalize()}")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Mean Emergence (Trial #)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('parameter_sweeps_with_sd.png')
print("Plots with error bars saved as parameter_sweeps_with_sd.png")