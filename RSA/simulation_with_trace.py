import numpy as np
import pandas as pd

def softmax(x, alpha=5.0):
    e_x = np.exp(alpha * (x - np.max(x)))
    return e_x / e_x.sum()

class RSADyad_TraceMemory:
    def __init__(self, depth1=0, depth2=0, n_colors=7, alpha=5.0, decay=0.80, 
                 lapse=0.10, stickiness=5.0, trace_decay=0.60):
        self.n_colors = n_colors
        self.depths = [depth1, depth2]
        self.alpha = alpha
        
        # Long-Term Memory 
        self.decay = decay                # Rate at which internal scores fade over time
        self.scores = [np.ones(n_colors) * 1000 for _ in range(2)]
        
        self.lapse = lapse                # random distraction
        self.stickiness = stickiness      # habit
        self.trace_decay = trace_decay    # short-term memory 
        self.traces = [np.zeros(n_colors) for _ in range(2)]

    def get_probs_iterative(self, agent_idx, indices):
        # Level 0
        l0_probs = []
        for i in range(2):
            s = self.scores[i][indices].copy()
            
            # short term "bonus"
            s += self.traces[i][indices] * self.stickiness
            
            l0_probs.append(softmax(s, self.alpha))
            
        # Level n > 0
        probs_at_level = {0: l0_probs}
        max_d = max(self.depths)
        
        for d in range(1, max_d + 1):
            # Pragmatic agent reasons about what the partner's choice implies at depth d-1
            p1_at_d = softmax(np.log(probs_at_level[d-1][1] + 1e-9), self.alpha)
            p2_at_d = softmax(np.log(probs_at_level[d-1][0] + 1e-9), self.alpha)
            probs_at_level[d] = [p1_at_d, p2_at_d]
            
        return probs_at_level[self.depths[agent_idx]][agent_idx]

    def run_trial(self, c1, c2):
        # Calculate Choice Probabilities based on RSA Depth
        p1 = self.get_probs_iterative(0, [c1, c2])
        p2 = self.get_probs_iterative(1, [c1, c2])
        
        # Decision with attention lapse
        choices = []
        for p in [p1, p2]:
            if np.random.rand() < self.lapse:
                choices.append(np.random.choice([c1, c2]))
            else:
                choices.append(np.random.choice([c1, c2], p=p))
        
        success = (choices[0] == choices[1])
        
        # Update Scores and Priming Traces
        for i in range(2):
            # 1. Update Long-Term Scores
            # penalty = 10 + (self.depths[i] * 10)
            penalty = 30 if self.depths[i] >= 2 else 10
            self.scores[i][choices[i]] += 10 if success else -penalty
            self.scores[i] *= self.decay
            
            # 2. Update Traces
            # First, fade all traces for this agent
            self.traces[i] *= self.trace_decay
            # Second, reset the trace for the color just chosen to 1.0 (Maximum warmth)
            self.traces[i][choices[i]] = 1.0
            
        return success

def run_experiment(depth1, depth2, n_sims=50, **params):
    emergence_times = []
    for _ in range(n_sims):
        dyad = RSADyad_TraceMemory(depth1=depth1, depth2=depth2, **params)
        history = []
        emergence = 294
        pairings = [(i, j) for i in range(7) for j in range(i+1, 7)]
        
        for t in range(294):
            if t % 21 == 0: np.random.shuffle(pairings)
            history.append(dyad.run_trial(*pairings[t % 21]))
            
            # Criterion: 80% accuracy for 21 consecutive windows
            if t >= 42 and emergence == 294:
                if all(np.mean(history[i:i+21]) >= 0.8 for i in range(t-42+1, t-21+1)):
                    emergence = t - 21
        emergence_times.append(emergence)

    successful_pairs = sum(1 for e in emergence_times if e < 294)
    return np.mean(emergence_times), np.std(emergence_times), successful_pairs

# --- EXECUTION ---
print("Running Simulation...")
params = {'alpha': 3.0, 'lapse': 0.10, 'stickiness': 2.0, 'trace_decay': 0.60, 'decay': 0.90}

conditions = [
    (0, 0, "Level 0 vs Level 0"),
    (1, 1, "Level 1 vs Level 1"),
    (2, 2, "Level 2 vs Level 2"),
    (0, 2, "Level 0 vs Level 2")
]

results = []
n_sims = 100
for d1, d2, label in conditions:
    mean_val, sd_val, success_count = run_experiment(d1, d2, n_sims, **params)
    results.append({"Condition": label, "Convention Pairs": f"{success_count}/{n_sims}","Mean Emergence": f"{mean_val:.2f}", "SD": f"{sd_val:.2f}" })

df_results = pd.DataFrame(results)
print("\n--- RESULTS: N = " + str(n_sims) + " Pairs---")
print(params)
print(df_results.to_string(index=False))