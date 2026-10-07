import numpy as np
import pandas as pd

def softmax(x, alpha=5.0):
    """Computes softmax probabilities with a sensitivity/optimality parameter."""
    e_x = np.exp(alpha * (x - np.max(x)))
    return e_x / e_x.sum()

class RSADyad_TraceMemory:
    def __init__(self, depth1=0, depth2=0, n_colors=7, alpha=5.0, decay=0.80, 
                 lapse=0.10, stickiness=5.0, trace_decay=0.60):
        self.n_colors = n_colors
        self.depths = [depth1, depth2]
        self.alpha = alpha
        
        # Long-Term Memory (RSA Scores)
        self.decay = decay                
        self.scores = [np.ones(n_colors) * 1000 for _ in range(2)]
        
        # Short-Term Priming Traces
        self.lapse = lapse                
        self.stickiness = stickiness      
        self.trace_decay = trace_decay    
        self.traces = [np.zeros(n_colors) for _ in range(2)]
        
        # --- NEW: Independent Elo Ratings for Emergence Metric ---
        # Initialized to 1000, updated on every choice via standard Elo math
        self.elo_ratings = [np.ones(n_colors) * 1000 for _ in range(2)]

    def get_probs_iterative(self, agent_idx, indices):
        # 1. Start with Literal Level (Level 0) beliefs based on scores + traces
        l0_probs = []
        for i in range(2):
            s = self.scores[i][indices].copy()
            s += self.traces[i][indices] * self.stickiness
            l0_probs.append(softmax(s, self.alpha))
            
        # 2. Iterate levels of social reasoning
        probs_at_level = {0: l0_probs}
        max_d = max(self.depths)
        
        for d in range(1, max_d + 1):
            p1_at_d = softmax(np.log(probs_at_level[d-1][1] + 1e-9), self.alpha)
            p2_at_d = softmax(np.log(probs_at_level[d-1][0] + 1e-9), self.alpha)
            probs_at_level[d] = [p1_at_d, p2_at_d]
            
        return probs_at_level[self.depths[agent_idx]][agent_idx]

    def update_elo(self, agent_idx, winner, loser, k=100):
        """Standard Elo rating update matching the R package."""
        r_win = self.elo_ratings[agent_idx][winner]
        r_lose = self.elo_ratings[agent_idx][loser]
        
        # Expected probability of winning
        expected_win = 1 / (1 + 10 ** ((r_lose - r_win) / 400))
        
        # Update ratings
        self.elo_ratings[agent_idx][winner] += k * (1 - expected_win)
        self.elo_ratings[agent_idx][loser] += k * (0 - (1 - expected_win))

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
        
        # --- NEW: Combined Accuracy Metric ---
        # 1 if both agents chose the color with strictly higher current Elo rating, AND matched
        comb_acc = 1
        for i in range(2):
            winner = choices[i]
            loser = c2 if winner == c1 else c1
            if self.elo_ratings[i][winner] <= self.elo_ratings[i][loser]:
                comb_acc = 0
                
        if not success:
            comb_acc = 0
            
        # Update internal state (Scores, Traces, and Elo)
        for i in range(2):
            winner = choices[i]
            loser = c2 if winner == c1 else c1
            
            # 1. Update Elo tracker based on revealed preference
            self.update_elo(i, winner, loser)
            
            # 2. Update RSA Scores
            penalty = 30 if self.depths[i] >= 2 else 10
            self.scores[i][choices[i]] += 10 if success else -penalty
            self.scores[i] *= self.decay
            
            # 3. Update Short-Term Priming Traces
            self.traces[i] *= self.trace_decay
            self.traces[i][choices[i]] = 1.0
            
        return comb_acc

def run_experiment(depth1, depth2, n_sims=50, **params):
    emergence_times = []
    for _ in range(n_sims):
        dyad = RSADyad_TraceMemory(depth1=depth1, depth2=depth2, **params)
        history = [] # Now stores combined accuracy instead of raw success
        pairings = [(i, j) for i in range(7) for j in range(i+1, 7)]
        
        for t in range(294):
            if t % 21 == 0: np.random.shuffle(pairings)
            history.append(dyad.run_trial(*pairings[t % 21]))
            
        # --- NEW: Emergence calculation matching emergence.py exactly ---
        window_size = 21
        combined_accuracy_per_window = []
        
        # Calculate rolling mean for 21-trial windows
        for i in range(window_size, len(history) + 1):
            combined_accuracy_per_window.append(np.mean(history[i-window_size:i]))
            
        emergence = 294
        count = 0
        rounds_to_convention = 21
        
        # Look for 21 consecutive windows above 0.80 threshold
        for i in range(len(combined_accuracy_per_window)):
            if combined_accuracy_per_window[i] >= 0.80:
                count += 1
            else:
                count = 0
                
            if count == rounds_to_convention:
                emergence = i + 1 # Matches the 'stable_at = i + 1' logic
                break
                
        emergence_times.append(emergence)

    successful_pairs = sum(1 for e in emergence_times if e < 294)
    return np.mean(emergence_times), np.std(emergence_times), successful_pairs

# --- EXECUTION ---
print("Running Simulation with Elo-based Combined Accuracy...")
params = {'alpha': 1.0, 'lapse': 0.10, 'stickiness': 2.0, 'trace_decay': 0.60, 'decay': 0.70}

conditions = [
    (0, 0, "Level 0 vs Level 0"),
    (1, 1, "Level 1 vs Level 1"),
    (2, 2, "Level 2 vs Level 2"),
    (0, 2, "Level 0 vs Level 2")
]

results = []
n_sims = 100
for d1, d2, label in conditions:
    mean_val, sd_val, success_count = run_experiment(d1, d2, n_sims=n_sims, **params)
    results.append({
        "Condition": label, 
        "Convention Pairs": f"{success_count}/{n_sims}",
        "Mean Emergence": f"{mean_val:.2f}", 
        "SD": f"{sd_val:.2f}" 
    })

df_results = pd.DataFrame(results)
print("\n--- RESULTS: N = " + str(n_sims) + " Pairs---")
print(params)
print(df_results.to_string(index=False))