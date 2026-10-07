import numpy as np
import pandas as pd

def softmax(x, alpha=5.0):
    e_x = np.exp(alpha * (x - np.max(x)))
    return e_x / e_x.sum()

class BayesianRSADyad:
    def __init__(self, d1, d2, n_colors=7, alpha=5.0, lapse=0.10, doubt_0=0.2, doubt_2=2.0):
        self.n_colors = n_colors
        self.depths = [d1, d2]
        self.alpha = alpha
        self.lapse = lapse
        self.doubt_factors = [doubt_2 if d >= 2 else doubt_0 for d in self.depths]
        
        # Bayesian counts (Beliefs)
        self.beliefs = [np.ones(n_colors) for _ in range(2)]
        self.belief_history = []

    def compute_probs(self, agent_idx, indices):
        # Current normalized beliefs (Level 0)
        curr_probs = [self.beliefs[i][indices] / self.beliefs[i][indices].sum() for i in range(2)]
        
        # Iterative RSA reasoning
        for d in range(1, max(self.depths) + 1):
            next_p = [softmax(np.log(curr_probs[1] + 1e-9), self.alpha),
                      softmax(np.log(curr_probs[0] + 1e-9), self.alpha)]
            curr_probs = next_p
            
        return curr_probs[agent_idx] if self.depths[agent_idx] > 0 else (self.beliefs[agent_idx][indices] / self.beliefs[agent_idx][indices].sum())

    def run_trial(self, c1, c2):
        p_choices = [self.compute_probs(i, [c1, c2]) for i in range(2)]
        choices = [np.random.choice([c1, c2]) if np.random.rand() < self.lapse 
                   else np.random.choice([c1, c2], p=p_choices[i]) for i in range(2)]
        
        success = (choices[0] == choices[1])
        for i in range(2):
            if success:
                self.beliefs[i][choices[i]] += 1.0
            else:
                # Apply Strategic Doubt: subtraction of weight [Equation 1 context]
                self.beliefs[i][choices[i]] = max(1.0, self.beliefs[i][choices[i]] - self.doubt_factors[i])
        
        self.belief_history.append(self.beliefs[0].copy())
        return success

def calculate_stability_s(belief_history, emergence_trial):
    """Implementation of Equation 1 from Mohan & Biro paper [cite: 178]"""
    if emergence_trial > len(belief_history) - 100:
        return None # Not enough trials left for 100-trial window 
    
    window = belief_history[emergence_trial : emergence_trial + 100]
    # Convert belief counts to ranks (0-6)
    ranks = [np.argsort(np.argsort(b)) for b in window]
    
    crossovers = 0
    for t in range(1, len(ranks)):
        # Count differences in rank positions between trial i and i-1 [cite: 178]
        crossovers += np.sum(ranks[t] != ranks[t-1])
        
    # Equation 1: S = 1 - (crossovers / (window_size * n_colors))
    # Note: Using your paper's logic where 1 is perfect stability [cite: 175]
    s_val = 1 - (crossovers / (100 * 7))
    return s_val

def simulate_full_study(n_sims=30):
    conditions = [(0, 0, "NI"), (0, 2, "Asymmetric"), (2, 2, "Informed")]
    summary = []

    for d1, d2, label in conditions:
        emergence_times, stability_scores = [], []
        
        for _ in range(n_sims):
            dyad = BayesianRSADyad(d1, d2)
            history = []
            pairings = [(i, j) for i in range(7) for j in range(i+1, 7)]
            trial_of_emergence = 294
            
            for t in range(294):
                if t % 21 == 0: np.random.shuffle(pairings)
                history.append(dyad.run_trial(*pairings[t % 21]))
                
                if t >= 42 and trial_of_emergence == 294:
                    if all(np.mean(history[i:i+21]) >= 0.8 for i in range(t-42+1, t-21+1)):
                        trial_of_emergence = t - 21
            
            s = calculate_stability_s(dyad.belief_history, trial_of_emergence)
            emergence_times.append(trial_of_emergence)
            if s is not None: stability_scores.append(s)
            
        summary.append({
            "Condition": label,
            "Mean Emergence": np.mean(emergence_times),
            "Mean Stability (S)": np.mean(stability_scores) if stability_scores else 0
        })
    
    return pd.DataFrame(summary)

# Run and Display
print(simulate_full_study())