import numpy as np
import pandas as pd

def softmax(x, alpha=5.0):
    e_x = np.exp(alpha * (x - np.max(x)))
    return e_x / e_x.sum()

class RSADyad:
    def __init__(self, depth1=0, depth2=0, n_colors=7, alpha=5.0, decay=0.90, lapse=0.10, stickiness=2.0):
        self.n_colors = n_colors
        self.depths = [depth1, depth2]
        self.alpha = alpha
        self.decay = decay      
        self.lapse = lapse      
        self.stickiness = stickiness 
        self.scores = [np.ones(n_colors) * 1000 for _ in range(2)]
        self.last_choices = [None, None]

    def compute_probs_iterative(self, agent_idx, indices):
        """Calculates probabilities iteratively to avoid RecursionError."""
        # 1. Start with Literal Listener (Level 0) logic for both agents
        # This is the 'ground truth' of their current Elo-like scores
        current_probs = []
        for i in range(2):
            s = self.scores[i][indices].copy()
            if self.last_choices[i] in indices:
                idx = 0 if indices[0] == self.last_choices[i] else 1
                s[idx] += self.stickiness
            current_probs.append(softmax(s, self.alpha))
            
        # 2. Iterate up to the required depth
        max_depth = max(self.depths)
        for d in range(1, max_depth + 1):
            new_probs = [None, None]
            # Agent 0 reasons about Agent 1's previous level
            new_probs[0] = softmax(np.log(current_probs[1] + 1e-9), self.alpha)
            # Agent 1 reasons about Agent 0's previous level
            new_probs[1] = softmax(np.log(current_probs[0] + 1e-9), self.alpha)
            current_probs = new_probs
            
        # 3. Return the specific probability for the requested agent
        return current_probs[agent_idx] if self.depths[agent_idx] > 0 else softmax(self.scores[agent_idx][indices], self.alpha)

    def run_trial(self, c1, c2):
        # Using the iterative method instead of recursive
        p1 = self.compute_probs_iterative(0, [c1, c2])
        p2 = self.compute_probs_iterative(1, [c1, c2])
        
        choices = []
        for probs in [p1, p2]:
            if np.random.rand() < self.lapse:
                choices.append(np.random.choice([c1, c2]))
            else:
                choices.append(np.random.choice([c1, c2], p=probs))
        
        success = (choices[0] == choices[1])
        for i in range(2):
            penalty = 30 if self.depths[i] >= 2 else 10
            self.scores[i][choices[i]] += 10 if success else -penalty
            self.scores[i] *= self.decay
            self.last_choices[i] = choices[i]
            
        return success

# The rest of the simulate_condition function remains the same as before

def simulate_condition(d1, d2, n_sims=100):
    emergence_times = []
    for _ in range(n_sims):
        dyad = RSADyad(depth1=d1, depth2=d2)
        history = []
        pairings = [(i, j) for i in range(7) for j in range(i+1, 7)]
        emergence = 294
        
        for t in range(294):
            if t % 21 == 0: np.random.shuffle(pairings)
            history.append(dyad.run_trial(*pairings[t % 21]))
            
            # Criterion: 80% accuracy for 21 consecutive windows [cite: 158-161]
            if t >= 42 and emergence == 294:
                if all(np.mean(history[i:i+21]) >= 0.8 for i in range(t-42+1, t-21+1)):
                    emergence = t - 21
        emergence_times.append(emergence)
    return emergence_times

# --- EXECUTION ---
print("Running simulations...")
results_ni = simulate_condition(0, 0) # Not Informed
results_asym = simulate_condition(0, 2) # Asymmetric
results_i = simulate_condition(2, 2) # Informed

print("\n--- RESULTS: MEAN TRIAL OF EMERGENCE ---")
print(f"NI (0 vs 0): {np.mean(results_ni):.2f} trials")
print(f"Asymmetric (0 vs 2): {np.mean(results_asym):.2f} trials")
print(f"Informed (2 vs 2): {np.mean(results_i):.2f} trials")