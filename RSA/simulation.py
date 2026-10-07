import numpy as np
import pandas as pd

def softmax(x, alpha=5.0):
    """Computes softmax probabilities with an optimality parameter."""
    e_x = np.exp(alpha * (x - np.max(x)))
    return e_x / e_x.sum()

class RSAPair:
    def __init__(self, n_colors=7, depth=0, alpha=5.0):
        self.n_colors = n_colors
        self.depth = depth
        self.alpha = alpha
        # Internal belief (lexicon) for each agent: Elo-like scores for colors
        self.agent1_belief = np.zeros(n_colors)
        self.agent2_belief = np.zeros(n_colors)
        
    def get_probs(self, belief, depth):
        """Recursive RSA probability calculation."""
        if depth == 0:
            # Literal: choose based on current internal scores
            return softmax(belief, self.alpha)
        else:
            # Pragmatic: Reason about the partner's likely choice
            partner_probs = self.get_probs(belief, depth - 1)
            # Informativity: Higher utility if partner is also likely to pick it
            utilities = np.log(partner_probs + 1e-9)
            return softmax(utilities, self.alpha)

    def run_trial(self, color_a, color_b):
        """Simulates a single trial with two available colors."""
        p1 = self.get_probs(self.agent1_belief[[color_a, color_b]], self.depth)
        p2 = self.get_probs(self.agent2_belief[[color_a, color_b]], self.depth)
        
        # Sample choices
        choice1 = np.random.choice([color_a, color_b], p=p1)
        choice2 = np.random.choice([color_a, color_b], p=p2)
        
        success = 1 if choice1 == choice2 else 0
        
        # Update beliefs (Reinforcement): Increase score if successful
        if success:
            self.agent1_belief[choice1] += 1
            self.agent2_belief[choice2] += 1
        else:
            # Penalty for mismatch
            self.agent1_belief[choice1] -= 0.5
            self.agent2_belief[choice2] -= 0.5
            
        return success

def simulate_session(depth, n_trials=294, threshold=0.8, window=21):
    pair = RSAPair(depth=depth)
    history = []
    
    # Generate trial color pairs (randomized 21-trial blocks)
    colors = list(range(7))
    all_pairings = [(i, j) for i in range(7) for j in range(i+1, 7)] # 21 pairings
    
    for t in range(n_trials):
        if t % 21 == 0: np.random.shuffle(all_pairings)
        c1, c2 = all_pairings[t % 21]
        history.append(pair.run_trial(c1, c2))
        
        # Check for convention emergence (Criterion: >80% for 21 windows)
        if t >= (window + window): # 42 trials as per paper criterion
            rolling_acc = [np.mean(history[i:i+window]) for i in range(t-window-window+1, t-window+1)]
            if all(acc >= threshold for acc in rolling_acc):
                return t - window # Return trial of emergence
    return n_trials

# Run Experiment: Compare Depth 0 (NI) vs Depth 2 (I)
n_sims = 1000
ni_results = [simulate_session(depth=0) for _ in range(n_sims)]
i_results = [simulate_session(depth=2) for _ in range(n_sims)]

print(f"Mean Trial of Emergence - NI (Depth 0): {np.mean(ni_results):.1f}")
print(f"Mean Trial of Emergence - Informed (Depth 2): {np.mean(i_results):.1f}")