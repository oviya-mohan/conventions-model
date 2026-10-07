import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def softmax(x, alpha=5.0):
    e_x = np.exp(alpha * (x - np.max(x)))
    return e_x / e_x.sum()

class RSAPair:
    def __init__(self, n_colors=7, depth=0, alpha=2.0, decay=0.98, lapse=0.10, stickiness=2.0):
        self.n_colors = n_colors
        self.depth = depth
        self.alpha = alpha
        self.decay = decay      # Memory Fuzziness: Scores fade by 2% each trial
        self.lapse = lapse      # Trembling Hand: 10% chance of a random choice
        self.stickiness = stickiness # Inertia: Bonus for repeating the last choice
        
        self.agent1_scores = np.ones(n_colors) * 1000
        self.agent2_scores = np.ones(n_colors) * 1000
        self.last_choice1, self.last_choice2 = None, None
        self.score_history = []

    def get_probs(self, scores, indices, depth, last_choice):
        local_scores = scores[indices].copy()
        if last_choice in indices:
            local_scores[0 if indices[0] == last_choice else 1] += self.stickiness
        if depth == 0:
            return softmax(local_scores, self.alpha)
        else:
            # Recursive Social Reasoning [cite: 2192, 2211]
            partner_probs = self.get_probs(scores, indices, depth - 1, last_choice)
            return softmax(np.log(partner_probs + 1e-9), self.alpha)

    def run_trial(self, c1, c2):
        # 1. Calculate RSA Probs
        p1 = self.get_probs(self.agent1_scores, [c1, c2], self.depth, self.last_choice1)
        p2 = self.get_probs(self.agent2_scores, [c1, c2], self.depth, self.last_choice2)
        
        # 2. Choice with Lapse (Attention error)
        c1_out = np.random.choice([c1, c2]) if np.random.rand() < self.lapse else np.random.choice([c1, c2], p=p1)
        c2_out = np.random.choice([c1, c2]) if np.random.rand() < self.lapse else np.random.choice([c1, c2], p=p2)
        success = (c1_out == c2_out)
        
        # 3. Update Scores (Elo-like) [cite: 1707]
        # Strategic Over-correction: Mismatch hurts more for informed reasoners
        if self.depth == 0:
            penalty = 10
        elif self.depth == 1:
            penalty = 20
        else: 
            penalty = 30 
        for scores, choice in [(self.agent1_scores, c1_out), (self.agent2_scores, c2_out)]:
            scores[choice] += 10 if success else -penalty
                
        # 4. Apply Memory Decay
        self.agent1_scores *= self.decay
        self.agent2_scores *= self.decay
        self.last_choice1, self.last_choice2 = c1_out, c2_out
        self.score_history.append(self.agent1_scores.copy())
        return success

def simulate_and_compare(n_sims=100):
    def run_sim(depth):
        pair = RSAPair(depth=depth)
        history, emergence = [], 294
        pairings = [(i, j) for i in range(7) for j in range(i+1, 7)]
        for t in range(294):
            if t % 21 == 0: np.random.shuffle(pairings)
            history.append(pair.run_trial(*pairings[t % 21]))
            if t >= 42 and emergence == 294:
                if all(np.mean(history[i:i+21]) >= 0.8 for i in range(t-42+1, t-21+1)):
                    emergence = t - 21
        return emergence

    L0_results = [run_sim(0) for _ in range(n_sims)]
    L1_results = [run_sim(1) for _ in range(n_sims)]
    L2_results = [run_sim(2) for _ in range(n_sims)]
    print(f"Level 0 Mean Emergence: {np.mean(L0_results):.2f} trials")
    print(f"Level 1 Mean Emergence: {np.mean(L1_results):.2f} trials")
    print(f"Level 2 Mean Emergence: {np.mean(L2_results):.2f} trials")

simulate_and_compare()