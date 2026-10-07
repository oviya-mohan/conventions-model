import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import mode

if __name__ == "__main__":
    # Load the saved InferenceData object
    try:
        idata = az.from_netcdf('coordination_model_idata.nc')
        print("Model output loaded successfully.")
    except FileNotFoundError:
        print("Error: 'coordination_model_idata.nc' not found. Please run 'run_model.py' first.")
        exit()

    # Visualize posteriors of key parameters
    print("Generating trace plots...")
    az.plot_trace(idata, var_names=['p_levels', 'color_hierarchy', 'learning_rate', 'convention_strength'])
    plt.show()

    # Print the most likely level for each player
    print("\n--- Player Level Classification ---")
    posterior_levels = idata.posterior["player_levels"].values
    
    # Assuming two players
    n_players = posterior_levels.shape[0] if posterior_levels.ndim == 1 else posterior_levels.shape[2]
    all_samples = posterior_levels.reshape(n_players, -1)

    level_names = {0: "Level-0 (Fixed Hierarchy)",
                   1: "Level-1 (Best-responding to L0)",
                   2: "Level-2 (Creating a convention)"}

    for i in range(n_players):
        most_likely_level = int(mode(all_samples[i, :])[0])
        count = np.sum(all_samples[i, :] == most_likely_level)
        probability = count / len(all_samples[i, :])
        level_name = level_names[most_likely_level]

        print(f"Player {i + 1} is most likely a {level_name} player.")
        print(f"  (Posterior Probability: {probability:.2%})")
