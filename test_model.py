import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt

def load_and_preprocess_data(file_path='I_OT_O_3ix0zaem.csv'):
    # (Same function as before)
    try:
        df = pd.read_csv(file_path, skiprows=3)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None, None, None, None

    if 'lose' in df.columns:
        df = df.drop(columns=['lose'])

    color_numbers = sorted(df['left_stimuli'].unique())
    color_map = {color: i for i, color in enumerate(color_numbers)}
    
    df['left_stimuli_mapped'] = df['left_stimuli'].map(color_map)
    df['right_stimuli_mapped'] = df['right_stimuli'].map(color_map)
    df['response_mapped'] = df['response'].map(color_map)

    n_trials = df['round_number'].max()
    n_players = df['participant_number'].max()
    n_colors = len(color_numbers)

    player1_data = df[df['participant_number'] == 1].sort_values('round_number').reset_index(drop=True)
    player2_data = df[df['participant_number'] == 2].sort_values('round_number').reset_index(drop=True)

    presented_stimuli = np.zeros((n_players, n_trials, 2), dtype=int)
    presented_stimuli[0, :, 0] = player1_data['left_stimuli_mapped'].values
    presented_stimuli[0, :, 1] = player1_data['right_stimuli_mapped'].values
    presented_stimuli[1, :, 0] = player2_data['left_stimuli_mapped'].values
    presented_stimuli[1, :, 1] = player2_data['right_stimuli_mapped'].values

    observed_choices = np.zeros((n_players, n_trials), dtype=int)
    observed_choices[0, :] = player1_data['response_mapped'].values
    observed_choices[1, :] = player2_data['response_mapped'].values
    
    return n_players, n_trials, n_colors, presented_stimuli, observed_choices

def build_coordination_model(n_players, n_trials, n_colors, presented_stimuli, observed_choices):
    # (Same model definition as before)
    with pm.Model() as coordination_model:
        p_levels = pm.Dirichlet("p_levels", a=np.ones(3))
        player_levels = pm.Categorical("player_levels", p=p_levels, shape=n_players)
        color_hierarchy = pm.Normal("color_hierarchy", mu=0, sigma=1, shape=n_colors)
        learning_rate = pm.Beta("learning_rate", alpha=1, beta=1, shape=n_players)
        convention_strength = pm.Normal("convention_strength", mu=0, sigma=1, shape=n_players)
        
        # Placeholder for the complex likelihood logic.
        @pm.Deterministic(name="choice_probabilities", shape=(n_players, n_trials, n_colors))
        def get_choice_probs(player_levels, color_hierarchy, learning_rate, convention_strength):
            # Complex PyTensor logic for choice probabilities based on levels goes here
            return pt.zeros((n_players, n_trials, n_colors))

        pm.Categorical("observed_choices_model", p=get_choice_probs, observed=observed_choices)
    return coordination_model

if __name__ == "__main__":
    n_players, n_trials, n_colors, presented_stimuli, observed_choices = load_and_preprocess_data()
    
    if n_players is not None:
        model = build_coordination_model(n_players, n_trials, n_colors, presented_stimuli, observed_choices)

        with model:
            print("Starting MCMC sampling...")
            idata = pm.sample(draws=2000, tune=2000, cores=2, return_inferencedata=True)
            print("Sampling complete.")
            
            # Save the InferenceData object to a file
            az.to_netcdf(idata, 'coordination_model_idata.nc')
            print("Model output saved to 'coordination_model_idata.nc'")

