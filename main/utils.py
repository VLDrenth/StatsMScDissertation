import dill
import time
import os
import json
import matplotlib.pyplot as plt



def save_experiment(config, results, experiment_id, base_dir='experiments'):
    """Saves experiment configuration and results to files using dill."""
    # Create a directory for this experiment
    exp_dir = os.path.join(base_dir, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(exp_dir, 'config.pkl')
    with open(config_path, 'wb') as f:
        dill.dump(config, f)

    # Save results
    results_path = os.path.join(exp_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        dill.dump(results, f)

    print(f"Experiment saved in {exp_dir}")
    return exp_dir


def load_experiment(experiment_id, base_dir='experiments'):
    """Loads experiment configuration and results from files using dill."""
    exp_dir = os.path.join(base_dir, experiment_id)
    config_path = os.path.join(exp_dir, 'config.pkl')
    results_path = os.path.join(exp_dir, 'results.pkl')

    if not os.path.exists(config_path) or not os.path.exists(results_path):
        raise FileNotFoundError(f"Experiment files not found for ID: {experiment_id}")

    with open(config_path, 'rb') as f:
        config = dill.load(f)
    with open(results_path, 'rb') as f:
        results = dill.load(f)

    return config, results

def load_experiments(experiment_ids, base_dir='experiments'):
    """Loads multiple experiments given a list of experiment IDs."""
    return [load_experiment(exp_id, base_dir) for exp_id in experiment_ids]

def set_style():
    plt.style.use('seaborn-v0_8-colorblind')
    current_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Create a new color cycle by replacing the 7th color
    new_colors = current_colors.copy()

    color_code = '#8E4585'
    new_colors.append(color_code)

    # Set the new color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=new_colors)
