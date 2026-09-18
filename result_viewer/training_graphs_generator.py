import re
import glob
import matplotlib.pyplot as plt
import numpy as np

def extract_training(file_path):
    training_data = []
    pattern = "Training with "
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        training_data = content.split(pattern)

    return training_data

def extract_rewards(content):
    rewards = []
    pattern = r"reward\s*:\s*([-+]?\d*\.?\d+)"
    
    for line in content.splitlines():
        match = re.search(pattern, line)
        if match:
            rewards.append(float(match.group(1)))
                
    return rewards

file_name = glob.glob("stdout_control.*")[0]
training_data = extract_training(file_name)[1:]

for content in training_data:
    hyperparameters = [hp.split('=') for hp in content.splitlines()[0].split(',')]
    raw_rewards = extract_rewards(content)

    filtered_rewards = [r if r >= 0 else np.nan for r in raw_rewards]

    epochs = list(range(1, len(filtered_rewards) + 1))

    plt.figure(figsize=(10, 5))

    plt.plot(epochs, filtered_rewards, linestyle='-', color='b', label='Valid reward')

    plt.plot(epochs, filtered_rewards, marker='o', color='b', linestyle='None')

    title = f"Convergence of rewards across epochs, {', '.join(f'{name}={value}' for name, value in hyperparameters)}"
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.xlim(0, 50)

    plt.savefig(f"reward_convergence_{'_'.join(f'{name}_{value}' for name, value in hyperparameters)}.png", dpi=300)
    # plt.show()