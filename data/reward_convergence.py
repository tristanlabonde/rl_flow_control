import re
import glob
import matplotlib.pyplot as plt
import numpy as np

def extract_rewards(file_path):
    rewards = []
    pattern = r"reward\s*:\s*([-+]?\d*\.?\d+)"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                rewards.append(float(match.group(1)))
                
    return rewards

file_name = glob.glob("stdout_control.*")[0]
raw_rewards = extract_rewards(file_name)

filtered_rewards = [r if r >= 0 else np.nan for r in raw_rewards]

epochs = list(range(1, len(filtered_rewards) + 1))

plt.figure(figsize=(10, 5))

plt.plot(epochs, filtered_rewards, linestyle='-', color='b', label='Valid reward')

plt.plot(epochs, filtered_rewards, marker='o', color='b', linestyle='None')

plt.title("Convergence of rewards across epochs")
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.xlim(0, 50)

plt.savefig("reward_convergence.png", dpi=300)
plt.show()