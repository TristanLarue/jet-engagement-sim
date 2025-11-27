import json
import matplotlib.pyplot as plt
import os

# Path to training stats JSON file
STATS_PATH = os.path.join(os.path.dirname(__file__), "jet_ai_stable", "training_stats.json")

def load_stats(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def plot_avg_reward(stats):
    epochs = [entry["epoch"] for entry in stats if "epoch" in entry and "avg_reward" in entry]
    avg_rewards = [entry["avg_reward"] for entry in stats if "epoch" in entry and "avg_reward" in entry]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_rewards, marker="o", linestyle="-", color="b")
    plt.title("Evolution of Average Reward per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    stats = load_stats(STATS_PATH)
    plot_avg_reward(stats)

if __name__ == "__main__":
    main()
