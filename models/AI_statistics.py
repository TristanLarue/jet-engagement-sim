import json
import matplotlib.pyplot as plt
import os

# Path to training stats JSON file
STATS_PATH = os.path.join(os.path.dirname(__file__), "jet_ppo", "training_stats.json")

def load_stats(path):
    if not os.path.exists(path):
        print(f"Stats file not found: {path}")
        print(f"Looking in directory: {os.path.dirname(path)}")
        if os.path.exists(os.path.dirname(path)):
            print(f"Contents: {os.listdir(os.path.dirname(path))}")
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data

def plot_avg_reward(stats):
    updates = [entry["update"] for entry in stats if "update" in entry and "avg_reward" in entry]
    avg_rewards = [entry["avg_reward"] for entry in stats if "update" in entry and "avg_reward" in entry]
    
    if not updates:
        print("No training data found. Make sure the AI has been training and saving stats.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(updates, avg_rewards, marker="o", linestyle="-", color="b")
    plt.title("Evolution of Average Reward per Update")
    plt.xlabel("Update")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    stats = load_stats(STATS_PATH)
    plot_avg_reward(stats)

if __name__ == "__main__":
    main()
