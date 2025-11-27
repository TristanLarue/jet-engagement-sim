from viz import initialize_viz, cleanup_viz
from simulation import simulate_epochs
import sys

def initialize_program(epochs,sprint):
    initialize_viz()
    simulate_epochs(epochs,sprint)

if __name__ == "__main__":
    epochs = int(input("Input the amount of epochs of simulation to run?: "))
    sprint = input("Would you like the simulation to sprint? (yes/no): ").strip().lower() == "yes"

    if epochs == 0:
        epochs = 1000
    try:
        initialize_program(epochs,sprint)
    except KeyboardInterrupt:
        print("\n===| Program interrupted by user |===")
    except Exception as e:
        print(f"An error occurred: {e}")
    cleanup_viz()
    sys.exit(0)