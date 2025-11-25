from viz import initialize_viz, cleanup_viz
from simulation import initialize_simulation
import sys

def initialize_program():
    initialize_viz()
    initialize_simulation()

if __name__ == "__main__":
    try:
        initialize_program()
    except KeyboardInterrupt:
        print("\n===| Program interrupted by user |===")
    except Exception as e:
        print(f"An error occurred: {e}")
    cleanup_viz()
    sys.exit(0)