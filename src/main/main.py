import sys
from viz import setup_viz, clean_viz
from simulation import run
import sys

def main(epochs: int = 1, sprint: bool = False, scenario=None) -> int:
    
    if not sprint:
        setup_viz()
    try:
        run(epochs=epochs, sprint=sprint)
        return 0
    except KeyboardInterrupt:
        print("\n===| Program interrupted by user |===")
        return 130
    finally:
        if not sprint:
            clean_viz()

if __name__ == "__main__":
    training = input("Would you like to train the model? (y/n): ").strip().lower() == "y"
    epochAmount = 1
    if training:
        epochAmount = 10000
    sys.exit(main(epochs=epochAmount, sprint=training))