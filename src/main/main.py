import sys
import os
import warnings
import logging
import reinforcementlearning as dl


# Set environment variables BEFORE any other imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_WARNINGS"] = "0"

# Suppress all warnings
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from viz import setup_viz, clean_viz
from simulation import run

def main(epochs: int = 1, sprint: bool = False) -> int:
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
    dl.TRAINING = training
    if training:
        epochAmount = 10000
    sys.exit(main(epochs=epochAmount, sprint=training))
    