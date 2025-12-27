import sys
from viz import setup_viz, clean_viz
from simulation import run

def main(epochs: int = 1, sprint: bool = False) -> int:
    setup_viz()
    try:
        run(epochs=epochs, sprint=sprint)
        return 0
    except KeyboardInterrupt:
        print("\n===| Program interrupted by user |===")
        return 130
    finally:
        clean_viz()

if __name__ == "__main__":
    sys.exit(main())
