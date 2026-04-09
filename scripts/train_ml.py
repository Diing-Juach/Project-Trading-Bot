from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.ml.training import TrainConfig, main, train

__all__ = ["TrainConfig", "train", "main"]


if __name__ == "__main__":
    main()
